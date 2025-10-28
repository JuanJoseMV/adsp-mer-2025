import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F

from it_multimodal_er.training import get_loss_function
from transformers.utils import ModelOutput
from torch import Tensor
# from huggingface_hub import PyTorchModelHubMixin
from it_multimodal_er.configs import ModelConfig
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
)
from it_multimodal_er.models import (
    AudioModelFactory,
    TextModelFactory,
    Neck,
    FusionModuleFactory,
    ClassifierFactory
)

# class MultiModalModelForClassification(nn.Module, PyTorchModelHubMixin):
class MultiModalModelForClassification(PreTrainedModel):
    # TODO: use autoconfig class
    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        # super(MultiModalModelForClassification, self).__init__()
        super(MultiModalModelForClassification, self).__init__(config)
        self.config = config
        self.text_model_config = AutoConfig.from_pretrained(config.text_model_name)
        self.text_model = TextModelFactory.create_text_model(
            model_type=config.text_model_type,
            model_config=self.text_model_config
        )
        self.text_model.freeze_parameters()

        # self.audio_model_config = AutoConfig.from_pretrained(config.audio_model_name)
        self.audio_model = AudioModelFactory.create_audio_model(
            model_type=config.audio_model_type,
            model_config=config
        )
        # self.audio_model.freeze_feature_encoder()
        # self.audio_model.freeze_parameters()

        num_layers = self.audio_model.backbone_model.config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.audio_weights = nn.Parameter(torch.ones(num_layers)/num_layers)

        self.text_neck = Neck(
            self.text_model_config.hidden_size,
            config.neck_dim,
        ) 
        self.audio_neck = Neck(
            self.audio_model.backbone_model.config.hidden_size,
            config.neck_dim,
        )

        self.fusion = FusionModuleFactory.create_fusion_module(
            embed_dim=config.neck_dim, 
            **config.fusion_kwargs
        )
        
        self.classifier = nn.Linear(self.fusion.projection_dim, config.num_classes)

        label_weights_tensor = Tensor(config.label_weights)
        self.loss_fnct = get_loss_function(label_weights_tensor)

    def forward(
            self, 
            input_features: Tensor, 
            # audio_attention_mask: Tensor,
            input_ids: Tensor, 
            text_attention_mask: Tensor,
            labels: Tensor = None
        ) -> ModelOutput:
        text_output = self.text_model(input_ids=input_ids, attention_mask=text_attention_mask).last_hidden_state
        audio_output = self.audio_model(
            input_features=input_features, 
            # attention_mask=audio_attention_mask
        ).encoder_hidden_states

        ###AUDIO POOLING###
        # 1. stacked feature
        stacked_feature = torch.stack(audio_output, dim=0)
        
        # 2. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        stacked_feature = stacked_feature.view(self.audio_model.backbone_model.config.num_hidden_layers + 1, -1)
        
        norm_weights = F.softmax(self.audio_weights, dim=-1)

        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        pooled_audio_output = weighted_feature.view(*origin_shape)
        ####

        projected_text_output = self.text_neck(text_output)
        projected_audio_output = self.audio_neck(pooled_audio_output)

        fusion_output = self.fusion(
            text_embeddings=projected_text_output, 
            audio_embeddings=projected_audio_output
        )

        logits = self.classifier(fusion_output)

        loss = None
        if labels is not None:
            loss = self.loss_fnct(logits, labels) #TODO: create loss file

        return ModelOutput(
            loss=loss,
            logits=logits
        )
