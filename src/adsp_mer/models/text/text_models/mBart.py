import torch
import torch.nn as nn

from .base_text_model import BaseTextModel
from transformers.utils import ModelOutput
from torch import Tensor
from transformers import (
    PreTrainedModel,
    MBartModel,
    MBartConfig,
    set_seed
)

set_seed(42)

class MBartWrapper(PreTrainedModel, BaseTextModel):
    def __init__(self, config: MBartConfig):
        super(MBartWrapper, self).__init__(config)
        self.mbart = MBartModel(config)

        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the model.
        In this way, the parameters (weights) will not be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> ModelOutput:
        """
        Forward pass of the model.

        Parameters:
        ----------
            input_ids (torch.Tensor): The input features of shape (B, M).
            attention_mask (torch.Tensor): The attention mask of shape (B, M).

            Where:
                B: Batch size.
                M: Number of token ids.

        Returns:
        --------
            ModelOutput: The output of the model containing the hidden states, and the loss (for finetuning).
            The hidden states are the weighted sum of the hidden states of the transformer layers with shape (B, N, D).

            Where:
                B: Batch size.
                M: Number of token ids.
        
        """
        outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states, dim=1)
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1) # (MxD)

        return ModelOutput(
            hidden_states=hidden_states,
        )