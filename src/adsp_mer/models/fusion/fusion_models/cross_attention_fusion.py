import torch
import torch.nn as nn

from .base_fusion_module import BaseFusionModule
from torch import Tensor

class CrossAttentionFusion(nn.Module, BaseFusionModule):
    def __init__(self, embed_dim, num_heads, projection_dim):
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.linear = nn.Linear(self.embed_dim, self.projection_dim)

    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the module.
        In this way, the parameters (weights) will not be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text_embeddings, audio_embeddings) -> Tensor:
        """
        Forward pass of the CrossAttentionFusion module. Takes as input the textual and audio embeddings and returns the fused representation.
        The textual and audio embeddings are passed through a multihead attention layer.
        The output of the attention layer is then projected to the output dimension.

        Args:
        ----------
        text_embeddings: Tensor
            Textual embeddings of shape (N x d) where N is the number of tokens and d is the embedding dimension.
        audio_embeddings: Tensor
            Audio embeddings of shape (M x d) where M is the number of audio features and d is the embedding dimension.

        Returns:
        ----------
        output: Tensor
            Output tensor of shape (d) representing the fused representation of the input embeddings.
        
        """
        attn_output, _ = self.multihead_attn(
            query=audio_embeddings, 
            key=text_embeddings, 
            value=text_embeddings  
        ) # TODO: Bi-directional attention

        projection = self.linear(attn_output)
        output = torch.mean(projection, dim=1)

        return output