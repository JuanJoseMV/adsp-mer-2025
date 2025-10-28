import torch
import torch.nn as nn

from .base_fusion_module import BaseFusionModule
from torch import Tensor

class PoolingFusion(nn.Module, BaseFusionModule):
    def __init__(self, embed_dim, projection_dim):
        super(PoolingFusion, self).__init__()
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        
        self.linear = nn.Linear(2 * self.embed_dim, self.projection_dim) # TODO: try variable hidden_dim
        self.relu = nn.ReLU() 
        self.projection = nn.Linear(self.projection_dim, self.projection_dim)

    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the module.
        In this way, the parameters (weights) will not be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text_embeddings, audio_embeddings) -> Tensor:
        """
        Forward pass of the PoolingFusion module. Takes as input the textual and audio embeddings and returns the fused representation.
        The textual and audio embeddings are concatenated and passed through a linear layer followed by a ReLU activation function.
        The output of the ReLU layer is then projected to the output dimension.
        
        Args:
        ----------
        text_embeddings: torch.Tensor
            Textual embeddings of shape (N x d) where N is the number of tokens and d is the embedding dimension.
        audio_embeddings: torch.Tensor
            Audio embeddings of shape (M x d) where M is the number of audio features and d is the embedding dimension.
        
        Returns:
        ----------
        output: torch.Tensor
        Output tensor of shape (d) representing the fused representation of the input embeddings.
            
        """
        text_pooled = torch.mean(text_embeddings, dim=1)
        audio_pooled = torch.mean(audio_embeddings, dim=1)
        
        concatenated = torch.cat((text_pooled, audio_pooled), dim=1)
        
        linear_output = self.linear(concatenated)
        relu_output = self.relu(linear_output)
        projected = self.projection(relu_output)
        
        return projected