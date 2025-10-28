from typing import Protocol
from torch import Tensor

class BaseFusionModule(Protocol):
    """
    Base class for fusion modules. All fusion modules should inherit from this class.

    Supported fusion modules:
    --------------------------
        - PoolingFusion
        - SelfAttentionFusion
        - CrossAttentionFusion

    Attributes:
    -----------
        embed_dim: int
            The dimension of the input embeddings.
        projection_dim: int
            The dimension of the projected embeddings.
        
    Methods:
    --------
        freeze_parameters() -> None:
            Freezes the parameters (weights) of the fusion module.
        forward(*args, **kwargs) -> Tensor:
            Defines the computation performed at every call.

    """
    embed_dim: int
    projection_dim: int

    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the module.
        In this way, the parameters (weights) will not be updated during training.
        """
        ...

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass of the model.
        """
        ...