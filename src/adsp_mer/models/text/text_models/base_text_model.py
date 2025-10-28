from transformers.utils import ModelOutput
from typing import Protocol

class BaseTextModel(Protocol):
    """
    Base class for text models. All text models should inherit from this class.

    Supported text models:
    --------------------------
        - XLMRoberta
        - MBart
        
    Methods:
    --------
        freeze_parameters() -> None:
            Freezes the parameters (weights) of the text model.
        freeze_feature_encoder() -> None:
            Freezes the feature encoder of the text model.
        forward(*args, **kwargs) -> ModelOutput:
            Defines the computation performed at every call.

    """
    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the model.
        In this way, the parameters (weights) will not be updated during training.
        """
        ...

    def forward(self, *args, **kwargs) -> ModelOutput:
        """
        Forward pass of the model.
        """
        ...