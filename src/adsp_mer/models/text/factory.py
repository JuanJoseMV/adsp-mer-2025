from .text_models import (
    BaseTextModel,
    XLMRobertaWrapper,
    MBartWrapper
)
from transformers import PretrainedConfig

class TextModelFactory:
    @staticmethod
    def create_text_model(model_type: str, model_config: PretrainedConfig) -> BaseTextModel:
        """
        Creates and returns a text model based on the provided configuration.

        Args:
            model_type (str): The type of the model to create.
            model_config (PretrainedConfig): The configuration object for the model. 
                It should contain a "model_type" attribute to determine the type of model to create.
        Returns:
            BaseTextModel: An instance of the text model based on the provided configuration.
        Raises:
            ValueError: If the model name specified in the config is not recognized.
        
        """
        if model_type == "xlm-roberta":
            return XLMRobertaWrapper(model_config)
        elif model_type == "mbart":
            return MBartWrapper(model_config)
        else:
            raise ValueError(f"Unknown model name: {model_type}")