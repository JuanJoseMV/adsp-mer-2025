from it_multimodal_er.preprocessing import MultiModalPreprocessor
from it_multimodal_er.preprocessing import AudioPreprocessor
from it_multimodal_er.preprocessing import TextPreprocessor
from it_multimodal_er.preprocessing import PreprocessorConfig

class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(preprocessor_config: PreprocessorConfig):
        """
        Factory function to get the appropriate preprocessor based on the type.

        Parameters:
            preprocessor_config (PreprocessorConfig): Configuration object for the preprocessor.
        Returns:
            Preprocessor: An instance of the appropriate preprocessor class.
        Raises:
            ValueError: If the preprocessor_type is unknown.              
        """

        if preprocessor_config.preprocessor_type == "audio":
            return AudioPreprocessor(preprocessor_config)
        elif preprocessor_config.preprocessor_type == "text":
            return TextPreprocessor(preprocessor_config)
        elif preprocessor_config.preprocessor_type == "multimodal":
            return MultiModalPreprocessor(preprocessor_config)
        else:
            error_message = f"Unknown preprocessor type: {preprocessor_config.preprocessor_type}"
            raise ValueError(error_message)