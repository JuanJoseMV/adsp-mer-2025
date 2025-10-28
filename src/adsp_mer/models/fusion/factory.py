from .fusion_models import (
    BaseFusionModule,
    CrossAttentionFusion,
    SelfAttentionFusion,
    PoolingFusion,
)

class FusionModuleFactory:
    @staticmethod
    def create_fusion_module(fusion_type: str, **kwargs: dict) -> BaseFusionModule:
        """
        Factory function to get the appropriate fusion module based on the fusion type.

        Parameters:
        ----------
        fusion_type (str): The type of the fusion module. Must be one of 'cross_attention', 'pooling', or 'self_attention'.
        **kwargs (dict): Keyword arguments to initialize to the fusion module.

        Returns:
        --------
        BaseFusionModule: An instance of the fusion module based on the fusion type.

        Raises:
        -------
        ValueError: If the fusion_type is unknown.
        """
        if fusion_type == 'cross_attention':
            return CrossAttentionFusion(**kwargs)
        elif fusion_type == 'pooling':
            return PoolingFusion(**kwargs)
        elif fusion_type == 'self_attention':
            return SelfAttentionFusion(**kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")