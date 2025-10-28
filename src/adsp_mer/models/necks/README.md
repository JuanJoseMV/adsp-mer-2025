# Neck Module

The Neck module in the multimodal model is responsible for projecting the input features. This module is instantiated separately for audio and text features.

## Overview

The Neck module performs the following tasks:
- Projects input audio features
- Projects input text features

Each type of feature (audio and text) has its own instance of the Neck module to handle the projection independently.

## Usage

To use the Neck module, instantiate it separately for audio and text features as needed in your model.

```python
# Example instantiation for audio features
audio_neck = NeckModule(input_dim=audio_input_dim, output_dim=projection_dim)

# Example instantiation for text features
text_neck = NeckModule(input_dim=text_input_dim, output_dim=projection_dim)
```

## Conclusion

The Neck module is a simple yet crucial component of the multimodal model, ensuring that input features are properly projected for further processing.
