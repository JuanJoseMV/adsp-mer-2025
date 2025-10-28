# Fusion Module of the Multimodal Model

## Overview

The Fusion Module is a critical component of the multimodal model, responsible for integrating information from multiple modalities to improve the overall performance of the model. This module combines data from text and audio sources to create a unified representation that can be used for various downstream tasks.

There are three possible implementations for this module:

1. **Pooling-based**: This approach uses average pooling to combine features from different modalities (1xd and 1xd shaped). The pooled features are concatenated into a single representation (1,2xd shape), which is then passed through an MLP consisting of Linear + ReLU + Linear layers. The final output is projected into a different space of dimension "projection_dim".

2. **Self-attention-based**: This method concatenates the representations coming from text (Nxd shape) and audio (Mxd shape) to create a single representation ((N+M)xd shape). It then passes this concatenated representation through a self-attention mechanism to capture intra-modality relationships, followed by a projection and average pooling to create the final unified representation.

3. **Cross-attention-based**: 
This technique also aims to capture relationships between modalities but does so by separately passing the text and audio features to a cross-attention mechanism. In this setup, the text features serve as the query and values, while the audio features act as the key. This allows the model to learn how to align and integrate information from the two different modalities more effectively. The output of the cross-attention mechanism is then projected and average pooled to form the final unified representation.


## Configuration

The Fusion Module can be configured using a configuration file or by passing parameters directly to the module. Here is an example configuration:

```json
    {
        "fusion_kwargs": {
            "fusion_type": "pooling",
            "projection_dim": 256
        },
    }
```