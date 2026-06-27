# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Usage: construct the DINOv2 Vision Transformer (ViT) backbone, which Depth Anything V2 uses as its image encoder.

# standard feed-forward inside each transformer layer: Linear → activation (GELU) → Linear, 
#   used in non-SwiGLU variants of the ViT blocks.
from .mlp import Mlp    
# PatchEmbed — converts the input image into a sequence of patch tokens
from .patch_embed import PatchEmbed
# alternative feed-forward blocks using the SwiGLU activation (a gated linear unit variant with Swish/SiLU gating) instead of plain GELU-MLP.
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
# actual transformer block 
from .block import NestedTensorBlock
# a memory-efficient multi-head self-attention implementation, typically backed by xformers' memory-efficient attention kernels
from .attention import MemEffAttention
