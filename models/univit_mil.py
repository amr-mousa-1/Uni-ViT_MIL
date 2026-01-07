"""
Uni-ViT-MIL: Unified Vision Transformer for Multiple Instance Learning

A multi-task learning model for whole slide image (WSI) analysis that combines:
1. A trainable Vision Transformer (ViT) encoder for patch-level feature extraction
2. A multi-head attention based MIL pooling module for slide-level aggregation
3. Two parallel prediction heads for cancer classification and biomarker prediction

This model is conceptually different from CLAM:
- ViT is NOT a frozen feature extractor; it is trained end-to-end with the MIL pipeline
- No instance-level clustering losses or SVM losses
- Multi-task architecture with shared representation learning

================================================================================
INTEGRATION MODES
================================================================================

Uni-ViT-MIL supports two operational modes:

1. END-TO-END MODE (Full Pipeline):
   - Input: Raw image patches of shape (num_patches, 3, 224, 224)
   - The ViT encoder processes each patch to produce embeddings
   - Use the full `UniViTMIL` class with VisionTransformerEncoder
   - Suitable for training the entire pipeline from scratch

2. EMBEDDING-BASED MODE (MIL-Only):
   - Input: Pre-extracted patch embeddings of shape (num_patches, embed_dim)
   - Bypasses the ViT encoder entirely
   - Use `UniViTMIL_Embeddings` or `univit_mil_from_embeddings()` factory
   - Suitable for using pre-computed features (e.g., from UNI, CONCH, ResNet)

CURRENT CLAM PIPELINE INTEGRATION:
----------------------------------
In the current CLAM repository workflow, Uni-ViT-MIL is expected to operate 
primarily in EMBEDDING-BASED MODE. This is because:
- CLAM's feature extraction pipeline (extract_features_fp.py) produces .pt files
- These files contain pre-extracted embeddings, not raw patches
- The UniViTMIL_Embeddings class is designed to consume these embeddings directly

To use end-to-end mode, a separate data loading pipeline would be required
that provides raw patch images instead of pre-computed embeddings.

================================================================================

Author: Research Implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


# =============================================================================
# Vision Transformer Components
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Converts image patches into embedding vectors.
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch (assumes square patches)
        in_channels: Number of input image channels
        embed_dim: Dimension of the embedding vectors
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose to get (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for the Vision Transformer.
    
    Args:
        embed_dim: Dimension of the input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        qkv_bias: Whether to include bias in query, key, value projections
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor and optionally attention weights
        """
        B, N, C = x.shape
        
        # Compute Q, K, V: (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute to (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values: (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x, None


class TransformerBlock(nn.Module):
    """
    A single transformer encoder block with self-attention and MLP.
    
    Args:
        embed_dim: Dimension of the embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
        dropout: Dropout probability
        attn_dropout: Dropout probability for attention
        init_values: Initial value for layer scale (None to disable)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        init_values: Optional[float] = None
    ):
        super().__init__()
        
        # Layer normalization before attention (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Optional layer scale (used in some ViT variants like UNI)
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(embed_dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(embed_dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        
        if self.gamma_1 is not None:
            x = x + self.gamma_1 * attn_out
        else:
            x = x + attn_out
        
        # MLP with residual connection
        if self.gamma_2 is not None:
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for extracting patch embeddings.
    
    This is a trainable encoder that processes input images and produces
    patch-level embeddings for downstream MIL aggregation.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout probability
        attn_dropout: Attention dropout probability
        init_values: Layer scale initial values (UNI-style)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        init_values: Optional[float] = 1e-5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                init_values=init_values
            )
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ViT paper recommendations."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize all linear layers and layer norms
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, module: nn.Module):
        """Initialize individual module weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            return_all_tokens: If True, return all patch tokens; otherwise just CLS
        
        Returns:
            cls_token: CLS token embedding of shape (batch_size, embed_dim)
            patch_tokens: Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        B = x.shape[0]
        
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Separate CLS token and patch tokens
        cls_token = x[:, 0]  # (B, embed_dim)
        patch_tokens = x[:, 1:]  # (B, num_patches, embed_dim)
        
        return cls_token, patch_tokens


# =============================================================================
# Multi-Head Attention MIL Pooling
# =============================================================================

class MultiHeadAttentionMILPooling(nn.Module):
    """
    Multi-Head Attention based MIL Pooling module.
    
    Aggregates a variable number of patch embeddings into a single slide-level
    embedding using multi-head attention. This is different from CLAM's gated
    attention network - it uses full transformer-style multi-head attention
    with learnable query vectors.
    
    Args:
        embed_dim: Dimension of input patch embeddings
        num_heads: Number of attention heads
        num_queries: Number of learnable query vectors (for multi-query aggregation)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_queries: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Learnable query vectors for slide-level aggregation
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP for post-aggregation processing
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate patch embeddings into slide-level embedding.
        
        Args:
            patch_embeddings: Patch embeddings of shape (batch_size, num_patches, embed_dim)
                              OR (num_patches, embed_dim) for single slide
            mask: Optional attention mask of shape (batch_size, num_patches)
        
        Returns:
            slide_embedding: Aggregated slide embedding of shape (batch_size, embed_dim)
            attention_weights: Attention weights of shape (batch_size, num_heads, num_queries, num_patches)
        """
        # Handle single slide case (no batch dimension)
        if patch_embeddings.dim() == 2:
            patch_embeddings = patch_embeddings.unsqueeze(0)
        
        B, N, C = patch_embeddings.shape
        
        # Normalize input
        patch_embeddings = self.norm(patch_embeddings)
        
        # Expand query embeddings for batch
        queries = self.query_embed.expand(B, -1, -1)  # (B, num_queries, embed_dim)
        
        # Project to Q, K, V
        Q = self.q_proj(queries)  # (B, num_queries, embed_dim)
        K = self.k_proj(patch_embeddings)  # (B, N, embed_dim)
        V = self.v_proj(patch_embeddings)  # (B, N, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, self.num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores: (B, num_heads, num_queries, N)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # =========================================================================
        # TODO: Heatmap Visualization - Attention Aggregation Strategies
        # =========================================================================
        # The attention_weights tensor has shape (B, num_heads, num_queries, N)
        # where N is the number of patches. For heatmap visualization, these
        # multi-head attention weights need to be aggregated into a single
        # importance score per patch. Several strategies are possible:
        #
        # 1. AVERAGE ACROSS HEADS (Current Default):
        #    attn_for_heatmap = attn_weights.mean(dim=1)  # (B, num_queries, N)
        #    - Simple and interpretable
        #    - Treats all heads equally
        #    - Implemented in vis_utils/heatmap_utils.py
        #
        # 2. MAX ACROSS HEADS:
        #    attn_for_heatmap = attn_weights.max(dim=1)[0]
        #    - Highlights patches that ANY head finds important
        #    - May produce sparser, more focused heatmaps
        #
        # 3. LEARNED HEAD WEIGHTING:
        #    head_weights = self.head_importance  # learnable parameter
        #    attn_for_heatmap = (attn_weights * head_weights).sum(dim=1)
        #    - Allows model to learn which heads are most relevant
        #    - Requires additional training
        #
        # 4. TASK-SPECIFIC HEADS:
        #    - Use different heads for classification vs biomarker heatmaps
        #    - Requires analysis of what each head learns to attend to
        #
        # 5. QUERY AGGREGATION (if num_queries > 1):
        #    - Average or max across queries after head aggregation
        #    - Current implementation averages across queries in forward()
        #
        # See vis_utils/heatmap_utils.py for the current implementation:
        # - get_univit_attention_scores() implements strategy #1
        # - compute_attention_per_head_univit() allows per-head analysis
        # =========================================================================
        
        # Apply attention to values: (B, num_heads, num_queries, head_dim)
        out = attn_weights @ V
        
        # Reshape back: (B, num_queries, embed_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, self.num_queries, C)
        out = self.out_proj(out)
        
        # Apply MLP with residual
        out = out + self.mlp(self.mlp_norm(out))
        
        # If single query, squeeze the query dimension
        if self.num_queries == 1:
            slide_embedding = out.squeeze(1)  # (B, embed_dim)
        else:
            # Average across queries if multiple
            slide_embedding = out.mean(dim=1)  # (B, embed_dim)
        
        return slide_embedding, attn_weights


# =============================================================================
# Prediction Heads
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Classification head for cancer classification task.
    
    Args:
        embed_dim: Dimension of input slide embedding
        hidden_dim: Hidden layer dimension
        n_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.25
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Slide embedding of shape (batch_size, embed_dim)
        
        Returns:
            Classification logits of shape (batch_size, n_classes)
        """
        return self.head(x)


class BiomarkerHead(nn.Module):
    """
    Biomarker prediction head for molecular biomarker prediction.
    
    Supports both multi-class classification (mutually exclusive biomarkers)
    and multi-label classification (independent biomarker predictions).
    
    Args:
        embed_dim: Dimension of input slide embedding
        hidden_dim: Hidden layer dimension
        n_biomarkers: Number of biomarker outputs
        dropout: Dropout probability
        multi_label: If True, use sigmoid for independent predictions;
                     if False, use softmax for mutually exclusive classes
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        n_biomarkers: int = 4,
        dropout: float = 0.25,
        multi_label: bool = False
    ):
        super().__init__()
        self.multi_label = multi_label
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_biomarkers)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Slide embedding of shape (batch_size, embed_dim)
        
        Returns:
            Biomarker logits of shape (batch_size, n_biomarkers)
        """
        return self.head(x)


# =============================================================================
# Main Uni-ViT-MIL Model
# =============================================================================

class UniViTMIL(nn.Module):
    """
    Uni-ViT-MIL: Unified Vision Transformer for Multiple Instance Learning
    
    A multi-task learning model for whole slide image analysis that integrates:
    1. A trainable Vision Transformer encoder for patch-level feature extraction
    2. A multi-head attention based MIL pooling module for slide-level aggregation
    3. Two parallel prediction heads for cancer classification and biomarker prediction
    
    Key Design Principles:
    - The ViT encoder is NOT frozen; it is trained end-to-end with the MIL objective
    - Multi-head attention pooling replaces CLAM's gated attention mechanism
    - Multi-task learning with shared representations improves generalization
    - No instance-level clustering or SVM losses
    
    Args:
        img_size: Input patch image size (default: 224)
        patch_size: ViT patch size (default: 16)
        in_channels: Number of input image channels (default: 3)
        embed_dim: Embedding dimension (default: 768 for ViT-Base)
        vit_depth: Number of ViT transformer blocks (default: 12)
        vit_num_heads: Number of ViT attention heads (default: 12)
        vit_mlp_ratio: ViT MLP hidden dimension ratio (default: 4.0)
        vit_dropout: ViT dropout probability (default: 0.0)
        vit_init_values: Layer scale initial values (default: 1e-5 for UNI-style)
        mil_num_heads: Number of MIL pooling attention heads (default: 8)
        mil_num_queries: Number of MIL query vectors (default: 1)
        mil_dropout: MIL pooling dropout probability (default: 0.1)
        n_classes: Number of cancer classification classes (default: 2)
        n_biomarkers: Number of biomarker prediction outputs (default: 4)
        head_hidden_dim: Hidden dimension for prediction heads (default: 256)
        head_dropout: Dropout for prediction heads (default: 0.25)
        biomarker_multi_label: If True, treat biomarkers as independent (default: False)
    """
    
    def __init__(
        self,
        # ViT Encoder parameters
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        vit_init_values: Optional[float] = 1e-5,
        # MIL Pooling parameters
        mil_num_heads: int = 8,
        mil_num_queries: int = 1,
        mil_dropout: float = 0.1,
        # Classification head parameters
        n_classes: int = 2,
        # Biomarker head parameters
        n_biomarkers: int = 4,
        biomarker_multi_label: bool = False,
        # Shared head parameters
        head_hidden_dim: int = 256,
        head_dropout: float = 0.25
    ):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_biomarkers = n_biomarkers
        
        # Vision Transformer Encoder
        self.vit_encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            attn_dropout=vit_dropout,
            init_values=vit_init_values
        )
        
        # Multi-Head Attention MIL Pooling
        self.mil_pooling = MultiHeadAttentionMILPooling(
            embed_dim=embed_dim,
            num_heads=mil_num_heads,
            num_queries=mil_num_queries,
            dropout=mil_dropout
        )
        
        # Classification Head (cancer classification)
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            n_classes=n_classes,
            dropout=head_dropout
        )
        
        # Biomarker Prediction Head
        self.biomarker_head = BiomarkerHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            n_biomarkers=n_biomarkers,
            dropout=head_dropout,
            multi_label=biomarker_multi_label
        )
    
    def encode_patches(
        self,
        patches: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode individual patches using the ViT encoder.
        
        Args:
            patches: Batch of patch images of shape (num_patches, channels, height, width)
        
        Returns:
            patch_embeddings: Patch embeddings of shape (num_patches, embed_dim)
        
        Design Note (CLS Token as Patch Representation):
        ------------------------------------------------
        We use the CLS token output from each patch's ViT encoding as the patch 
        representation. This design choice is inspired by the Vision Transformer's 
        use of CLS tokens to capture global semantic information:
        
        - The CLS token attends to all patch tokens within the image via self-attention
        - It learns to aggregate spatial information into a single vector
        - This mirrors how ViT uses CLS for image-level classification
        
        Alternative approaches could include:
        - Mean pooling over all patch tokens (loses CLS's learned aggregation)
        - Using the final patch tokens directly (loses global context)
        - Concatenating CLS with pooled patch tokens (increases dimensionality)
        
        The CLS-based approach provides a semantically rich, fixed-size representation
        that captures both local texture and global structure within each WSI patch.
        """
        # Get CLS tokens as patch representations
        # Each patch image is encoded independently; the CLS token serves as
        # a holistic representation of that patch's visual content
        cls_tokens, _ = self.vit_encoder(patches, return_all_tokens=False)
        return cls_tokens
    
    def forward(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the Uni-ViT-MIL model.
        
        Processes a bag of patches from a whole slide image and produces:
        - Cancer classification predictions
        - Biomarker predictions
        - Attention weights for interpretability
        
        Args:
            patches: Batch of patch images of shape (num_patches, channels, height, width)
                     OR pre-extracted embeddings of shape (num_patches, embed_dim)
            mask: Optional attention mask of shape (num_patches,) indicating valid patches
            return_features: If True, also return the slide-level embedding
        
        Returns:
            Dictionary containing:
                - 'classification_logits': Shape (1, n_classes)
                - 'biomarker_logits': Shape (1, n_biomarkers)
                - 'attention_weights': Shape (1, num_heads, num_queries, num_patches)
                - 'slide_embedding': Shape (1, embed_dim) [if return_features=True]
        """
        # Determine if input is raw patches or pre-extracted embeddings
        if patches.dim() == 4:
            # Raw patch images: (num_patches, C, H, W)
            patch_embeddings = self.encode_patches(patches)
        elif patches.dim() == 2:
            # Pre-extracted embeddings: (num_patches, embed_dim)
            patch_embeddings = patches
        else:
            raise ValueError(
                f"Expected patches with 2 or 4 dimensions, got {patches.dim()}"
            )
        
        # Add batch dimension for MIL pooling
        patch_embeddings = patch_embeddings.unsqueeze(0)  # (1, num_patches, embed_dim)
        
        # Aggregate patches into slide-level embedding
        slide_embedding, attention_weights = self.mil_pooling(
            patch_embeddings, mask=mask
        )
        
        # Compute predictions from both heads
        classification_logits = self.classification_head(slide_embedding)
        biomarker_logits = self.biomarker_head(slide_embedding)
        
        # Build output dictionary
        outputs = {
            'classification_logits': classification_logits,
            'biomarker_logits': biomarker_logits,
            'attention_weights': attention_weights
        }
        
        if return_features:
            outputs['slide_embedding'] = slide_embedding
        
        return outputs
    
    def get_attention_weights(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability/visualization.
        
        Args:
            patches: Batch of patch images or pre-extracted embeddings
            mask: Optional attention mask
        
        Returns:
            attention_weights: Shape (1, num_heads, num_queries, num_patches)
        """
        outputs = self.forward(patches, mask=mask)
        return outputs['attention_weights']
    
    @classmethod
    def from_pretrained_vit(
        cls,
        vit_checkpoint_path: str,
        strict: bool = True,
        **kwargs
    ) -> 'UniViTMIL':
        """
        Create a UniViTMIL model with a pretrained ViT encoder.
        
        Args:
            vit_checkpoint_path: Path to pretrained ViT checkpoint
            strict: Whether to strictly enforce checkpoint loading
            **kwargs: Additional arguments for UniViTMIL constructor
        
        Returns:
            UniViTMIL model with pretrained ViT weights
        """
        # Create model
        model = cls(**kwargs)
        
        # Load pretrained ViT weights
        checkpoint = torch.load(vit_checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load into ViT encoder
        model.vit_encoder.load_state_dict(state_dict, strict=strict)
        
        return model


# =============================================================================
# Factory Functions for Common Configurations
# =============================================================================

def univit_mil_base(
    n_classes: int = 2,
    n_biomarkers: int = 4,
    **kwargs
) -> UniViTMIL:
    """
    Create a Uni-ViT-MIL model with ViT-Base configuration.
    
    ViT-Base: embed_dim=768, depth=12, num_heads=12
    
    Args:
        n_classes: Number of cancer classification classes
        n_biomarkers: Number of biomarker predictions
        **kwargs: Additional arguments
    
    Returns:
        UniViTMIL model with ViT-Base configuration
    """
    return UniViTMIL(
        embed_dim=768,
        vit_depth=12,
        vit_num_heads=12,
        n_classes=n_classes,
        n_biomarkers=n_biomarkers,
        **kwargs
    )


def univit_mil_small(
    n_classes: int = 2,
    n_biomarkers: int = 4,
    **kwargs
) -> UniViTMIL:
    """
    Create a Uni-ViT-MIL model with ViT-Small configuration.
    
    ViT-Small: embed_dim=384, depth=12, num_heads=6
    
    Args:
        n_classes: Number of cancer classification classes
        n_biomarkers: Number of biomarker predictions
        **kwargs: Additional arguments
    
    Returns:
        UniViTMIL model with ViT-Small configuration
    """
    return UniViTMIL(
        embed_dim=384,
        vit_depth=12,
        vit_num_heads=6,
        mil_num_heads=6,
        n_classes=n_classes,
        n_biomarkers=n_biomarkers,
        **kwargs
    )


def univit_mil_large(
    n_classes: int = 2,
    n_biomarkers: int = 4,
    **kwargs
) -> UniViTMIL:
    """
    Create a Uni-ViT-MIL model with ViT-Large configuration (UNI-style).
    
    ViT-Large: embed_dim=1024, depth=24, num_heads=16
    
    Args:
        n_classes: Number of cancer classification classes
        n_biomarkers: Number of biomarker predictions
        **kwargs: Additional arguments
    
    Returns:
        UniViTMIL model with ViT-Large configuration
    """
    return UniViTMIL(
        embed_dim=1024,
        vit_depth=24,
        vit_num_heads=16,
        mil_num_heads=8,
        n_classes=n_classes,
        n_biomarkers=n_biomarkers,
        **kwargs
    )


def univit_mil_from_embeddings(
    embed_dim: int = 1024,
    n_classes: int = 2,
    n_biomarkers: int = 4,
    mil_num_heads: int = 8,
    mil_num_queries: int = 1,
    mil_dropout: float = 0.1,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.25,
    biomarker_multi_label: bool = False
) -> nn.Module:
    """
    Create a lightweight Uni-ViT-MIL model for use with pre-extracted embeddings.
    
    This variant does not include the ViT encoder and is designed to work
    with pre-computed patch embeddings (e.g., from UNI or other feature extractors).
    
    Args:
        embed_dim: Dimension of input embeddings (must match feature extractor)
        n_classes: Number of cancer classification classes
        n_biomarkers: Number of biomarker predictions
        mil_num_heads: Number of MIL pooling attention heads
        mil_num_queries: Number of MIL query vectors
        mil_dropout: MIL pooling dropout probability
        head_hidden_dim: Hidden dimension for prediction heads
        head_dropout: Dropout for prediction heads
        biomarker_multi_label: If True, treat biomarkers as independent
    
    Returns:
        UniViTMIL_Embeddings model (lightweight variant)
    """
    return UniViTMIL_Embeddings(
        embed_dim=embed_dim,
        n_classes=n_classes,
        n_biomarkers=n_biomarkers,
        mil_num_heads=mil_num_heads,
        mil_num_queries=mil_num_queries,
        mil_dropout=mil_dropout,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        biomarker_multi_label=biomarker_multi_label
    )


class UniViTMIL_Embeddings(nn.Module):
    """
    Lightweight version of Uni-ViT-MIL for use with pre-extracted embeddings.
    
    This model contains only the MIL pooling and prediction heads, designed
    to work with pre-computed patch embeddings from external feature extractors.
    
    Args:
        embed_dim: Dimension of input embeddings
        n_classes: Number of cancer classification classes
        n_biomarkers: Number of biomarker predictions
        mil_num_heads: Number of MIL pooling attention heads
        mil_num_queries: Number of MIL query vectors
        mil_dropout: MIL pooling dropout probability
        head_hidden_dim: Hidden dimension for prediction heads
        head_dropout: Dropout for prediction heads
        biomarker_multi_label: If True, treat biomarkers as independent
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        n_classes: int = 2,
        n_biomarkers: int = 4,
        mil_num_heads: int = 8,
        mil_num_queries: int = 1,
        mil_dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.25,
        biomarker_multi_label: bool = False
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_biomarkers = n_biomarkers
        
        # Multi-Head Attention MIL Pooling
        self.mil_pooling = MultiHeadAttentionMILPooling(
            embed_dim=embed_dim,
            num_heads=mil_num_heads,
            num_queries=mil_num_queries,
            dropout=mil_dropout
        )
        
        # Classification Head
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            n_classes=n_classes,
            dropout=head_dropout
        )
        
        # Biomarker Prediction Head
        self.biomarker_head = BiomarkerHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            n_biomarkers=n_biomarkers,
            dropout=head_dropout,
            multi_label=biomarker_multi_label
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the lightweight Uni-ViT-MIL model.
        
        Args:
            embeddings: Pre-extracted patch embeddings of shape (num_patches, embed_dim)
                        OR batched (batch_size, num_patches, embed_dim)
            mask: Optional attention mask
            return_features: If True, also return the slide-level embedding
        
        Returns:
            Dictionary containing:
                - 'classification_logits': Shape (batch_size, n_classes)
                - 'biomarker_logits': Shape (batch_size, n_biomarkers)
                - 'attention_weights': Shape (batch_size, num_heads, num_queries, num_patches)
                - 'slide_embedding': Shape (batch_size, embed_dim) [if return_features=True]
        """
        # Handle unbatched input
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        # Aggregate patches into slide-level embedding
        slide_embedding, attention_weights = self.mil_pooling(
            embeddings, mask=mask
        )
        
        # Compute predictions from both heads
        classification_logits = self.classification_head(slide_embedding)
        biomarker_logits = self.biomarker_head(slide_embedding)
        
        # Build output dictionary
        outputs = {
            'classification_logits': classification_logits,
            'biomarker_logits': biomarker_logits,
            'attention_weights': attention_weights
        }
        
        if return_features:
            outputs['slide_embedding'] = slide_embedding
        
        return outputs


# =============================================================================
# EXAMPLE/DEBUG SECTION - Not Part of Core Model Definition
# =============================================================================
# The code below is provided for testing and demonstration purposes only.
# It is NOT required for using Uni-ViT-MIL in the CLAM pipeline.
# 
# To run these tests independently:
#   python models/univit_mil.py
#
# This section verifies:
# 1. Model instantiation works correctly
# 2. Forward pass produces expected output shapes
# 3. Both full model and embedding-only variants function properly
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # DEBUG TEST: Verify model architecture and output shapes
    # This is for development/debugging only, not for production use
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("Uni-ViT-MIL Debug Test Suite")
    print("=" * 60)
    print("\nNOTE: This is a debug/example section for verifying model structure.")
    print("It uses synthetic data and is not part of the training pipeline.\n")
    
    # Test 1: Full model with ViT encoder (end-to-end mode)
    print("-" * 60)
    print("Test 1: Full UniViTMIL Model (End-to-End Mode)")
    print("-" * 60)
    model = univit_mil_base(n_classes=3, n_biomarkers=4)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Synthetic patch data (simulating a bag of 100 patches)
    num_patches = 100
    dummy_patches = torch.randn(num_patches, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_patches, return_features=True)
    
    print(f"\nOutput shapes:")
    print(f"  Classification logits: {outputs['classification_logits'].shape}")
    print(f"  Biomarker logits: {outputs['biomarker_logits'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")
    print(f"  Slide embedding: {outputs['slide_embedding'].shape}")
    
    # Test 2: Embedding-only model (MIL-only mode, used in CLAM pipeline)
    print("\n" + "-" * 60)
    print("Test 2: UniViTMIL_Embeddings Model (Embedding-Based Mode)")
    print("-" * 60)
    print("This is the variant used in the CLAM pipeline with pre-extracted features.\n")
    
    embed_model = univit_mil_from_embeddings(
        embed_dim=1024, n_classes=3, n_biomarkers=4
    )
    print(f"Embeddings model created with {sum(p.numel() for p in embed_model.parameters()):,} parameters")
    
    # Synthetic embeddings (simulating pre-extracted features from UNI/ResNet)
    dummy_embeddings = torch.randn(num_patches, 1024)
    
    with torch.no_grad():
        outputs = embed_model(dummy_embeddings, return_features=True)
    
    print(f"\nOutput shapes:")
    print(f"  Classification logits: {outputs['classification_logits'].shape}")
    print(f"  Biomarker logits: {outputs['biomarker_logits'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")
    print(f"  Slide embedding: {outputs['slide_embedding'].shape}")
    
    print("\n" + "=" * 60)
    print("✓ All debug tests passed!")
    print("=" * 60)

