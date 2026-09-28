"""
Model definitions for SonoRank paper.

Contains the three-level SonoRank pipeline models and baseline architectures:
  - Level 1: EmbeddingAutoencoder (encoder pretraining)
  - Level 2: MultiFingerTemporalRankingModel (multi-finger ranking backbone)
  - Level 3: FlexEmbeddingClassifier (downstream flex classification)
  - Baselines: CBMFModel (Zadok et al. 2025), BimbrawVGG (Bimbraw et al. 2023)

Reader's guide: the file follows the experimental pipeline. Shared residual
blocks and the autoencoder appear first, followed by the ranking backbone and
the classifier that reuses it. Baseline architectures are isolated afterward.
Checkpoint compatibility depends on preserving class names and tensor shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    """Count trainable and total parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class ResBlock(nn.Module):
    """
    Residual block with two convolutional layers.

    Architecture: Conv-BN-ReLU-Conv-BN + skip connection
    """

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input and output channels (must be same for residual)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class EmbeddingAutoencoder(nn.Module):
    """
    Convolutional autoencoder for learning image embeddings.

    Compresses 224x224 RGB images to a compact embedding vector and reconstructs them.
    Uses ResNet-style architecture with residual blocks for stable training.

    Architecture:
        Encoder: 224x224x3 → 64-dim embedding
        Decoder: 64-dim embedding → 224x224x3

    Key design choices:
        - No skip connections (forces bottleneck to learn full representation)
        - ResBlocks for stable deep training
        - Upsample + Conv instead of TransposedConv (avoids checkerboard artifacts)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        in_channels: int = 3,
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding vector (default: 64)
            base_channels: Number of channels in first conv layer (default: 64)
            in_channels: Number of input image channels (default: 3 for RGB)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.in_channels = in_channels

        # Channel progression: 64 → 128 → 256 → 512 → 512
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        # =========================
        # Encoder
        # =========================
        # 224x224x3 → 112x112x64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.enc_res1 = ResBlock(c1)

        # 112x112x64 → 56x56x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.enc_res2 = ResBlock(c2)

        # 56x56x128 → 28x28x256
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_res3 = ResBlock(c3)

        # 28x28x256 → 14x14x512
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.enc_res4 = ResBlock(c4)

        # 14x14x512 → 7x7x512
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 7x7x512 → 1x1x512
        self.enc_pool = nn.AdaptiveAvgPool2d(1)

        # FC layer: 512 → embedding_dim
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # =========================
        # Decoder
        # =========================
        # embedding_dim → 7x7x512
        self.dec_fc = nn.Linear(embedding_dim, c5 * 7 * 7)

        # 7x7x512 → 14x14x512
        self.dec_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c5, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.dec_res1 = ResBlock(c4)

        # 14x14x512 → 28x28x256
        self.dec_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c4, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.dec_res2 = ResBlock(c3)

        # 28x28x256 → 56x56x128
        self.dec_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.dec_res3 = ResBlock(c2)

        # 56x56x128 → 112x112x64
        self.dec_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.dec_res4 = ResBlock(c1)

        # 112x112x64 → 224x224x3
        self.dec_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to embedding vector.

        Args:
            x: Input image tensor [batch, 3, 224, 224]

        Returns:
            Embedding tensor [batch, embedding_dim]
        """
        # Encoder forward pass
        x = self.enc_conv1(x)
        x = self.enc_res1(x)

        x = self.enc_conv2(x)
        x = self.enc_res2(x)

        x = self.enc_conv3(x)
        x = self.enc_res3(x)

        x = self.enc_conv4(x)
        x = self.enc_res4(x)

        x = self.enc_conv5(x)

        # Global pooling and FC
        x = self.enc_pool(x)
        x = x.flatten(1)
        z = self.enc_fc(x)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding vector to reconstructed image.

        Args:
            z: Embedding tensor [batch, embedding_dim]

        Returns:
            Reconstructed image tensor [batch, 3, 224, 224]
        """
        # FC and reshape to spatial
        x = self.dec_fc(z)
        x = x.view(-1, self.base_channels * 8, 7, 7)

        # Decoder forward pass
        x = self.dec_up1(x)
        x = self.dec_res1(x)

        x = self.dec_up2(x)
        x = self.dec_res2(x)

        x = self.dec_up3(x)
        x = self.dec_res3(x)

        x = self.dec_up4(x)
        x = self.dec_res4(x)

        x = self.dec_final(x)

        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass.

        Args:
            x: Input image tensor [batch, 3, 224, 224]

        Returns:
            Tuple of (reconstructed_image, embedding)
            - reconstructed_image: [batch, 3, 224, 224]
            - embedding: [batch, embedding_dim]
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning."""
        for name, param in self.named_parameters():
            if name.startswith('enc_'):
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for name, param in self.named_parameters():
            if name.startswith('enc_'):
                param.requires_grad = True

    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for name, param in self.named_parameters():
            if name.startswith('dec_'):
                param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for name, param in self.named_parameters():
            if name.startswith('dec_'):
                param.requires_grad = True


class MultiFingerTemporalRankingModel(nn.Module):
    """
    Multi-finger temporal ranking model with 5 parallel comparison heads.

    Shares a single backbone (frame encoder + transformer) across all fingers,
    but uses separate comparison heads for each finger. This allows the model
    to simultaneously learn motion ranking for all 5 fingers from the same
    ultrasound sequences.

    Architecture:
        - Input: Two sequences of seq_len frames each
        - Frame encoder: Shared CNN encodes each frame to embedding_dim
        - Positional encoding: Learnable positional embeddings
        - Transformer encoder: Self-attention over temporal dimension
        - Temporal pooling: Mean pooling over sequence
        - 5 comparison heads (nn.ModuleDict): Each predicts P(A > B) for one finger
    """

    FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        sequence_length: int = 16,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        in_channels: int = 3,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.sequence_length = sequence_length
        self.in_channels = in_channels

        # Channel progression: 64 -> 128 -> 256 -> 512 -> 512
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        # =========================
        # Frame Encoder
        # =========================
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.enc_res1 = ResBlock(c1)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.enc_res2 = ResBlock(c2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_res3 = ResBlock(c3)

        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.enc_res4 = ResBlock(c4)

        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.enc_pool = nn.AdaptiveAvgPool2d(1)
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # =========================
        # Temporal Transformer
        # =========================
        self.pos_embedding = nn.Parameter(
            torch.randn(1, sequence_length, embedding_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # =========================
        # 5 Comparison heads (one per finger)
        # =========================
        self.comparison_heads = nn.ModuleDict({
            finger: nn.Sequential(
                nn.Linear(2 * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1),
            )
            for finger in self.FINGER_NAMES
        })

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single frame to embedding. [batch, 3, H, W] -> [batch, embedding_dim]"""
        x = self.enc_conv1(x)
        x = self.enc_res1(x)
        x = self.enc_conv2(x)
        x = self.enc_res2(x)
        x = self.enc_conv3(x)
        x = self.enc_res3(x)
        x = self.enc_conv4(x)
        x = self.enc_res4(x)
        x = self.enc_conv5(x)
        x = self.enc_pool(x)
        x = x.flatten(1)
        z = self.enc_fc(x)
        return z

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode a sequence through frame encoder + transformer. [batch, seq_len, 3, H, W] -> [batch, embedding_dim]"""
        batch_size, seq_len, C, H, W = sequence.shape

        frames_flat = sequence.view(batch_size * seq_len, C, H, W)
        frame_embeddings = self.encode_frame(frames_flat)
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)

        return self.encode_from_embeddings(frame_embeddings)

    def encode_from_embeddings(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Run pre-computed frame embeddings through positional encoding + transformer + pooling.

        Enables efficient batched processing: encode all frames once with encode_frame(),
        then build overlapping temporal sequences without re-encoding.

        Args:
            frame_embeddings: [batch, seq_len, embedding_dim]
        Returns:
            sequence_embedding: [batch, embedding_dim]
        """
        seq_len = frame_embeddings.size(1)
        x = frame_embeddings + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return x.mean(dim=1)

    def encode_from_embeddings_per_position(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Run pre-computed frame embeddings through positional encoding + transformer.

        Unlike encode_from_embeddings() which mean-pools to a single vector,
        this returns the per-position transformer outputs for use with
        per-position comparison (e.g., flex classification).

        Args:
            frame_embeddings: [batch, seq_len, embedding_dim]
        Returns:
            per_position_embeddings: [batch, seq_len, embedding_dim]
        """
        seq_len = frame_embeddings.size(1)
        x = frame_embeddings + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return x

    def forward(self, sequence_A: torch.Tensor, sequence_B: torch.Tensor) -> dict:
        """
        Compare two sequences across all 5 fingers.

        Args:
            sequence_A: [batch, seq_len, 3, H, W] first sequence
            sequence_B: [batch, seq_len, 3, H, W] second sequence

        Returns:
            dict mapping finger name -> [batch] logits for P(A > B)
        """
        emb_A = self.encode_sequence(sequence_A)
        emb_B = self.encode_sequence(sequence_B)

        combined = torch.cat([emb_A, emb_B], dim=1)  # [batch, 2 * emb_dim]

        logits = {}
        for finger in self.FINGER_NAMES:
            logit = self.comparison_heads[finger](combined)  # [batch, 1]
            logits[finger] = logit.squeeze(1)  # [batch]

        return logits

    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load encoder weights from a pretrained EmbeddingAutoencoder or similar checkpoint."""
        print(f"Loading pretrained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('enc_')}

        missing_keys, unexpected_keys = self.load_state_dict(encoder_keys, strict=False)

        print(f"  Loaded {len(encoder_keys)} encoder parameters")
        print(f"  Missing keys (transformer + heads - expected): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

    def freeze_encoder(self):
        """Freeze all frame encoder parameters."""
        encoder_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all frame encoder parameters."""
        encoder_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, transformer_lr, heads_lr):
        """
        Get parameter groups with discriminative learning rates.

        Args:
            encoder_lr_early: LR for early encoder layers (conv1-3)
            encoder_lr_late: LR for late encoder layers (conv4-5, fc)
            transformer_lr: LR for positional encoding and transformer
            heads_lr: LR for all 5 comparison heads

        Returns:
            List of parameter group dictionaries for optimizer
        """
        early_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
        ]
        early_params = []
        for module in early_modules:
            early_params.extend(list(module.parameters()))

        late_modules = [
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        late_params = []
        for module in late_modules:
            late_params.extend(list(module.parameters()))

        transformer_params = [self.pos_embedding] + list(self.transformer_encoder.parameters())

        # All 5 heads share the same LR group
        heads_params = []
        for finger in self.FINGER_NAMES:
            heads_params.extend(list(self.comparison_heads[finger].parameters()))

        return [
            {'params': early_params, 'lr': encoder_lr_early},
            {'params': late_params, 'lr': encoder_lr_late},
            {'params': transformer_params, 'lr': transformer_lr},
            {'params': heads_params, 'lr': heads_lr},
        ]


class FlexEmbeddingClassifier(nn.Module):
    """
    Flex classifier using concatenated query + rest embeddings from a pretrained ranking model.

    Loads encoder + transformer from MultiFingerTemporalRankingModel, adds an MLP head
    that maps concatenated [query_emb; rest_emb] to 5 binary flex predictions.

    Architecture (default, no_rest=False):
        query_sequence  [B, T, 3, H, W] → encode_sequence → [B, embedding_dim] ─┐
                                                                                   ├ concat → [B, 2*emb_dim] → MLP → [B, 5]
        rest_sequence   [B, T, 3, H, W] → encode_sequence → [B, embedding_dim] ─┘

    Architecture (no_rest=True):
        query_sequence  [B, T, 3, H, W] → encode_sequence → [B, embedding_dim] → MLP → [B, 5]
    """

    FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        sequence_length: int = 16,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        mlp_hidden_dim: int = 128,
        mlp_dropout: float = 0.3,
        num_fingers: int = 5,
        no_rest: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_fingers = num_fingers
        self.no_rest = no_rest

        # Backbone: encoder + transformer (same architecture as ranking model)
        self.backbone = MultiFingerTemporalRankingModel(
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sequence_length=sequence_length,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
        )

        # MLP classification head
        mlp_input_dim = embedding_dim if no_rest else 2 * embedding_dim
        self.flex_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, num_fingers),
        )

    def load_ranking_weights(self, checkpoint_path: str, config_path: str,
                             device='cpu', encoder_only: bool = False):
        """Load weights from a pretrained ranking model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pth file.
            config_path: Path to config .json file.
            device: Device to load checkpoint onto.
            encoder_only: If True, load only CNN encoder weights (enc_*),
                leaving transformer randomly initialized.
        """
        import json
        print(f"Loading ranking backbone from: {checkpoint_path}")
        print(f"  Mode: {'encoder only' if encoder_only else 'full backbone'}")

        with open(config_path) as f:
            config = json.load(f)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if encoder_only:
            full_state = checkpoint['model_state_dict']
            encoder_state = {k: v for k, v in full_state.items() if k.startswith('enc_')}
            self.backbone.load_state_dict(encoder_state, strict=False)
            print(f"  Loaded {len(encoder_state)} encoder params, "
                  f"skipped {len(full_state) - len(encoder_state)} "
                  f"(transformer + comparison heads)")
        else:
            self.backbone.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', '?')
        metrics = checkpoint.get('metrics', {})
        mean_auc = metrics.get('mean_auc', 'N/A')
        print(f"  Loaded from epoch {epoch}, mean AUC: {mean_auc}")
        print(f"  Config: embedding_dim={config['embedding_dim']}, "
              f"hidden_dim={config['hidden_dim']}, "
              f"seq_len={config['sequence_length']}")

        return config

    def forward(self, query_sequence: torch.Tensor, rest_sequence: torch.Tensor = None) -> torch.Tensor:
        """
        Classify flex/no-flex per finger from query (and optionally rest) sequence embeddings.

        Args:
            query_sequence: [batch, seq_len, 3, H, W]
            rest_sequence: [batch, seq_len, 3, H, W] (ignored when no_rest=True)
        Returns:
            logits: [batch, 5] per-finger flex logits
        """
        query_emb = self.backbone.encode_sequence(query_sequence)  # [B, embedding_dim]
        if self.no_rest:
            return self.flex_head(query_emb)                        # [B, 5]
        rest_emb = self.backbone.encode_sequence(rest_sequence)    # [B, embedding_dim]
        combined = torch.cat([query_emb, rest_emb], dim=1)         # [B, 2 * embedding_dim]
        return self.flex_head(combined)                             # [B, 5]

    def freeze_encoder(self):
        """Freeze CNN encoder layers (keep transformer + flex_head trainable)."""
        for name, param in self.backbone.named_parameters():
            if name.startswith('enc_'):
                param.requires_grad = False

    def freeze_backbone(self):
        """Freeze entire backbone (encoder + transformer), keep only flex_head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late,
                                     transformer_lr, flex_head_lr):
        """
        Discriminative LR param groups for fine-tuning.

        Returns 4 groups: encoder early, encoder late, transformer, flex_head.
        Comparison heads in backbone are excluded (no gradients).
        """
        # Freeze comparison heads so they don't consume optimizer memory
        for finger in self.backbone.FINGER_NAMES:
            for param in self.backbone.comparison_heads[finger].parameters():
                param.requires_grad = False

        # Get first 3 groups from backbone (encoder early, encoder late, transformer)
        backbone_groups = self.backbone.get_discriminative_lr_params(
            encoder_lr_early, encoder_lr_late, transformer_lr, heads_lr=0.0
        )
        param_groups = backbone_groups[:3]

        # Add flex_head params
        param_groups.append({
            'params': list(self.flex_head.parameters()),
            'lr': flex_head_lr,
        })

        return param_groups


class CBMFEncoder(nn.Module):
    """
    UNet-style encoder for CBMF baseline.

    5 encoding blocks (DoubleConv + BN + optional Dropout + MaxPool) followed
    by IR (intermediate representation) encoding that collapses spatial dims
    to a fixed-size vector.

    Architecture (for 224x224 input):
        Block 0: 3→32,  MaxPool → 112×112
        Block 1: 32→64, MaxPool → 56×56
        Block 2: 64→128, MaxPool → 28×28
        Block 3: 128→256, Dropout(0.3), MaxPool → 14×14
        Block 4: 256→512, Dropout(0.3), no pool → 14×14
        IR: Conv(14×14) → Flatten → FC(512) → FC(128)
    """

    def __init__(self, in_channels: int = 3, n_filters: int = 32,
                 ir_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.ir_dim = ir_dim

        # Block 0
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_filters),
        )
        self.pool0 = nn.MaxPool2d(2)

        # Block 1
        f1 = n_filters * 2
        self.block1 = nn.Sequential(
            nn.Conv2d(n_filters, f1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(f1),
        )
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        f2 = n_filters * 4
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(f2),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Block 3 (with dropout)
        f3 = n_filters * 8
        self.block3 = nn.Sequential(
            nn.Conv2d(f2, f3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3, f3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(f3),
            nn.Dropout2d(dropout),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Block 4 (with dropout, no pooling)
        f4 = n_filters * 16  # 512
        self.block4 = nn.Sequential(
            nn.Conv2d(f3, f4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f4, f4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(f4),
            nn.Dropout2d(dropout),
        )

        # IR encoding: spatial collapse → dense → ir_dim
        # For 224×224 input: after 4 MaxPool(2) → 14×14 feature map
        fmap_size = 14  # 224 // (2**4)
        self.ir_conv = nn.Sequential(
            nn.Conv2d(f4, f4, kernel_size=fmap_size),
            nn.ReLU(inplace=True),
        )
        self.ir_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f4, f4),
            nn.ReLU(inplace=True),
            nn.Linear(f4, ir_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single frame. [B, 3, H, W] → [B, ir_dim]"""
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        x = self.ir_conv(x)
        x = self.ir_fc(x)
        return x


class CBMFModel(nn.Module):
    """
    Configuration-Based Multi-Frame (CBMF) model for flex classification.

    Two-stage architecture:
      1. Encoder stage: UNet encoder extracts per-frame 128D features,
         encoder GRU predicts per-frame joint angles.
      2. Decoder stage: decoder GRU receives concatenated [IR features, predicted configs]
         and produces per-frame flex logits.

    Phase 1 training: only encoder + encoder GRU (joint angle regression).
    Phase 2 training: full model (joint angle regression + flex classification).
    """

    # Joint columns used as intermediate representation targets
    JOINT_COLUMNS = [
        'joint11', 'joint12',   # index
        'joint21', 'joint22',   # middle
        'joint31', 'joint32',   # ring
        'joint41', 'joint42',   # pinky
        'jointt5', 'jointt6',   # thumb
    ]
    NUM_JOINTS = len(JOINT_COLUMNS)
    NUM_FINGERS = 5

    def __init__(
        self,
        in_channels: int = 3,
        n_filters: int = 32,
        ir_dim: int = 128,
        encoder_dropout: float = 0.3,
        num_joints: int = 10,
        num_fingers: int = 5,
        enc_gru1_hidden: int = 1024,
        enc_gru2_hidden: int = 256,
        dec_gru1_hidden: int = 256,
        dec_gru2_hidden: int = 128,
    ):
        super().__init__()
        self.ir_dim = ir_dim
        self.num_joints = num_joints
        self.num_fingers = num_fingers

        # --- Frame encoder ---
        self.encoder = CBMFEncoder(
            in_channels=in_channels,
            n_filters=n_filters,
            ir_dim=ir_dim,
            dropout=encoder_dropout,
        )

        # --- Encoder GRU (predicts joint angles) ---
        self.enc_gru1 = nn.GRU(ir_dim, enc_gru1_hidden, batch_first=True)
        self.enc_gru2 = nn.GRU(enc_gru1_hidden, enc_gru2_hidden, batch_first=True)
        self.enc_config_proj = nn.Linear(enc_gru2_hidden, num_joints)

        # --- Decoder GRU (classifies flex from configs + residual IR) ---
        dec_input_dim = ir_dim + num_joints
        self.dec_gru1 = nn.GRU(dec_input_dim, dec_gru1_hidden, batch_first=True)
        self.dec_gru2 = nn.GRU(dec_gru1_hidden, dec_gru2_hidden, batch_first=True)
        self.dec_flex_proj = nn.Linear(dec_gru2_hidden, num_fingers)

    def encode_frames(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode each frame in a sequence independently.

        Args:
            sequence: [B, T, C, H, W]
        Returns:
            ir_features: [B, T, ir_dim]
        """
        B, T, C, H, W = sequence.shape
        frames_flat = sequence.reshape(B * T, C, H, W)
        ir_flat = self.encoder(frames_flat)       # [B*T, ir_dim]
        return ir_flat.reshape(B, T, self.ir_dim)

    def predict_configs(self, ir_features: torch.Tensor) -> torch.Tensor:
        """Run encoder GRU to predict per-frame joint angles.

        Args:
            ir_features: [B, T, ir_dim]
        Returns:
            config_pred: [B, T, num_joints]
        """
        x, _ = self.enc_gru1(ir_features)   # [B, T, 1024]
        x, _ = self.enc_gru2(x)             # [B, T, 256]
        config_pred = self.enc_config_proj(x)  # [B, T, num_joints]
        return config_pred

    def predict_flex(self, ir_features: torch.Tensor,
                     config_pred: torch.Tensor) -> torch.Tensor:
        """Run decoder GRU to predict per-frame flex logits.

        Args:
            ir_features: [B, T, ir_dim] residual encoder features
            config_pred: [B, T, num_joints] predicted joint angles
        Returns:
            flex_logits: [B, T, num_fingers] per-frame flex logits
        """
        dec_input = torch.cat([ir_features, config_pred], dim=2)  # [B, T, ir_dim + num_joints]
        x, _ = self.dec_gru1(dec_input)  # [B, T, 256]
        x, _ = self.dec_gru2(x)          # [B, T, 128]
        flex_logits = self.dec_flex_proj(x)  # [B, T, num_fingers]
        return flex_logits

    def forward_phase1(self, query_sequence: torch.Tensor):
        """Phase 1 forward: encoder + GRU → joint angle predictions only.

        Args:
            query_sequence: [B, T, C, H, W]
        Returns:
            config_pred: [B, T, num_joints]
        """
        ir_features = self.encode_frames(query_sequence)
        config_pred = self.predict_configs(ir_features)
        return config_pred

    def forward(self, query_sequence: torch.Tensor):
        """Full forward: encoder → config prediction → flex classification.

        Args:
            query_sequence: [B, T, C, H, W]
        Returns:
            config_pred: [B, T, num_joints] predicted joint angles
            flex_logits: [B, 5] window-level flex logits (mean over frames)
            flex_logits_per_frame: [B, T, 5] per-frame flex logits
        """
        ir_features = self.encode_frames(query_sequence)
        config_pred = self.predict_configs(ir_features)
        flex_logits_per_frame = self.predict_flex(ir_features, config_pred)
        flex_logits = flex_logits_per_frame.mean(dim=1)  # [B, 5]
        return config_pred, flex_logits, flex_logits_per_frame

    def freeze_encoder(self):
        """Freeze UNet encoder (keep GRUs trainable)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class BimbrawVGG(nn.Module):
    """Modified VGG16 for single-frame flex classification (Bimbraw et al. 2023).

    Architecture: 5 conv blocks (64->128->256->512->512), each with
    Conv2d(3x3, pad=1) -> ReLU -> BN -> MaxPool(2x2).
    Classifier: Flatten(25088) -> FC(16) -> ReLU -> BN -> Dropout -> FC(5).
    """

    def __init__(self, num_fingers: int = 5, dropout: float = 0.5):
        super().__init__()
        self.block1 = self._make_block(3, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)
        self.block5 = self._make_block(512, 512)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.Linear(16, num_fingers),
        )

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Forward pass. [B, 3, 224, 224] -> [B, 5] logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.classifier(x)
