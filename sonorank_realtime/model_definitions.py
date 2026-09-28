"""Self-contained model definitions needed by real-time SonoRank inference.

Reader's guide: ``FlexEmbeddingClassifier`` is the deployed model. It depends
on ``MultiFingerTemporalRankingModel`` and the shared encoder blocks defined
above it. Historical architectures remain so legacy checkpoints can still be
inspected, but the normal real-time entry point does not instantiate them.

This file intentionally mirrors training-time class names and tensor layouts.
Changing them can make existing checkpoints impossible to load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def create_model(
    architecture='unet',
    encoder_name='resnet34',
    encoder_weights='imagenet',
    num_classes=8,
    in_channels=3,
):
    """
    Create a segmentation model.

    Args:
        architecture: Model architecture ('unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus', 'fpn')
        encoder_name: Encoder backbone ('resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', etc.)
        encoder_weights: Pretrained weights ('imagenet', None)
        num_classes: Number of segmentation classes
        in_channels: Number of input channels

    Returns:
        PyTorch model
    """
    architecture = architecture.lower()

    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


class SiameseJointRegressor(nn.Module):
    """
    Siamese network for joint angle delta prediction.

    Uses a shared encoder to process two images (t and t+k), concatenates
    their features, and predicts joint angle deltas through a regression head.

    The encoder can be initialized from a pretrained segmentation model.
    """

    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_joints=5,
        hidden_dims=[512, 256],
        dropout=0.3,
    ):
        """
        Initialize Siamese joint regressor.

        Args:
            encoder_name: Encoder backbone (e.g., 'resnet34', 'resnet50')
            encoder_weights: Pretrained weights ('imagenet' or None)
            num_joints: Number of joints to predict deltas for
            hidden_dims: Hidden layer dimensions for regression head
            dropout: Dropout rate in regression head
        """
        super().__init__()

        # Create encoder using segmentation_models_pytorch
        # We use Unet just to get the encoder, but we won't use the decoder
        temp_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,  # Dummy value, we won't use the decoder
        )

        # Extract only the encoder
        self.encoder = temp_model.encoder

        # Get encoder output channels
        # For ResNet34: encoder output is [batch, 512, H/32, W/32]
        # We'll use global average pooling to get [batch, 512]
        if 'resnet' in encoder_name.lower():
            if encoder_name == 'resnet18' or encoder_name == 'resnet34':
                self.encoder_out_channels = 512
            else:  # resnet50, resnet101, resnet152
                self.encoder_out_channels = 2048
        elif 'efficientnet' in encoder_name.lower():
            # EfficientNet varies, but we can infer from the encoder
            self.encoder_out_channels = temp_model.encoder.out_channels[-1]
        else:
            # Default fallback
            self.encoder_out_channels = 512

        # Global average pooling to convert spatial features to vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head
        # Input: concatenated features from both images [2 * encoder_out_channels]
        # Output: joint deltas [num_joints]
        layers = []
        in_features = 2 * self.encoder_out_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_features, num_joints))

        self.regression_head = nn.Sequential(*layers)

        # Store configuration
        self.encoder_name = encoder_name
        self.num_joints = num_joints

    def forward(self, image_t, image_t_plus_k):
        """
        Forward pass through Siamese network.

        Args:
            image_t: First image [batch, 3, H, W]
            image_t_plus_k: Second image [batch, 3, H, W]

        Returns:
            Joint angle deltas [batch, num_joints]
        """
        # Encode both images through shared encoder
        features_t = self.encoder(image_t)[-1]  # Get final feature map
        features_t_plus_k = self.encoder(image_t_plus_k)[-1]

        # Global average pooling: [batch, channels, H, W] -> [batch, channels, 1, 1]
        features_t = self.pool(features_t)
        features_t_plus_k = self.pool(features_t_plus_k)

        # Flatten: [batch, channels, 1, 1] -> [batch, channels]
        features_t = features_t.flatten(1)
        features_t_plus_k = features_t_plus_k.flatten(1)

        # Concatenate features
        combined_features = torch.cat([features_t, features_t_plus_k], dim=1)

        # Predict joint deltas
        joint_deltas = self.regression_head(combined_features)

        return joint_deltas

    def load_encoder_from_segmentation(self, checkpoint_path, strict=True):
        """
        Load encoder weights from a pretrained segmentation model checkpoint.

        Args:
            checkpoint_path: Path to segmentation model checkpoint
            strict: Whether to strictly enforce that keys match

        Returns:
            Dictionary with loading information
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter to only encoder keys
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key] = value

        # Load into our encoder
        result = self.encoder.load_state_dict(encoder_state_dict, strict=strict)

        # load_state_dict returns _IncompatibleKeys when strict=False
        # or None when strict=True and successful
        if result is None:
            missing_keys = []
            unexpected_keys = []
        else:
            missing_keys = result.missing_keys if hasattr(result, 'missing_keys') else []
            unexpected_keys = result.unexpected_keys if hasattr(result, 'unexpected_keys') else []

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'loaded_keys': len(encoder_state_dict),
        }

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning Phase 1."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for transfer learning Phase 2."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, head_lr):
        """
        Get parameter groups for discriminative learning rates.

        Args:
            encoder_lr_early: Learning rate for early encoder layers
            encoder_lr_late: Learning rate for late encoder layers
            head_lr: Learning rate for regression head

        Returns:
            List of parameter groups for optimizer
        """
        encoder_params = list(self.encoder.parameters())
        split_idx = len(encoder_params) // 2

        return [
            {'params': encoder_params[:split_idx], 'lr': encoder_lr_early},
            {'params': encoder_params[split_idx:], 'lr': encoder_lr_late},
            {'params': self.regression_head.parameters(), 'lr': head_lr},
        ]


class SiameseJointClassifier(nn.Module):
    """
    Siamese network for joint motion classification.

    Uses a shared encoder to process two images (t and t+k), computes the
    feature difference (t+k - t) to focus on motion rather than morphology,
    and predicts binary motion labels (1=moving, 0=static) for each joint.

    The encoder can be initialized from a pretrained segmentation model.
    """

    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_joints=5,
        hidden_dims=[512, 256],
        dropout=0.3,
    ):
        """
        Initialize Siamese joint classifier.

        Args:
            encoder_name: Encoder backbone (e.g., 'resnet34', 'resnet50')
            encoder_weights: Pretrained weights ('imagenet' or None)
            num_joints: Number of joints to classify
            hidden_dims: Hidden layer dimensions for classification head
            dropout: Dropout rate in classification head
        """
        super().__init__()

        # Create encoder using segmentation_models_pytorch
        # We use Unet just to get the encoder, but we won't use the decoder
        temp_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,  # Dummy value, we won't use the decoder
        )

        # Extract only the encoder
        self.encoder = temp_model.encoder

        # Get encoder output channels
        # For ResNet34: encoder output is [batch, 512, H/32, W/32]
        # We'll use global average pooling to get [batch, 512]
        if 'resnet' in encoder_name.lower():
            if encoder_name == 'resnet18' or encoder_name == 'resnet34':
                self.encoder_out_channels = 512
            else:  # resnet50, resnet101, resnet152
                self.encoder_out_channels = 2048
        elif 'efficientnet' in encoder_name.lower():
            # EfficientNet varies, but we can infer from the encoder
            self.encoder_out_channels = temp_model.encoder.out_channels[-1]
        else:
            # Default fallback
            self.encoder_out_channels = 512

        # Global average pooling to convert spatial features to vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        # Input: feature difference (t+k - t) [encoder_out_channels]
        # Output: binary logits for each joint [num_joints]
        layers = []
        in_features = self.encoder_out_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim

        # Final output layer (no sigmoid - we'll use BCEWithLogitsLoss)
        layers.append(nn.Linear(in_features, num_joints))

        self.classification_head = nn.Sequential(*layers)

        # Store configuration
        self.encoder_name = encoder_name
        self.num_joints = num_joints

    def forward(self, image_t, image_t_plus_k):
        """
        Forward pass through Siamese network.

        Args:
            image_t: First image [batch, 3, H, W]
            image_t_plus_k: Second image [batch, 3, H, W]

        Returns:
            Binary logits for each joint [batch, num_joints]
            (apply sigmoid to get probabilities)
        """
        # Encode both images through shared encoder
        features_t = self.encoder(image_t)[-1]  # Get final feature map
        features_t_plus_k = self.encoder(image_t_plus_k)[-1]

        # Global average pooling: [batch, channels, H, W] -> [batch, channels, 1, 1]
        features_t = self.pool(features_t)
        features_t_plus_k = self.pool(features_t_plus_k)

        # Flatten: [batch, channels, 1, 1] -> [batch, channels]
        features_t = features_t.flatten(1)
        features_t_plus_k = features_t_plus_k.flatten(1)

        # Compute feature difference (motion-focused, removes static morphology)
        motion_features = features_t_plus_k - features_t

        # Predict binary motion labels (logits)
        motion_logits = self.classification_head(motion_features)

        return motion_logits

    def load_encoder_from_segmentation(self, checkpoint_path, strict=True):
        """
        Load encoder weights from a pretrained segmentation model checkpoint.

        Args:
            checkpoint_path: Path to segmentation model checkpoint
            strict: Whether to strictly enforce that keys match

        Returns:
            Dictionary with loading information
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter to only encoder keys
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key] = value

        # Load into our encoder
        result = self.encoder.load_state_dict(encoder_state_dict, strict=strict)

        # load_state_dict returns _IncompatibleKeys when strict=False
        # or None when strict=True and successful
        if result is None:
            missing_keys = []
            unexpected_keys = []
        else:
            missing_keys = result.missing_keys if hasattr(result, 'missing_keys') else []
            unexpected_keys = result.unexpected_keys if hasattr(result, 'unexpected_keys') else []

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'loaded_keys': len(encoder_state_dict),
        }

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning Phase 1."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for transfer learning Phase 2."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, head_lr):
        """
        Get parameter groups for discriminative learning rates.

        Args:
            encoder_lr_early: Learning rate for early encoder layers
            encoder_lr_late: Learning rate for late encoder layers
            head_lr: Learning rate for classification head

        Returns:
            List of parameter groups for optimizer
        """
        encoder_params = list(self.encoder.parameters())
        split_idx = len(encoder_params) // 2

        return [
            {'params': encoder_params[:split_idx], 'lr': encoder_lr_early},
            {'params': encoder_params[split_idx:], 'lr': encoder_lr_late},
            {'params': self.classification_head.parameters(), 'lr': head_lr},
        ]


def count_parameters(model):
    """Count trainable and total parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test segmentation model creation
    print("=" * 80)
    print("Testing Segmentation Model")
    print("=" * 80)

    model = create_model(
        architecture='unet',
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_classes=8,
    )

    print(f"Model: U-Net with ResNet34 encoder")
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: ({batch_size}, 8, 224, 224)")

    assert y.shape == (batch_size, 8, 224, 224), "Output shape mismatch!"
    print("✓ Segmentation model test passed!")
    print()

    # Test Siamese joint regressor
    print("=" * 80)
    print("Testing Siamese Joint Regressor")
    print("=" * 80)

    siamese_model = SiameseJointRegressor(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_joints=5,
        hidden_dims=[512, 256],
        dropout=0.3,
    )

    print(f"Model: Siamese Joint Regressor with ResNet34 encoder")
    print(f"Total parameters: {count_parameters(siamese_model):,}")

    # Count encoder vs regression head parameters
    encoder_params = count_parameters(siamese_model.encoder)
    head_params = count_parameters(siamese_model.regression_head)
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Regression head parameters: {head_params:,}")

    # Test forward pass
    image_t = torch.randn(batch_size, 3, 224, 224)
    image_t_plus_k = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        joint_deltas = siamese_model(image_t, image_t_plus_k)

    print(f"Input shapes:")
    print(f"  image_t: {image_t.shape}")
    print(f"  image_t_plus_k: {image_t_plus_k.shape}")
    print(f"Output shape: {joint_deltas.shape}")
    print(f"Expected output shape: ({batch_size}, 5)")

    assert joint_deltas.shape == (batch_size, 5), "Output shape mismatch!"
    print("✓ Siamese model test passed!")
    print()

    # Test transfer learning utilities
    print("=" * 80)
    print("Testing Transfer Learning Utilities")
    print("=" * 80)

    print("Testing freeze/unfreeze:")
    siamese_model.freeze_encoder()
    frozen_params = sum(1 for p in siamese_model.encoder.parameters() if not p.requires_grad)
    total_encoder_params = sum(1 for p in siamese_model.encoder.parameters())
    print(f"  Frozen encoder parameters: {frozen_params}/{total_encoder_params}")
    assert frozen_params == total_encoder_params, "Not all encoder params frozen!"

    siamese_model.unfreeze_encoder()
    unfrozen_params = sum(1 for p in siamese_model.encoder.parameters() if p.requires_grad)
    print(f"  Unfrozen encoder parameters: {unfrozen_params}/{total_encoder_params}")
    assert unfrozen_params == total_encoder_params, "Not all encoder params unfrozen!"
    print("✓ Freeze/unfreeze test passed!")
    print()

    print("Testing discriminative learning rate parameter groups:")
    param_groups = siamese_model.get_discriminative_lr_params(
        encoder_lr_early=1e-5,
        encoder_lr_late=5e-5,
        head_lr=3e-4,
    )
    print(f"  Number of parameter groups: {len(param_groups)}")
    print(f"  Early encoder layers LR: {param_groups[0]['lr']}")
    print(f"  Late encoder layers LR: {param_groups[1]['lr']}")
    print(f"  Regression head LR: {param_groups[2]['lr']}")
    assert len(param_groups) == 3, "Expected 3 parameter groups!"
    print("✓ Discriminative LR test passed!")
    print()

    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)


class PairwiseRankingModel(nn.Module):
    """
    Pairwise ranking model for comparing motion magnitude between two temporal pairs.

    Uses a shared encoder (same architecture as EmbeddingAutoencoder) to process 4 images
    (two temporal pairs), computes feature differences for each pair, concatenates them,
    and predicts which pair shows greater motion through a comparison head.

    Architecture:
        - Shared encoder processes: image_t1, image_t1+k1, image_t2, image_t2+k2
        - Encoder outputs 64-dim embeddings (same as EmbeddingAutoencoder)
        - Compute diff_A = emb(t1+k1) - emb(t1)
        - Compute diff_B = emb(t2+k2) - emb(t2)
        - Concatenate [diff_A, diff_B] -> 128-dim
        - Comparison head outputs: logit for P(A > B)

    The encoder can be initialized from a pretrained EmbeddingAutoencoder checkpoint.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        """
        Initialize Pairwise ranking model with EmbeddingAutoencoder-style encoder.

        Args:
            embedding_dim: Dimension of the embedding vector (default: 64, same as EmbeddingAutoencoder)
            base_channels: Number of channels in first conv layer (default: 64)
            hidden_dim: Hidden dimension for comparison head MLP
            dropout: Dropout rate in comparison head
            in_channels: Number of input image channels (default: 3 for RGB)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.in_channels = in_channels

        # Channel progression: 64 -> 128 -> 256 -> 512 -> 512
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        # =========================
        # Encoder (same as EmbeddingAutoencoder)
        # =========================
        # 224x224x3 -> 112x112x64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.enc_res1 = ResBlock(c1)

        # 112x112x64 -> 56x56x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.enc_res2 = ResBlock(c2)

        # 56x56x128 -> 28x28x256
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_res3 = ResBlock(c3)

        # 28x28x256 -> 14x14x512
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.enc_res4 = ResBlock(c4)

        # 14x14x512 -> 7x7x512
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 7x7x512 -> 1x1x512
        self.enc_pool = nn.AdaptiveAvgPool2d(1)

        # FC layer: 512 -> embedding_dim
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # =========================
        # Comparison head
        # =========================
        # Input: 2 * embedding_dim (concatenated differences) = 128-dim
        # Output: 1 (binary logit for P(A > B))
        self.comparison_head = nn.Sequential(
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image to embedding.

        Args:
            x: [batch, 3, H, W] input image

        Returns:
            [batch, embedding_dim] embedding vector
        """
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

    def forward(self, img_t1, img_t1_k1, img_t2, img_t2_k2):
        """
        Forward pass for pairwise ranking.

        Args:
            img_t1: [batch, 3, H, W] - First pair, time t1
            img_t1_k1: [batch, 3, H, W] - First pair, time t1+k1
            img_t2: [batch, 3, H, W] - Second pair, time t2
            img_t2_k2: [batch, 3, H, W] - Second pair, time t2+k2

        Returns:
            logit: [batch] - Logit for P(A > B), where A is first pair, B is second pair
        """
        # Encode all 4 images through shared encoder
        emb_t1 = self.encode(img_t1)          # [batch, embedding_dim]
        emb_t1_k1 = self.encode(img_t1_k1)
        emb_t2 = self.encode(img_t2)
        emb_t2_k2 = self.encode(img_t2_k2)

        # Compute differences (motion features)
        diff_A = emb_t1_k1 - emb_t1  # [batch, embedding_dim]
        diff_B = emb_t2_k2 - emb_t2  # [batch, embedding_dim]

        # Concatenate and compare
        combined = torch.cat([diff_A, diff_B], dim=1)  # [batch, 2 * embedding_dim]
        logit = self.comparison_head(combined)  # [batch, 1]

        return logit.squeeze(1)  # [batch]

    def load_pretrained_encoder(self, checkpoint_path: str):
        """
        Load encoder weights from a pretrained EmbeddingAutoencoder checkpoint.

        Args:
            checkpoint_path: Path to EmbeddingAutoencoder checkpoint (.pth file)
        """
        print(f"Loading pretrained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter to only encoder keys (enc_*)
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('enc_')}

        # Load encoder weights
        missing_keys, unexpected_keys = self.load_state_dict(encoder_keys, strict=False)

        print(f"  Loaded {len(encoder_keys)} encoder parameters")
        print(f"  Missing keys (comparison head - expected): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

    def freeze_encoder(self):
        """Freeze all encoder parameters for transfer learning."""
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
        """Unfreeze all encoder parameters for fine-tuning."""
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

    def get_encoder_params(self):
        """Get all encoder parameters as a list."""
        encoder_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        params = []
        for module in encoder_modules:
            params.extend(list(module.parameters()))
        return params

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, head_lr):
        """
        Get parameter groups with discriminative learning rates.

        Divides encoder into early and late layers with different LRs,
        and assigns a separate LR to the comparison head.

        Args:
            encoder_lr_early: Learning rate for early encoder layers (conv1-3)
            encoder_lr_late: Learning rate for late encoder layers (conv4-5, fc)
            head_lr: Learning rate for comparison head

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Early encoder layers: conv1, res1, conv2, res2, conv3, res3
        early_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
        ]
        early_params = []
        for module in early_modules:
            early_params.extend(list(module.parameters()))

        # Late encoder layers: conv4, res4, conv5, pool, fc
        late_modules = [
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        late_params = []
        for module in late_modules:
            late_params.extend(list(module.parameters()))

        return [
            {'params': early_params, 'lr': encoder_lr_early},
            {'params': late_params, 'lr': encoder_lr_late},
            {'params': self.comparison_head.parameters(), 'lr': head_lr},
        ]


class TemporalRankingModel(nn.Module):
    """
    Temporal ranking model using Transformer encoder over frame sequences.

    Compares two sequences of consecutive frames to predict which has greater motion.
    Uses a shared frame encoder (same architecture as EmbeddingAutoencoder/PairwiseRankingModel)
    to encode each frame, then applies a Transformer encoder over the temporal dimension
    to capture motion dynamics, and finally uses a comparison head for ranking.

    Architecture:
        - Input: Two sequences of seq_len frames each
        - Frame encoder: Shared CNN encodes each frame to embedding_dim
        - Positional encoding: Learnable positional embeddings added to frame embeddings
        - Transformer encoder: Self-attention over temporal dimension
        - Temporal pooling: Mean pooling over sequence to get sequence embedding
        - Comparison head: MLP predicts logit for P(A > B)

    The encoder can be initialized from a pretrained EmbeddingAutoencoder or
    PairwiseRankingModel checkpoint.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        sequence_length: int = 10,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        in_channels: int = 3,
    ):
        """
        Initialize Temporal ranking model.

        Args:
            embedding_dim: Dimension of the frame embedding vector (default: 64)
            base_channels: Number of channels in first conv layer (default: 64)
            hidden_dim: Hidden dimension for transformer FFN and comparison head
            dropout: Dropout rate in transformer and comparison head
            sequence_length: Number of frames per sequence (default: 10)
            num_transformer_layers: Number of transformer encoder layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            in_channels: Number of input image channels (default: 3 for RGB)
        """
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
        # Frame Encoder (same as PairwiseRankingModel/EmbeddingAutoencoder)
        # =========================
        # 224x224x3 -> 112x112x64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.enc_res1 = ResBlock(c1)

        # 112x112x64 -> 56x56x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.enc_res2 = ResBlock(c2)

        # 56x56x128 -> 28x28x256
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_res3 = ResBlock(c3)

        # 28x28x256 -> 14x14x512
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.enc_res4 = ResBlock(c4)

        # 14x14x512 -> 7x7x512
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 7x7x512 -> 1x1x512
        self.enc_pool = nn.AdaptiveAvgPool2d(1)

        # FC layer: 512 -> embedding_dim
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # =========================
        # Temporal Transformer
        # =========================
        # Learnable positional encoding for temporal position
        self.pos_embedding = nn.Parameter(
            torch.randn(1, sequence_length, embedding_dim) * 0.02
        )

        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # =========================
        # Comparison head
        # =========================
        # Input: 2 * embedding_dim (concatenated sequence embeddings) = 128-dim
        # Output: 1 (binary logit for P(A > B))
        self.comparison_head = nn.Sequential(
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

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single frame (or batch of frames) to embedding.

        Args:
            x: [batch, 3, H, W] input image(s)

        Returns:
            [batch, embedding_dim] embedding vector(s)
        """
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
        """
        Encode a sequence of frames through frame encoder + transformer.

        Args:
            sequence: [batch, seq_len, 3, H, W] input sequence

        Returns:
            [batch, embedding_dim] pooled sequence embedding
        """
        batch_size, seq_len, C, H, W = sequence.shape

        # Encode each frame: reshape to [batch * seq_len, C, H, W]
        frames_flat = sequence.view(batch_size * seq_len, C, H, W)
        frame_embeddings = self.encode_frame(frames_flat)  # [batch * seq_len, emb_dim]

        # Reshape back: [batch, seq_len, emb_dim]
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)

        # Add positional encoding (handle variable sequence lengths)
        frame_embeddings = frame_embeddings + self.pos_embedding[:, :seq_len, :]

        # Transformer encoder
        temporal_features = self.transformer_encoder(frame_embeddings)  # [batch, seq_len, emb_dim]

        # Pool over time (mean pooling)
        sequence_embedding = temporal_features.mean(dim=1)  # [batch, emb_dim]

        return sequence_embedding

    def forward(self, sequence_A: torch.Tensor, sequence_B: torch.Tensor) -> torch.Tensor:
        """
        Compare two sequences and predict which has more motion.

        Args:
            sequence_A: [batch, seq_len, 3, H, W] first sequence
            sequence_B: [batch, seq_len, 3, H, W] second sequence

        Returns:
            [batch] logits for P(A > B)
        """
        # Encode both sequences
        emb_A = self.encode_sequence(sequence_A)  # [batch, emb_dim]
        emb_B = self.encode_sequence(sequence_B)  # [batch, emb_dim]

        # Concatenate and compare
        combined = torch.cat([emb_A, emb_B], dim=1)  # [batch, 2 * emb_dim]
        logit = self.comparison_head(combined)  # [batch, 1]

        return logit.squeeze(1)  # [batch]

    def load_pretrained_encoder(self, checkpoint_path: str):
        """
        Load encoder weights from a pretrained EmbeddingAutoencoder or PairwiseRankingModel checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
        """
        print(f"Loading pretrained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter to only encoder keys (enc_*)
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('enc_')}

        # Load encoder weights
        missing_keys, unexpected_keys = self.load_state_dict(encoder_keys, strict=False)

        print(f"  Loaded {len(encoder_keys)} encoder parameters")
        print(f"  Missing keys (transformer + comparison head - expected): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

    def freeze_encoder(self):
        """Freeze all frame encoder parameters for transfer learning."""
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
        """Unfreeze all frame encoder parameters for fine-tuning."""
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

    def get_encoder_params(self):
        """Get all frame encoder parameters as a list."""
        encoder_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        params = []
        for module in encoder_modules:
            params.extend(list(module.parameters()))
        return params

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, transformer_lr, head_lr):
        """
        Get parameter groups with discriminative learning rates.

        Divides parameters into: early encoder, late encoder, transformer, and comparison head.

        Args:
            encoder_lr_early: Learning rate for early encoder layers (conv1-3)
            encoder_lr_late: Learning rate for late encoder layers (conv4-5, fc)
            transformer_lr: Learning rate for positional encoding and transformer
            head_lr: Learning rate for comparison head

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Early encoder layers: conv1, res1, conv2, res2, conv3, res3
        early_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
        ]
        early_params = []
        for module in early_modules:
            early_params.extend(list(module.parameters()))

        # Late encoder layers: conv4, res4, conv5, pool, fc
        late_modules = [
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        late_params = []
        for module in late_modules:
            late_params.extend(list(module.parameters()))

        # Transformer params: positional embedding + transformer encoder
        transformer_params = [self.pos_embedding] + list(self.transformer_encoder.parameters())

        return [
            {'params': early_params, 'lr': encoder_lr_early},
            {'params': late_params, 'lr': encoder_lr_late},
            {'params': transformer_params, 'lr': transformer_lr},
            {'params': self.comparison_head.parameters(), 'lr': head_lr},
        ]


class TemporalDeltaDiffRegressor(nn.Module):
    """
    Temporal delta difference regression model using Transformer encoder.

    Predicts the signed difference in motion magnitude between two sequences:
        target = delta_A - delta_B (signed, can be positive or negative)

    Same architecture as TemporalRankingModel but with regression head instead of
    classification head. Output is a continuous value representing how much more
    sequence A moved compared to sequence B.

    Architecture:
        - Input: Two sequences of seq_len frames each
        - Frame encoder: Shared CNN encodes each frame to embedding_dim
        - Positional encoding: Learnable positional embeddings added to frame embeddings
        - Transformer encoder: Self-attention over temporal dimension
        - Temporal pooling: Mean pooling over sequence to get sequence embedding
        - Regression head: MLP predicts continuous delta_diff value

    The encoder can be initialized from a pretrained EmbeddingAutoencoder or
    TemporalRankingModel checkpoint.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        sequence_length: int = 10,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        in_channels: int = 3,
    ):
        """
        Initialize Temporal delta difference regression model.

        Args:
            embedding_dim: Dimension of the frame embedding vector (default: 64)
            base_channels: Number of channels in first conv layer (default: 64)
            hidden_dim: Hidden dimension for transformer FFN and regression head
            dropout: Dropout rate in transformer and regression head
            sequence_length: Number of frames per sequence (default: 10)
            num_transformer_layers: Number of transformer encoder layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            in_channels: Number of input image channels (default: 3 for RGB)
        """
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
        # Frame Encoder (same as TemporalRankingModel)
        # =========================
        # 224x224x3 -> 112x112x64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.enc_res1 = ResBlock(c1)

        # 112x112x64 -> 56x56x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.enc_res2 = ResBlock(c2)

        # 56x56x128 -> 28x28x256
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_res3 = ResBlock(c3)

        # 28x28x256 -> 14x14x512
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.enc_res4 = ResBlock(c4)

        # 14x14x512 -> 7x7x512
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 7x7x512 -> 1x1x512
        self.enc_pool = nn.AdaptiveAvgPool2d(1)

        # FC layer: 512 -> embedding_dim
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # =========================
        # Temporal Transformer
        # =========================
        # Learnable positional encoding for temporal position
        self.pos_embedding = nn.Parameter(
            torch.randn(1, sequence_length, embedding_dim) * 0.02
        )

        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # =========================
        # Regression head
        # =========================
        # Input: 2 * embedding_dim (concatenated sequence embeddings) = 128-dim
        # Output: 1 (continuous delta_diff value: delta_A - delta_B)
        self.regression_head = nn.Sequential(
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

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single frame (or batch of frames) to embedding.

        Args:
            x: [batch, 3, H, W] input image(s)

        Returns:
            [batch, embedding_dim] embedding vector(s)
        """
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
        """
        Encode a sequence of frames through frame encoder + transformer.

        Args:
            sequence: [batch, seq_len, 3, H, W] input sequence

        Returns:
            [batch, embedding_dim] pooled sequence embedding
        """
        batch_size, seq_len, C, H, W = sequence.shape

        # Encode each frame: reshape to [batch * seq_len, C, H, W]
        frames_flat = sequence.view(batch_size * seq_len, C, H, W)
        frame_embeddings = self.encode_frame(frames_flat)  # [batch * seq_len, emb_dim]

        # Reshape back: [batch, seq_len, emb_dim]
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)

        # Add positional encoding (handle variable sequence lengths)
        frame_embeddings = frame_embeddings + self.pos_embedding[:, :seq_len, :]

        # Transformer encoder
        temporal_features = self.transformer_encoder(frame_embeddings)  # [batch, seq_len, emb_dim]

        # Pool over time (mean pooling)
        sequence_embedding = temporal_features.mean(dim=1)  # [batch, emb_dim]

        return sequence_embedding

    def forward(self, sequence_A: torch.Tensor, sequence_B: torch.Tensor) -> torch.Tensor:
        """
        Predict signed difference in motion magnitude between two sequences.

        Args:
            sequence_A: [batch, seq_len, 3, H, W] first sequence
            sequence_B: [batch, seq_len, 3, H, W] second sequence

        Returns:
            [batch] predicted delta_diff values (delta_A - delta_B)
        """
        # Encode both sequences
        emb_A = self.encode_sequence(sequence_A)  # [batch, emb_dim]
        emb_B = self.encode_sequence(sequence_B)  # [batch, emb_dim]

        # Concatenate and predict delta_diff
        combined = torch.cat([emb_A, emb_B], dim=1)  # [batch, 2 * emb_dim]
        delta_diff = self.regression_head(combined)  # [batch, 1]

        return delta_diff.squeeze(1)  # [batch]

    def load_pretrained_encoder(self, checkpoint_path: str):
        """
        Load encoder weights from a pretrained EmbeddingAutoencoder, TemporalRankingModel,
        or PairwiseRankingModel checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
        """
        print(f"Loading pretrained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter to only encoder keys (enc_*)
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('enc_')}

        # Load encoder weights
        missing_keys, unexpected_keys = self.load_state_dict(encoder_keys, strict=False)

        print(f"  Loaded {len(encoder_keys)} encoder parameters")
        print(f"  Missing keys (transformer + regression head - expected): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

    def freeze_encoder(self):
        """Freeze all frame encoder parameters for transfer learning."""
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
        """Unfreeze all frame encoder parameters for fine-tuning."""
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

    def get_encoder_params(self):
        """Get all frame encoder parameters as a list."""
        encoder_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        params = []
        for module in encoder_modules:
            params.extend(list(module.parameters()))
        return params

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, transformer_lr, head_lr):
        """
        Get parameter groups with discriminative learning rates.

        Divides parameters into: early encoder, late encoder, transformer, and regression head.

        Args:
            encoder_lr_early: Learning rate for early encoder layers (conv1-3)
            encoder_lr_late: Learning rate for late encoder layers (conv4-5, fc)
            transformer_lr: Learning rate for positional encoding and transformer
            head_lr: Learning rate for regression head

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Early encoder layers: conv1, res1, conv2, res2, conv3, res3
        early_modules = [
            self.enc_conv1, self.enc_res1,
            self.enc_conv2, self.enc_res2,
            self.enc_conv3, self.enc_res3,
        ]
        early_params = []
        for module in early_modules:
            early_params.extend(list(module.parameters()))

        # Late encoder layers: conv4, res4, conv5, pool, fc
        late_modules = [
            self.enc_conv4, self.enc_res4,
            self.enc_conv5, self.enc_pool, self.enc_fc,
        ]
        late_params = []
        for module in late_modules:
            late_params.extend(list(module.parameters()))

        # Transformer params: positional embedding + transformer encoder
        transformer_params = [self.pos_embedding] + list(self.transformer_encoder.parameters())

        return [
            {'params': early_params, 'lr': encoder_lr_early},
            {'params': late_params, 'lr': encoder_lr_late},
            {'params': transformer_params, 'lr': transformer_lr},
            {'params': self.regression_head.parameters(), 'lr': head_lr},
        ]


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
        # Frame Encoder (same as TemporalRankingModel)
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


class VelocityEmbeddingRegressor(nn.Module):
    """
    Velocity regressor using concatenated query + rest embeddings from a pretrained ranking model.

    Predicts signed mean angular velocity (deg/s) per finger from ultrasound
    image sequences. Negative = flexion, positive = extension, near zero = rest.

    Architecture (identical to FlexEmbeddingClassifier, but regression output):
        query_sequence  [B, T, 3, H, W] → encode_sequence → [B, embedding_dim] ─┐
                                                                                   ├ concat → [B, 2*emb_dim] → MLP → [B, 5]
        rest_sequence   [B, T, 3, H, W] → encode_sequence → [B, embedding_dim] ─┘
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
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_fingers = num_fingers

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

        # MLP regression head: [2 * embedding_dim] → [num_fingers] velocities
        self.velocity_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, num_fingers),
        )

    def load_ranking_weights(self, checkpoint_path: str, config_path: str,
                             device='cpu', encoder_only: bool = False):
        """Load weights from a pretrained ranking model checkpoint."""
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

    def forward(self, query_sequence: torch.Tensor, rest_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict per-finger mean angular velocity from query and rest sequence embeddings.

        Args:
            query_sequence: [batch, seq_len, 3, H, W]
            rest_sequence: [batch, seq_len, 3, H, W]
        Returns:
            velocities: [batch, 5] predicted mean angular velocity (deg/s) per finger
        """
        query_emb = self.backbone.encode_sequence(query_sequence)  # [B, embedding_dim]
        rest_emb = self.backbone.encode_sequence(rest_sequence)    # [B, embedding_dim]
        combined = torch.cat([query_emb, rest_emb], dim=1)         # [B, 2 * embedding_dim]
        return self.velocity_head(combined)                         # [B, 5]

    def freeze_encoder(self):
        """Freeze CNN encoder layers (keep transformer + velocity_head trainable)."""
        for name, param in self.backbone.named_parameters():
            if name.startswith('enc_'):
                param.requires_grad = False

    def freeze_backbone(self):
        """Freeze entire backbone (encoder + transformer), keep only velocity_head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late,
                                     transformer_lr, velocity_head_lr):
        """
        Discriminative LR param groups for fine-tuning.

        Returns 4 groups: encoder early, encoder late, transformer, velocity_head.
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

        # Add velocity_head params
        param_groups.append({
            'params': list(self.velocity_head.parameters()),
            'lr': velocity_head_lr,
        })

        return param_groups


class PWCNet(nn.Module):
    """
    PWC-Net: Pyramid, Warping, and Cost Volume Network for Optical Flow.

    Simplified implementation optimized for real-time ultrasound applications.

    Architecture:
    1. Feature Pyramid: Extract multi-scale features from both images
    2. Warping: Warp features from image2 using upsampled flow from coarser level
    3. Cost Volume: Compute correlation between image1 features and warped image2 features
    4. Flow Estimator: Predict flow residual and refine
    5. Context Network: Post-process final flow

    Reference: "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"
               Sun et al., CVPR 2018
    """

    def __init__(
        self,
        num_levels=6,
        search_range=4,
        output_level=4,
        feature_channels=[16, 32, 64, 96, 128, 196],
    ):
        """
        Initialize PWC-Net.

        Args:
            num_levels: Number of pyramid levels (default 6)
            search_range: Search range for cost volume (default 4)
            output_level: Pyramid level to output flow from (default 4, starts from 0)
            feature_channels: Feature channels at each pyramid level
        """
        super().__init__()

        self.num_levels = num_levels
        self.search_range = search_range
        self.output_level = output_level
        self.feature_channels = feature_channels

        # Leaky ReLU activation (must be created first)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        # Feature pyramid extractor
        self.feature_pyramid = self._make_feature_pyramid()

        # Flow estimators for each level
        self.flow_estimators = nn.ModuleList()
        for level in range(num_levels):
            self.flow_estimators.append(self._make_flow_estimator(level))

        # Context network (applied to final flow)
        self.context_network = self._make_context_network()

    def _make_feature_pyramid(self):
        """Create feature pyramid extractor."""
        layers = nn.ModuleList()

        # Level 0: Input is RGB image (3 channels)
        # Each level doubles the channels and halves the resolution
        in_channels = 3

        for level in range(self.num_levels):
            out_channels = self.feature_channels[level]

            # Two conv layers per level
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                self.leaky_relu,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                self.leaky_relu,
            )

            layers.append(layer)
            in_channels = out_channels

        return layers

    def _make_flow_estimator(self, level):
        """
        Create flow estimator for a specific pyramid level.

        Input: concatenation of [cost_volume, features1, upsampled_flow]
        Output: flow residual (2 channels)
        """
        # Cost volume size: (2*search_range+1)^2
        cost_volume_channels = (2 * self.search_range + 1) ** 2

        # Features from image1
        feature_channels = self.feature_channels[level]

        # Upsampled flow from coarser level (2 channels) or 0 for coarsest level
        flow_channels = 2

        # Total input channels
        in_channels = cost_volume_channels + feature_channels + flow_channels

        # Flow estimator: stack of conv layers
        layers = [
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),  # Output: 2-channel flow
        ]

        return nn.Sequential(*layers)

    def _make_context_network(self):
        """
        Create context network for flow refinement.

        Dilated convolutions to enlarge receptive field without downsampling.
        """
        layers = [
            nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            self.leaky_relu,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            self.leaky_relu,
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
            self.leaky_relu,
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
            self.leaky_relu,
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            self.leaky_relu,
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
        ]

        return nn.Sequential(*layers)

    def compute_cost_volume(self, features1, features2, search_range):
        """
        Compute cost volume by correlation.

        Args:
            features1: Features from image1 [B, C, H, W]
            features2: Features from image2 (warped) [B, C, H, W]
            search_range: Search range (e.g., 4 means searching in [-4, 4] range)

        Returns:
            Cost volume [B, (2*search_range+1)^2, H, W]
        """
        batch, channels, height, width = features1.shape

        # Normalize features
        features1 = features1 / (torch.norm(features1, dim=1, keepdim=True) + 1e-8)
        features2 = features2 / (torch.norm(features2, dim=1, keepdim=True) + 1e-8)

        # Compute cost volume by shifting features2
        cost_volume = []

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                # Shift features2
                shifted_features2 = torch.zeros_like(features2)

                # Compute valid region after shift
                h_start = max(0, dy)
                h_end = min(height, height + dy)
                w_start = max(0, dx)
                w_end = min(width, width + dx)

                h_start_src = max(0, -dy)
                h_end_src = min(height, height - dy)
                w_start_src = max(0, -dx)
                w_end_src = min(width, width - dx)

                if h_end > h_start and w_end > w_start:
                    shifted_features2[:, :, h_start:h_end, w_start:w_end] = \
                        features2[:, :, h_start_src:h_end_src, w_start_src:w_end_src]

                # Compute correlation (element-wise product then sum over channels)
                correlation = (features1 * shifted_features2).sum(dim=1, keepdim=True)
                cost_volume.append(correlation)

        # Stack along channel dimension
        cost_volume = torch.cat(cost_volume, dim=1)

        return cost_volume

    def warp_features(self, features, flow):
        """
        Warp features using optical flow.

        Args:
            features: Features to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]

        Returns:
            Warped features [B, C, H, W]
        """
        batch, channels, height, width = features.shape

        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=torch.float32),
            torch.arange(width, device=features.device, dtype=torch.float32),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)
        grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)

        # Add flow to grid
        grid_x = grid_x + flow[:, 0, :, :]
        grid_y = grid_y + flow[:, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        grid_x = 2.0 * grid_x / (width - 1) - 1.0
        grid_y = 2.0 * grid_y / (height - 1) - 1.0

        # Stack to create grid [B, H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Warp using grid_sample
        warped_features = torch.nn.functional.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped_features

    def forward(self, image1, image2):
        """
        Forward pass through PWC-Net.

        Args:
            image1: First image [B, 3, H, W]
            image2: Second image [B, 3, H, W]

        Returns:
            Optical flow [B, 2, H, W] at output_level resolution
        """
        # Extract feature pyramids for both images
        features1_pyramid = []
        features2_pyramid = []

        x1 = image1
        x2 = image2

        for level in range(self.num_levels):
            x1 = self.feature_pyramid[level](x1)
            x2 = self.feature_pyramid[level](x2)
            features1_pyramid.append(x1)
            features2_pyramid.append(x2)

        # Coarse-to-fine flow estimation
        flow = None

        for level in range(self.num_levels - 1, self.output_level - 1, -1):
            features1 = features1_pyramid[level]
            features2 = features2_pyramid[level]

            # Upsample flow from previous (coarser) level
            if flow is None:
                # Coarsest level: initialize flow to zero
                batch, _, height, width = features1.shape
                flow = torch.zeros(batch, 2, height, width, device=features1.device)
            else:
                # Upsample flow to current level resolution
                flow = torch.nn.functional.interpolate(
                    flow,
                    size=(features1.shape[2], features1.shape[3]),
                    mode='bilinear',
                    align_corners=True
                ) * 2.0  # Scale flow values

            # Warp features2 using upsampled flow
            features2_warped = self.warp_features(features2, flow)

            # Compute cost volume
            cost_volume = self.compute_cost_volume(features1, features2_warped, self.search_range)

            # Concatenate cost volume, features1, and upsampled flow
            flow_estimator_input = torch.cat([cost_volume, features1, flow], dim=1)

            # Estimate flow residual
            flow_residual = self.flow_estimators[level](flow_estimator_input)

            # Update flow
            flow = flow + flow_residual

        # Apply context network to final flow
        flow = flow + self.context_network(flow)

        # Upsample flow to original image resolution
        # From output_level to level 0
        upsample_factor = 2 ** self.output_level
        flow = torch.nn.functional.interpolate(
            flow,
            size=(image1.shape[2], image1.shape[3]),
            mode='bilinear',
            align_corners=True
        ) * upsample_factor

        return flow


class FlowRankingModel(nn.Module):
    """
    Pairwise ranking model for comparing motion magnitude using optical flow fields.

    Takes two pre-computed optical flow fields as input and predicts which represents
    greater motion magnitude. Uses a lightweight encoder designed for 2-channel flow input.

    Architecture:
        1. Optional average pooling to downsample flow fields (reduces computation)
        2. Lightweight 4-level convolutional encoder
        3. Global average pooling to get feature vectors
        4. Comparison head: concatenate features → MLP → binary logit

    Unlike PairwiseRankingModel (4 RGB images), this model:
        - Takes 2 flow fields (2-channel: dx, dy)
        - Uses smaller custom encoder (~0.5M params vs ResNet34's 21M)
        - No ImageNet pretraining (flow fields have different statistics than RGB)
    """

    def __init__(
        self,
        encoder_channels=[16, 32, 64, 128],
        hidden_dim=256,
        dropout=0.3,
        flow_downsample_factor=2,
        flow_norm_factor=3.0,
    ):
        """
        Initialize flow ranking model.

        Args:
            encoder_channels: List of output channels for each encoder level (default: [16, 32, 64, 128])
            hidden_dim: Hidden dimension for comparison head MLP (default: 256)
            dropout: Dropout rate in comparison head (default: 0.3)
            flow_downsample_factor: Downsample flow by this factor before encoding (default: 2 → 112x112)
                                   Set to 1 for no downsampling, 4 for aggressive downsampling to 56x56
            flow_norm_factor: Divide flow values by this factor for normalization (default: 3.0)
                             Typical flow range is [-10, 10], so dividing by 3 gives [-3.3, 3.3]
        """
        super().__init__()

        self.encoder_channels = encoder_channels
        self.flow_downsample_factor = flow_downsample_factor
        self.flow_norm_factor = flow_norm_factor

        # Optional downsampling layer (average pooling)
        if flow_downsample_factor > 1:
            self.downsample = nn.AvgPool2d(
                kernel_size=flow_downsample_factor,
                stride=flow_downsample_factor,
            )
        else:
            self.downsample = nn.Identity()

        # Lightweight encoder for flow fields
        # Input: 2 channels (dx, dy)
        # Each level: Conv → BatchNorm → LeakyReLU → Conv → BatchNorm → LeakyReLU → Downsample
        self.encoder = self._make_encoder()

        # Final encoder output channels
        self.encoder_out_channels = encoder_channels[-1]

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Comparison head: concat [feat_A, feat_B] → MLP → binary logit
        # Input: 2 * encoder_out_channels
        # Output: 1 (binary logit for P(A > B))
        self.comparison_head = nn.Sequential(
            nn.Linear(2 * self.encoder_out_channels, hidden_dim),
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

    def _make_encoder(self):
        """
        Create lightweight encoder for flow fields.

        Each level:
            - Two conv layers with BatchNorm and LeakyReLU
            - Stride-2 conv for downsampling (halves spatial resolution)

        Returns encoder as ModuleList of levels.
        """
        encoder = nn.ModuleList()

        in_channels = 2  # Flow has 2 channels (dx, dy)

        for level_idx, out_channels in enumerate(self.encoder_channels):
            # First conv: stride=2 for downsampling
            # Second conv: stride=1 for feature refinement
            level = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
            encoder.append(level)
            in_channels = out_channels

        return encoder

    def forward(self, flow_A, flow_B):
        """
        Forward pass for pairwise flow ranking.

        Args:
            flow_A: [batch, 2, H, W] - First flow field (dx, dy)
            flow_B: [batch, 2, H, W] - Second flow field (dx, dy)

        Returns:
            logit: [batch] - Logit for P(A > B), where A and B are motion magnitudes
        """
        # Normalize flow values
        if self.flow_norm_factor != 1.0:
            flow_A = flow_A / self.flow_norm_factor
            flow_B = flow_B / self.flow_norm_factor

        # Downsample flows if requested
        flow_A = self.downsample(flow_A)
        flow_B = self.downsample(flow_B)

        # Encode both flows through shared encoder
        feat_A = flow_A
        feat_B = flow_B

        for level in self.encoder:
            feat_A = level(feat_A)
            feat_B = level(feat_B)

        # Global average pooling: [batch, channels, H, W] → [batch, channels]
        feat_A = self.pool(feat_A).flatten(1)
        feat_B = self.pool(feat_B).flatten(1)

        # Concatenate features and compare
        combined = torch.cat([feat_A, feat_B], dim=1)  # [batch, 2 * encoder_out_channels]
        logit = self.comparison_head(combined)  # [batch, 1]

        return logit.squeeze(1)  # [batch]

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_discriminative_lr_params(self, encoder_lr_early, encoder_lr_late, head_lr):
        """
        Get parameter groups with discriminative learning rates.

        Args:
            encoder_lr_early: Learning rate for early encoder layers
            encoder_lr_late: Learning rate for late encoder layers
            head_lr: Learning rate for comparison head

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Get all encoder parameters
        encoder_params = list(self.encoder.parameters())

        # Split encoder into early and late
        split_idx = len(encoder_params) // 2
        early_params = encoder_params[:split_idx]
        late_params = encoder_params[split_idx:]

        return [
            {'params': early_params, 'lr': encoder_lr_early},
            {'params': late_params, 'lr': encoder_lr_late},
            {'params': self.comparison_head.parameters(), 'lr': head_lr},
        ]


# =============================================================================
# Embedding Autoencoder for Image Representation Learning
# =============================================================================


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


class ContrastiveMotionEncoder(nn.Module):
    """
    Contrastive encoder for motion-aware embeddings.

    Supports two backbone options:
    1. 'resnet18': Pretrained ResNet18 from torchvision (recommended)
    2. 'custom': Custom encoder matching EmbeddingAutoencoder architecture

    Architecture:
        Input: [batch, 3, 224, 224] single frame
        → Backbone (ResNet18 or custom)
        → Embedding: [batch, embedding_dim]
        → Projection Head: Linear → BN → ReLU → Linear
        → Projection: [batch, projection_dim] L2-normalized

    The projection head output is used for contrastive loss during training,
    while the embedding is used for downstream tasks.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        base_channels: int = 64,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        in_channels: int = 3,
        backbone: str = 'resnet18',
        pretrained: bool = True,
    ):
        """
        Initialize the ContrastiveMotionEncoder.

        Args:
            embedding_dim: Dimension of the encoder output embedding
            base_channels: Base number of channels in encoder (only for custom backbone)
            projection_dim: Dimension of the projection head output
            projection_hidden_dim: Hidden dimension in projection head
            in_channels: Number of input image channels (default 3 for RGB)
            backbone: Backbone type ('resnet18' or 'custom')
            pretrained: Whether to use pretrained weights (only for resnet18)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.projection_dim = projection_dim
        self.backbone_type = backbone
        self.pretrained = pretrained

        if backbone == 'resnet18':
            self._build_resnet18_encoder(embedding_dim, pretrained)
        elif backbone == 'custom':
            self._build_custom_encoder(embedding_dim, base_channels, in_channels)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet18' or 'custom'.")

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

        # Initialize projection head
        self._init_projection_head()

    def _build_resnet18_encoder(self, embedding_dim: int, pretrained: bool):
        """Build encoder using pretrained ResNet18 backbone."""
        import torchvision.models as models

        # Load ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
            print("Using pretrained ResNet18 backbone (ImageNet weights)")
        else:
            resnet = models.resnet18(weights=None)
            print("Using ResNet18 backbone with random initialization")

        # Extract layers (remove final FC and avgpool)
        self.backbone = nn.Sequential(
            resnet.conv1,      # 224 -> 112
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # 112 -> 56
            resnet.layer1,     # 56 -> 56, 64 channels
            resnet.layer2,     # 56 -> 28, 128 channels
            resnet.layer3,     # 28 -> 14, 256 channels
            resnet.layer4,     # 14 -> 7, 512 channels
        )

        # Global pooling and embedding projection
        self.enc_pool = nn.AdaptiveAvgPool2d(1)
        self.enc_fc = nn.Linear(512, embedding_dim)  # ResNet18 outputs 512 channels

        # Initialize the FC layer
        nn.init.xavier_uniform_(self.enc_fc.weight)
        nn.init.zeros_(self.enc_fc.bias)

    def _build_custom_encoder(self, embedding_dim: int, base_channels: int, in_channels: int):
        """Build custom encoder (same as EmbeddingAutoencoder)."""
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        # Build sequential backbone
        self.backbone = nn.Sequential(
            # 224x224x3 → 112x112x64
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResBlock(c1),
            # 112x112x64 → 56x56x128
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ResBlock(c2),
            # 56x56x128 → 28x28x256
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            ResBlock(c3),
            # 28x28x256 → 14x14x512
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            ResBlock(c4),
            # 14x14x512 → 7x7x512
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # Global pooling and embedding projection
        self.enc_pool = nn.AdaptiveAvgPool2d(1)
        self.enc_fc = nn.Linear(c5, embedding_dim)

        # Initialize custom backbone
        self._init_custom_backbone()

    def _init_custom_backbone(self):
        """Initialize weights for custom backbone."""
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.enc_fc.weight)
        nn.init.zeros_(self.enc_fc.bias)

    def _init_projection_head(self):
        """Initialize projection head weights."""
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to embedding vector.

        Args:
            x: Input image tensor [batch, 3, 224, 224]

        Returns:
            Embedding tensor [batch, embedding_dim]
        """
        x = self.backbone(x)
        x = self.enc_pool(x)
        x = x.flatten(1)
        z = self.enc_fc(x)
        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project embedding for contrastive loss.

        Args:
            z: Embedding tensor [batch, embedding_dim]

        Returns:
            L2-normalized projection [batch, projection_dim]
        """
        h = self.projection_head(z)
        return F.normalize(h, dim=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning both embedding and projection.

        Args:
            x: Input image tensor [batch, 3, 224, 224]

        Returns:
            Tuple of (embedding, projection):
                - embedding: [batch, embedding_dim] for downstream tasks
                - projection: [batch, projection_dim] L2-normalized for contrastive loss
        """
        z = self.encode(x)
        h = self.project(z)
        return z, h

    def load_pretrained_encoder(self, checkpoint_path: str):
        """
        Load encoder weights from a pre-trained EmbeddingAutoencoder checkpoint.

        Only works with 'custom' backbone. For 'resnet18', use pretrained=True instead.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if self.backbone_type == 'resnet18':
            print("Warning: load_pretrained_encoder is designed for custom backbone.")
            print("For resnet18, pretrained weights are loaded automatically via pretrained=True")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter to encoder-only keys (those starting with 'enc_')
        encoder_keys = {k: v for k, v in state_dict.items() if k.startswith('enc_')}

        # Load encoder weights
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)

        print(f"Loaded encoder weights from {checkpoint_path}")
        print(f"  Loaded {len(encoder_keys)} encoder parameters")
        if missing:
            # Filter out projection head keys from missing (expected)
            missing_non_proj = [k for k in missing if not k.startswith('projection_head')]
            if missing_non_proj:
                print(f"  Missing encoder keys: {missing_non_proj}")

    def freeze_encoder(self):
        """Freeze encoder/backbone parameters (train only projection head and FC)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Froze backbone parameters ({sum(p.numel() for p in self.backbone.parameters()):,} params)")

    def unfreeze_encoder(self):
        """Unfreeze encoder/backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_early_layers(self, freeze_until: str = 'layer2'):
        """
        Freeze early layers of ResNet18 backbone (fine-tune later layers).

        Args:
            freeze_until: Freeze layers up to and including this layer.
                         Options: 'conv1', 'layer1', 'layer2', 'layer3'
        """
        if self.backbone_type != 'resnet18':
            print("freeze_early_layers only applies to resnet18 backbone")
            return

        freeze_layers = {
            'conv1': ['0', '1', '2', '3'],  # conv1, bn1, relu, maxpool
            'layer1': ['0', '1', '2', '3', '4'],
            'layer2': ['0', '1', '2', '3', '4', '5'],
            'layer3': ['0', '1', '2', '3', '4', '5', '6'],
        }

        if freeze_until not in freeze_layers:
            raise ValueError(f"Unknown layer: {freeze_until}. Choose from {list(freeze_layers.keys())}")

        layers_to_freeze = freeze_layers[freeze_until]
        frozen_count = 0

        for name, param in self.backbone.named_parameters():
            layer_idx = name.split('.')[0]
            if layer_idx in layers_to_freeze:
                param.requires_grad = False
                frozen_count += param.numel()

        print(f"Froze layers up to {freeze_until} ({frozen_count:,} parameters)")


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from "Supervised Contrastive Learning" (Khosla et al., 2020).

    For motion-aware embeddings, labels are motion state patterns encoded as integers (0-31).
    Frames with the same motion state are considered positive pairs.

    Loss formula:
        L = -Σᵢ (1/|P(i)|) Σₚ∈P(i) log[ exp(zᵢ·zₚ/τ) / Σₐ∈A(i) exp(zᵢ·zₐ/τ) ]

    Where:
        - P(i) = set of positives for anchor i (same label)
        - A(i) = all samples except anchor i
        - τ = temperature parameter

    Optional cross-subject mining: only count positives from different subjects
    to encourage subject-invariant representations.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Initialize the SupConLoss.

        Args:
            temperature: Temperature parameter for scaling similarities
            base_temperature: Base temperature for loss scaling
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: [batch, dim] L2-normalized projected embeddings
            labels: [batch] motion state as integer (0-31)
            subject_ids: [batch] subject IDs for cross-subject mining (optional)

        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute similarity matrix: [batch, batch]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for same label (positive pairs)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)

        # Optional: Cross-subject mining (only positives from different subjects)
        if subject_ids is not None:
            subject_ids = subject_ids.contiguous().view(-1, 1)
            mask_diff_subject = (~torch.eq(subject_ids, subject_ids.T)).float().to(device)
            mask_pos = mask_pos * mask_diff_subject

        # Remove self-comparison from positives
        mask_self = torch.eye(batch_size, dtype=torch.float32, device=device)
        mask_pos = mask_pos - mask_self

        # Number of positives per anchor
        num_pos = mask_pos.sum(1)

        # For numerical stability: subtract max from logits
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute log_prob = logits - log(sum(exp(logits))) excluding self
        exp_logits = torch.exp(logits) * (1 - mask_self)  # Exclude self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (num_pos + 1e-8)

        # Loss (negative log-likelihood)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # Only count anchors with at least one positive
        valid_mask = (num_pos > 0).float()
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss


class TransformerMotionClassifier(nn.Module):
    """
    Transformer-based motion classifier using frozen embedding encoder.

    Takes sequences of pre-computed embeddings from EmbeddingAutoencoder,
    applies a causal transformer, and outputs per-frame motion probabilities
    for 5 fingers.

    Architecture:
        - Input: [batch, seq_len, embedding_dim] pre-computed embeddings
        - Learnable positional encoding
        - 4x TransformerEncoderLayer with causal masking
        - Classification head applied at each position
        - Output: [batch, seq_len, 5] logits

    The causal mask ensures each position only attends to previous positions,
    making this suitable for real-time inference.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_fingers: int = 5,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        """
        Initialize the TransformerMotionClassifier.

        Args:
            embedding_dim: Dimension of input embeddings (should match EmbeddingAutoencoder)
            num_fingers: Number of output classes (default 5 for 5 fingers)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            ffn_dim: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length (for positional encoding)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_fingers = num_fingers
        self.max_seq_len = max_seq_len

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embedding_dim) * 0.02
        )

        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head (applied at each position)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, num_fingers),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, embedding_dim]

        Returns:
            logits: Motion probabilities (before sigmoid) [batch, seq_len, num_fingers]
        """
        batch, seq_len, _ = embeddings.shape

        # Add positional encoding
        x = embeddings + self.pos_embedding[:, :seq_len, :]

        # Create causal mask (upper triangular = masked)
        # True values are masked out (not attended to)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=embeddings.device),
            diagonal=1
        ).bool()

        # Apply transformer
        x = self.transformer(x, mask=causal_mask)

        # Classify each position
        logits = self.classifier(x)  # [batch, seq_len, num_fingers]

        return logits

    def get_attention_weights(self, embeddings: torch.Tensor) -> list:
        """
        Get attention weights for visualization.

        Note: This requires accessing internal transformer layers.

        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, embedding_dim]

        Returns:
            List of attention weight tensors, one per layer
        """
        # Add positional encoding
        batch, seq_len, _ = embeddings.shape
        x = embeddings + self.pos_embedding[:, :seq_len, :]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=embeddings.device),
            diagonal=1
        ).bool()

        # Collect attention weights from each layer
        attention_weights = []
        for layer in self.transformer.layers:
            # Self-attention with need_weights=True
            attn_output, attn_weights_layer = layer.self_attn(
                x, x, x,
                attn_mask=causal_mask,
                need_weights=True,
                average_attn_weights=True
            )
            attention_weights.append(attn_weights_layer.detach())

            # Apply rest of the layer
            x = layer(x, src_mask=causal_mask)

        return attention_weights


class FlexClassifier(nn.Module):
    """
    Second-stage model that classifies per-finger flexion from P(flex>rest) signals.

    Takes temporal windows of 5-channel probability signals from a frozen
    multi-finger ranking model and outputs per-frame binary flex probabilities.

    Architecture:
        - Input projection: Linear(5, hidden_dim)
        - Learnable positional encoding
        - Causal Transformer encoder (N layers)
        - Classification head at each position
        - Output: [batch, window_size, 5] logits

    The causal mask ensures each position only attends to current and past
    positions, enabling real-time inference.
    """

    FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        num_fingers: int = 5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_fingers = num_fingers
        self.max_seq_len = max_seq_len

        # Project 5-dim input to hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Per-position classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, num_fingers),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: [batch, window_size, 5] P(flex>rest) probabilities

        Returns:
            logits: [batch, window_size, 5] flex classification logits
        """
        batch, seq_len, _ = signals.shape

        # Project input
        x = self.input_projection(signals)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=signals.device),
            diagonal=1
        ).bool()

        # Transformer
        x = self.transformer(x, mask=causal_mask)

        # Classify each position
        logits = self.classifier(x)  # [batch, seq_len, 5]

        return logits


class FlexClassifierGRU(nn.Module):
    """
    Lightweight GRU-based flex classifier with the same interface as FlexClassifier.

    Takes [batch, T, 5] P(flex>rest) signals and outputs [batch, T, 5] logits.
    Uses a bidirectional GRU followed by a per-position classification head.
    """

    FINGER_NAMES = ['index', 'middle', 'ring', 'pinky', 'thumb']

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_fingers: int = 5,
        bidirectional: bool = True,
        # Accept and ignore transformer-specific args for compatibility
        num_heads: int = 4,
        ffn_dim: int = 128,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_fingers = num_fingers
        self.bidirectional = bidirectional

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Classification head
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_fingers),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: [batch, T, 5] P(flex>rest) probabilities

        Returns:
            logits: [batch, T, 5] flex classification logits
        """
        x = self.input_projection(signals)  # [batch, T, hidden_dim]
        x, _ = self.gru(x)  # [batch, T, hidden_dim * (2 if bidir)]
        logits = self.classifier(x)  # [batch, T, 5]
        return logits


# =============================================================================
# CBMF Baseline (Configuration-Based Multi-Frame)
# =============================================================================
# Faithful reimplementation of the CBMF model from:
#   Zadok et al., "Inferring Fine Finger Motions for Prosthetic Control",
#   ASME J. Biomech. Eng., 2025.
#
# Two-stage model:
#   Stage 1: UNet encoder + GRU predicts per-frame joint angles (MSE loss)
#   Stage 2: Decoder GRU classifies flex from predicted configs + residual
#            encoder features (lambda*MSE + BCE loss)
# =============================================================================


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
