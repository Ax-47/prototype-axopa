import copy

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Hierarchical JEPA
#
# Level 0 ("local"):  a conv trunk turns the masked image into a 7x7 grid of patch
#                      tokens (49 tokens, each `token_dim`-wide). This is the
#                      fine-grained, spatially-resolved representation.
#
# Level 1 ("global"):  a small transformer aggregates the 49 local tokens (via a
#                      learned CLS token) into one `latent_dim` global vector. This is
#                      the coarse, scene-level representation used for classification.
#
# Prediction (and the EMA target) happen at BOTH levels:
#   - the global predictor predicts the next global latent (used for the classifier
#     and for imagining the full hidden image),
#   - the local predictor predicts the next grid of local tokens (used to imagine the
#     next partial observation).
# --------------------------------------------------------------------------------------


class LocalEncoder(nn.Module):
    """Level 0: image+mask -> a 7x7 grid of local patch tokens."""

    def __init__(self, token_dim: int = 128):
        super().__init__()

        self.token_dim = token_dim

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, token_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Raw conv+ReLU activations are unbounded. Unlike the global path
        # (where the transformer's internal LayerNorm keeps things in a sane
        # range), these tokens feed the local predictor/decoder directly, so
        # without this norm large activations can saturate the decoder's
        # final Sigmoid to ~0 everywhere (a uniformly black image).
        self.token_norm = nn.LayerNorm(token_dim)

    def forward(self, image, mask):
        x = torch.cat(
            [image, mask],
            dim=1,
        )

        feature_map = self.net(x)  # (B, token_dim, 7, 7)

        batch_size, channels, height, width = feature_map.shape

        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, 49, token_dim)

        tokens = self.token_norm(tokens)

        return tokens, (height, width)


class GlobalAggregator(nn.Module):
    """Level 1: local tokens -> one global scene-level latent (via a CLS token)."""

    def __init__(
        self,
        token_dim: int = 128,
        latent_dim: int = 128,
        num_tokens: int = 49,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 2,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_proj = nn.Linear(token_dim, latent_dim)

    def forward(self, tokens):
        batch_size = tokens.shape[0]

        tokens = tokens + self.pos_embedding

        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat(
            [cls, tokens],
            dim=1,
        )

        x = self.transformer(x)

        z_global = self.out_proj(x[:, 0])

        return z_global


class HierarchicalEncoder(nn.Module):
    """Wraps the local encoder + global aggregator into a single 2-level encoder."""

    def __init__(
        self,
        latent_dim: int = 128,
        token_dim: int = 128,
        num_tokens: int = 49,
    ):
        super().__init__()

        self.local_encoder = LocalEncoder(token_dim=token_dim)

        self.global_aggregator = GlobalAggregator(
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_tokens=num_tokens,
        )

    def forward(self, image, mask):
        tokens, spatial_shape = self.local_encoder(
            image,
            mask,
        )

        z_global = self.global_aggregator(tokens)

        return z_global, tokens


class ActionEncoder(nn.Module):
    def __init__(
        self,
        num_actions: int = 10,
        action_dim: int = 32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_actions,
            action_dim,
        )

    def forward(self, action):
        return self.embedding(action)


class GlobalPredictor(nn.Module):
    """Level 1 predictor: predicts the next global latent given an action."""

    def __init__(
        self,
        latent_dim=128,
        action_dim=32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                latent_dim,
            ),
        )

    def forward(
        self,
        z,
        action_embedding,
    ):
        x = torch.cat(
            [z, action_embedding],
            dim=-1,
        )

        return self.net(x)


class LocalPredictor(nn.Module):
    """Level 0 predictor: predicts the next grid of local tokens given an action.

    The action embedding is broadcast to every token position and the predictor
    is applied per-token (with a residual connection), so it predicts how each
    patch's token changes once the chosen cell is revealed.
    """

    def __init__(
        self,
        token_dim=128,
        action_dim=32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                token_dim + action_dim,
                token_dim * 2,
            ),
            nn.ReLU(),
            nn.Linear(
                token_dim * 2,
                token_dim,
            ),
        )

        # Keeps the residual output bounded (matching the LayerNorm already
        # applied in LocalEncoder), so magnitudes don't drift and saturate
        # the decoder's Sigmoid after repeated predict() calls.
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens,
        action_embedding,
    ):
        batch_size, num_tokens, token_dim = tokens.shape

        action_expanded = action_embedding.unsqueeze(1).expand(
            batch_size,
            num_tokens,
            -1,
        )

        x = torch.cat(
            [tokens, action_expanded],
            dim=-1,
        )

        return self.norm(tokens + self.net(x))


class Classifier(nn.Module):
    """Operates on the global (level-1) latent."""

    def __init__(
        self,
        latent_dim=128,
        num_classes=10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, z):
        return self.net(z)


class SpatialDecoder(nn.Module):
    """Level-0 decoder: predicted local tokens -> next partial observation.

    Unlike the old vector-based decoder, this consumes the (B, 49, token_dim)
    grid of local tokens directly, reshaped back into its native 7x7 spatial
    layout, since that's already where the local predictor operates.
    """

    def __init__(
        self,
        token_dim=128,
        grid_size=7,
    ):
        super().__init__()

        self.grid_size = grid_size

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                token_dim,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, tokens):
        batch_size, num_tokens, token_dim = tokens.shape

        x = tokens.transpose(1, 2).reshape(
            batch_size,
            token_dim,
            self.grid_size,
            self.grid_size,
        )

        return self.deconv(x)


class FullImageDecoder(nn.Module):
    """Level-1 decoder: predicted global latent -> complete hidden MNIST image."""

    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                128 * 7 * 7,
            ),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)

        x = x.view(
            -1,
            128,
            7,
            7,
        )

        return self.deconv(x)


class ActiveJEPA(nn.Module):
    """Hierarchical Active JEPA: a local (patch-token) level and a global
    (scene) level, each with its own predictor, decoder, and EMA target.
    """

    def __init__(
        self,
        latent_dim=128,
        token_dim=128,
        action_dim=32,
        num_actions=10,
        num_classes=10,
        num_tokens=49,
    ):
        super().__init__()

        self.encoder = HierarchicalEncoder(
            latent_dim=latent_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
        )

        self.target_encoder = copy.deepcopy(self.encoder)

        self.action_encoder = ActionEncoder(
            num_actions=num_actions,
            action_dim=action_dim,
        )

        # Level 1 (global) predictor + decoder.
        self.global_predictor = GlobalPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        self.classifier = Classifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        self.full_image_decoder = FullImageDecoder(latent_dim=latent_dim)

        # Level 0 (local) predictor + decoder.
        self.local_predictor = LocalPredictor(
            token_dim=token_dim,
            action_dim=action_dim,
        )

        self.decoder = SpatialDecoder(token_dim=token_dim)

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_target(
        self,
        momentum=0.996,
    ):
        for online, target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data.mul_(momentum)
            target.data.add_(
                online.data,
                alpha=1.0 - momentum,
            )

    def encode(
        self,
        image,
        mask,
    ):
        """Returns (z_global, tokens): the global latent and the local token grid."""

        return self.encoder(
            image,
            mask,
        )

    @torch.no_grad()
    def encode_target(
        self,
        image,
        mask,
    ):
        """Returns (z_global_target, tokens_target) from the EMA target encoder."""

        return self.target_encoder(
            image,
            mask,
        )

    def predict(
        self,
        z,
        tokens,
        action,
    ):
        """Runs both levels of prediction for a given action.

        Returns (z_pred_global, tokens_pred).
        """

        action_embedding = self.action_encoder(action)

        z_pred = self.global_predictor(
            z,
            action_embedding,
        )

        tokens_pred = self.local_predictor(
            tokens,
            action_embedding,
        )

        return z_pred, tokens_pred

    def forward(
        self,
        image,
        mask,
        next_image,
        next_mask,
        full_image,
        action,
    ):
        z, tokens = self.encoder(
            image,
            mask,
        )

        with torch.no_grad():
            z_target, tokens_target = self.target_encoder(
                next_image,
                next_mask,
            )

        action_embedding = self.action_encoder(action)

        z_pred = self.global_predictor(
            z,
            action_embedding,
        )

        tokens_pred = self.local_predictor(
            tokens,
            action_embedding,
        )

        logits_current = self.classifier(z)

        logits_pred = self.classifier(z_pred)

        # Level 0: predict next partial observation from the local tokens.
        reconstruction = self.decoder(tokens_pred)

        # Level 1: predict the complete hidden image from the global latent.
        full_reconstruction = self.full_image_decoder(z_pred)

        return {
            "z": z,
            "z_target": z_target,
            "z_pred": z_pred,
            "tokens": tokens,
            "tokens_target": tokens_target,
            "tokens_pred": tokens_pred,
            "logits_current": logits_current,
            "logits_pred": logits_pred,
            "reconstruction": reconstruction,
            "full_reconstruction": full_reconstruction,
        }
