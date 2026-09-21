import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Hierarchical JEPA (3 levels)
#
# Level 0 ("local"):  a conv trunk turns the masked image into a 7x7 grid of patch
#                      tokens (49 tokens, each `token_dim`-wide). Fine-grained,
#                      spatially-resolved representation.
#
# Level 1 ("mid"):    the 49 local tokens are pooled down into a 3x3 grid of
#                      coarser "region" tokens (9 tokens). A middle rung between
#                      individual patches and the whole scene. This also happens to
#                      be the exact same 3x3 grid the 9 reveal-actions index into.
#
# Level 2 ("global"): a small transformer aggregates the 9 mid tokens (via a
#                      learned CLS token) into one `latent_dim` global vector. The
#                      coarsest, scene-level representation, used for classification.
#
# Prediction (and the EMA target) happen at ALL THREE levels:
#   - the global predictor predicts the next global latent (used for the classifier
#     and for imagining the full hidden image),
#   - the mid predictor predicts the next grid of region tokens (used to imagine a
#     coarser next partial observation),
#   - the local predictor predicts the next grid of local tokens (used to imagine
#     the finest-grained next partial observation).
#
# Beyond the base 3-level predict/decode pipeline, this version adds:
#   1. Position-gated mid/local predictors: since action 0-8 IS a 3x3 grid
#      cell, the mid/local predictors are told (via a gate derived straight
#      from the action, not learned) which token(s) the reveal actually
#      touches, instead of broadcasting the action identically everywhere
#      and hoping the network learns the routing from scratch.
#   2. An explicit STOP action (index 9): all three predictors are exact
#      identity maps when action == STOP, rather than learned to be
#      approximately identity.
#   3. A residual + LayerNorm GlobalPredictor, consistent with the other
#      two levels (previously it was a plain unnormalized MLP).
#   4. A cosine EMA momentum schedule for the target encoder (BYOL-style),
#      instead of one constant momentum for all of training.
#   5. Top-down FiLM modulation: the global latent modulates the mid tokens,
#      and the mid tokens modulate the local tokens, before each level's own
#      predictor runs -- a top-down pathway alongside the bottom-up encoder.
#   6. `ActiveJEPA.select_action`, a zero-extra-training heuristic that
#      uses predicted-change magnitude as an information-gain proxy (with
#      reveal-mask masking + step-budget awareness) -- a cheap fallback/
#      sanity-check next to a properly trained policy (this repo's
#      environment.py + rl.py + train_rl.py train a real DQN for that).
#   7. `ActiveJEPA.rollout` for multi-step latent-space imagination, with a
#      docstring note on autoregressive drift.
#   8. `AccuracyHead`: a small head that predicts, for every action, the
#      probability that the classifier will be CORRECT after that action
#      (for STOP: correct right now). It is trained on the real next
#      observation, so it estimates the expected value of information
#      directly. Classifying an "imagined" z_pred does not: z_pred is a
#      regression-to-the-mean latent that carries no information beyond z,
#      and softmax(classifier(E[z'])) != E[softmax(classifier(z'))].
#   9. No dropout in the encoder (and the target encoder is always kept in
#      eval mode), so the EMA targets are deterministic.
#  10. Position gates are derived from the real pixel footprint of each cell
#      and dilated by `gate_dilation` local tokens, because the conv trunk's
#      receptive field (~18 px) and the overlapping adaptive-pool windows
#      mean revealing one cell also changes the tokens around it.
# --------------------------------------------------------------------------------------

MID_GRID_SIZE = 3
LOCAL_GRID_SIZE = 7
IMAGE_SIZE = 28
NUM_MID_CELLS = MID_GRID_SIZE * MID_GRID_SIZE  # 9 -- also the number of reveal actions
STOP_ACTION = NUM_MID_CELLS  # 9
NUM_ACTIONS = NUM_MID_CELLS + 1  # 10: 9 reveals + STOP

# Pixel boundaries of the 3x3 reveal grid (must match data.CELL_ROWS / CELL_COLS)
# and one pixel in the middle of each row/column, used to read the "which cells
# are open" mask back out of a (B, 1, 28, 28) pixel mask.
CELL_BOUNDS = (0, 9, 18, IMAGE_SIZE)
CELL_CENTERS = [4, 13, 23]


def build_gate_tables(dilation: int = 1):
    """Precomputes, for every reveal action 0-8, which mid tokens (3x3 grid)
    and which local tokens (7x7 grid) the reveal can change.

    Returns (mid_table (9, 9), local_table (9, 49)), both 0/1 floats.

      1. Start from the exact 28x28 pixel footprint of the cell.
      2. A local token (4x4 pixels) is gated if it overlaps that footprint.
      3. Dilate by `dilation` tokens (the conv trunk's receptive field is
         ~18 px, so tokens next to the cell change too).
      4. A mid token is gated if any local token in its (adaptive-pool)
         window is gated.
    """
    coverage = torch.zeros(NUM_MID_CELLS, 1, IMAGE_SIZE, IMAGE_SIZE)

    for cell in range(NUM_MID_CELLS):
        row, col = divmod(cell, MID_GRID_SIZE)

        coverage[
            cell,
            0,
            CELL_BOUNDS[row] : CELL_BOUNDS[row + 1],
            CELL_BOUNDS[col] : CELL_BOUNDS[col + 1],
        ] = 1.0

    stride = IMAGE_SIZE // LOCAL_GRID_SIZE  # 4 pixels per local token

    local = (F.avg_pool2d(coverage, stride) > 0).float()  # (9, 1, 7, 7)

    if dilation > 0:
        local = F.max_pool2d(
            local,
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        )

    mid = F.adaptive_max_pool2d(
        local,
        (MID_GRID_SIZE, MID_GRID_SIZE),
    )  # (9, 1, 3, 3)

    return mid.flatten(1), local.flatten(1)


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
        # without this norm large activations can saturate a decoder's
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


class MidPooler(nn.Module):
    """Level 1: pools the 49 local (7x7) tokens down into a 3x3 grid of
    coarser "region" tokens (9 tokens). Sits between the fine-grained local
    tokens and the single global scene vector.
    """

    def __init__(
        self,
        token_dim: int = 128,
        local_grid_size: int = LOCAL_GRID_SIZE,
        mid_grid_size: int = MID_GRID_SIZE,
    ):
        super().__init__()

        self.token_dim = token_dim

        self.local_grid_size = local_grid_size

        self.mid_grid_size = mid_grid_size

        self.pool = nn.AdaptiveAvgPool2d((mid_grid_size, mid_grid_size))

        self.proj = nn.Linear(token_dim, token_dim)

        # Same reasoning as LocalEncoder's token_norm: keep the pooled/
        # projected region tokens bounded before they feed the mid
        # predictor/decoder and the global aggregator.
        self.token_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens_local):
        batch_size, num_tokens, token_dim = tokens_local.shape

        x = tokens_local.transpose(1, 2).reshape(
            batch_size,
            token_dim,
            self.local_grid_size,
            self.local_grid_size,
        )

        x = self.pool(x)  # (B, token_dim, 3, 3)

        tokens_mid = x.flatten(2).transpose(1, 2)  # (B, 9, token_dim)

        tokens_mid = self.proj(tokens_mid)

        tokens_mid = self.token_norm(tokens_mid)

        return tokens_mid


class GlobalAggregator(nn.Module):
    """Level 2: mid tokens -> one global scene-level latent (via a CLS token)."""

    def __init__(
        self,
        token_dim: int = 128,
        latent_dim: int = 128,
        num_tokens: int = NUM_MID_CELLS,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 2,
            # nn.TransformerEncoderLayer defaults to dropout=0.1. With
            # model.train() that also put the *target* encoder in train mode,
            # so the EMA targets were noisy. No dropout -> deterministic targets.
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_proj = nn.Linear(token_dim, latent_dim)

    def forward(self, tokens_mid):
        batch_size = tokens_mid.shape[0]

        tokens_mid = tokens_mid + self.pos_embedding

        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat(
            [cls, tokens_mid],
            dim=1,
        )

        x = self.transformer(x)

        z_global = self.out_proj(x[:, 0])

        return z_global


class HierarchicalEncoder(nn.Module):
    """Wraps the local encoder + mid pooler + global aggregator into a
    single 3-level encoder.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        token_dim: int = 128,
        local_grid_size: int = LOCAL_GRID_SIZE,
        mid_grid_size: int = MID_GRID_SIZE,
    ):
        super().__init__()

        self.local_encoder = LocalEncoder(token_dim=token_dim)

        self.mid_pooler = MidPooler(
            token_dim=token_dim,
            local_grid_size=local_grid_size,
            mid_grid_size=mid_grid_size,
        )

        self.global_aggregator = GlobalAggregator(
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_tokens=mid_grid_size * mid_grid_size,
        )

    def forward(self, image, mask):
        tokens_local, spatial_shape = self.local_encoder(
            image,
            mask,
        )

        tokens_mid = self.mid_pooler(tokens_local)

        z_global = self.global_aggregator(tokens_mid)

        return z_global, tokens_mid, tokens_local


class ActionEncoder(nn.Module):
    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        action_dim: int = 32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_actions,
            action_dim,
        )

    def forward(self, action):
        return self.embedding(action)


class FiLM(nn.Module):
    """Top-down modulation: a coarser-level representation (the "context")
    produces a per-channel scale/shift applied to a finer-level
    representation (the "target") before the finer level's own predictor
    runs. This gives the hierarchy an explicit top-down pathway -- coarse
    belief conditions fine-grained processing -- alongside the bottom-up
    encoder, which is closer to the predictive-coding picture of top-down
    priors than a purely feedforward encoder/predictor stack.
    """

    def __init__(self, context_dim: int, target_dim: int):
        super().__init__()

        self.to_gamma_beta = nn.Linear(context_dim, target_dim * 2)

        # Zero-init so FiLM starts as a no-op (gamma=0 -> scale=1, beta=0)
        # and the model has to learn to actually use the top-down signal,
        # rather than starting from a random perturbation of the target.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, target, context):
        gamma, beta = self.to_gamma_beta(context).chunk(2, dim=-1)

        if target.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return target * (1 + gamma) + beta


class GlobalPredictor(nn.Module):
    """Level 2 predictor: predicts the next global latent given an action.

    Residual + LayerNorm, matching Mid/LocalPredictor below (the network
    predicts a *delta* on top of z, not z_next from scratch), and an exact
    identity map when action == STOP.
    """

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

        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z,
        action_embedding,
        is_stop,
    ):
        x = torch.cat(
            [z, action_embedding],
            dim=-1,
        )

        pred = self.norm(z + self.net(x))

        return torch.where(is_stop.view(-1, 1), z, pred)


class MidPredictor(nn.Module):
    """Level 1 predictor: predicts the next grid of region-level tokens
    given an action. Same broadcast + residual + LayerNorm pattern as
    LocalPredictor, just operating on the coarser 3x3 grid.

    Two additions over a plain broadcast-to-every-token predictor:
      - `gate` (B, 9): a one-hot-ish weight (built from the action, not
        learned -- see `ActiveJEPA._action_gates`) telling the predictor
        which region the action actually reveals, so the delta lands there
        rather than being applied identically everywhere.
      - `is_stop` (B,): forces an exact identity map for STOP actions,
        returning `tokens_raw` (the un-modulated, bottom-up tokens)
        untouched rather than something merely close to identity.
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

        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens_ctx,
        tokens_raw,
        action_embedding,
        gate,
        is_stop,
    ):
        batch_size, num_tokens, token_dim = tokens_ctx.shape

        action_expanded = action_embedding.unsqueeze(1).expand(
            batch_size,
            num_tokens,
            -1,
        )

        x = torch.cat(
            [tokens_ctx, action_expanded],
            dim=-1,
        )

        delta = self.net(x)

        pred = self.norm(tokens_ctx + gate.unsqueeze(-1) * delta)

        return torch.where(is_stop.view(-1, 1, 1), tokens_raw, pred)


class LocalPredictor(nn.Module):
    """Level 0 predictor: predicts the next grid of local tokens given an action.

    The action embedding is broadcast to every token position and the predictor
    is applied per-token (with a residual connection), so it predicts how each
    patch's token changes once the chosen cell is revealed. `gate` and
    `is_stop` play the same role as in MidPredictor -- see there for details.
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
        tokens_ctx,
        tokens_raw,
        action_embedding,
        gate,
        is_stop,
    ):
        batch_size, num_tokens, token_dim = tokens_ctx.shape

        action_expanded = action_embedding.unsqueeze(1).expand(
            batch_size,
            num_tokens,
            -1,
        )

        x = torch.cat(
            [tokens_ctx, action_expanded],
            dim=-1,
        )

        delta = self.net(x)

        pred = self.norm(tokens_ctx + gate.unsqueeze(-1) * delta)

        return torch.where(is_stop.view(-1, 1, 1), tokens_raw, pred)


class Classifier(nn.Module):
    """Operates on the global (level-2) latent."""

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


class AccuracyHead(nn.Module):
    """Predicts P(classifier is correct) for every action, from the current
    global latent and which cells are already open.

      output[:, 0..8] : P(correct) AFTER revealing that cell
      output[:, 9]    : P(correct) if we STOP right now

    Returns logits (apply a sigmoid for probabilities). Trained with BCE
    against whether the classifier really is correct on the real next
    observation, so it learns E[correct | z, opened, action] -- the quantity
    the RL agent and the greedy planner actually need -- instead of the
    confidence of an averaged, imagined latent.
    """

    def __init__(
        self,
        latent_dim=128,
        num_cells=NUM_MID_CELLS,
        num_actions=NUM_ACTIONS,
        hidden_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim + num_cells,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                num_actions,
            ),
        )

    def forward(self, z, opened_mask):
        return self.net(
            torch.cat(
                [z, opened_mask],
                dim=-1,
            )
        )


class SpatialDecoder(nn.Module):
    """Level-0 decoder: predicted local tokens -> next partial observation
    (finest resolution).

    Consumes the (B, 49, token_dim) grid of local tokens directly, reshaped
    back into its native 7x7 spatial layout, since that's already where the
    local predictor operates.
    """

    def __init__(
        self,
        token_dim=128,
        grid_size=LOCAL_GRID_SIZE,
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


class MidDecoder(nn.Module):
    """Level-1 decoder: predicted region tokens -> next partial observation,
    at a coarser (3x3-region) resolution than SpatialDecoder's 7x7.

    7 -> 28 needed two stride-2 steps (x4). 3 -> 28 isn't a power of 2, so
    this uses two stride-3 ConvTranspose2d steps instead: 3 -> 9 -> 28.
    """

    def __init__(
        self,
        token_dim=128,
        grid_size=MID_GRID_SIZE,
    ):
        super().__init__()

        self.grid_size = grid_size

        self.deconv = nn.Sequential(
            # 3 -> 9: out = (3-1)*3 - 2*1 + 5 = 9
            nn.ConvTranspose2d(
                token_dim,
                64,
                kernel_size=5,
                stride=3,
                padding=1,
            ),
            nn.ReLU(),
            # 9 -> 28: out = (9-1)*3 - 2*1 + 6 = 28
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=6,
                stride=3,
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
    """Level-2 decoder: predicted global latent -> complete hidden MNIST image."""

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
    """3-level Hierarchical Active JEPA: local (patch), mid (region), and
    global (scene) levels, each with its own predictor, decoder, and EMA
    target (the EMA target covers the whole encoder, i.e. all three levels
    at once, since they're one nested module).
    """

    STOP_ACTION = STOP_ACTION

    def __init__(
        self,
        latent_dim=128,
        token_dim=128,
        action_dim=32,
        num_actions=NUM_ACTIONS,
        num_classes=10,
        local_grid_size=LOCAL_GRID_SIZE,
        mid_grid_size=MID_GRID_SIZE,
        gate_dilation=1,
    ):
        super().__init__()

        self.encoder = HierarchicalEncoder(
            latent_dim=latent_dim,
            token_dim=token_dim,
            local_grid_size=local_grid_size,
            mid_grid_size=mid_grid_size,
        )

        self.target_encoder = copy.deepcopy(self.encoder)

        self.action_encoder = ActionEncoder(
            num_actions=num_actions,
            action_dim=action_dim,
        )

        # Top-down pathway: global belief modulates mid tokens, mid belief
        # modulates local tokens, before each level's own predictor runs.
        self.global_to_mid_film = FiLM(context_dim=latent_dim, target_dim=token_dim)

        self.mid_to_local_film = FiLM(context_dim=token_dim, target_dim=token_dim)

        # Level 2 (global) predictor + decoder.
        self.global_predictor = GlobalPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        self.classifier = Classifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        # Expected-accuracy head (see AccuracyHead). Reads z.detach(), so it
        # never shapes the representation -- it only learns to read it.
        self.accuracy_head = AccuracyHead(
            latent_dim=latent_dim,
            num_cells=NUM_MID_CELLS,
            num_actions=num_actions,
        )

        # Constant position-gate lookup tables (see build_gate_tables).
        # Not persistent: derived from constants, so old checkpoints and new
        # ones stay interchangeable regardless of `gate_dilation`.
        mid_gate_table, local_gate_table = build_gate_tables(gate_dilation)

        self.register_buffer("mid_gate_table", mid_gate_table, persistent=False)

        self.register_buffer("local_gate_table", local_gate_table, persistent=False)

        self.full_image_decoder = FullImageDecoder(latent_dim=latent_dim)

        # Level 1 (mid) predictor + decoder.
        self.mid_predictor = MidPredictor(
            token_dim=token_dim,
            action_dim=action_dim,
        )

        self.mid_decoder = MidDecoder(
            token_dim=token_dim,
            grid_size=mid_grid_size,
        )

        # Level 0 (local) predictor + decoder.
        self.local_predictor = LocalPredictor(
            token_dim=token_dim,
            action_dim=action_dim,
        )

        self.decoder = SpatialDecoder(
            token_dim=token_dim,
            grid_size=local_grid_size,
        )

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        """The EMA target encoder is never trained, so it always stays in eval
        mode -- targets must not depend on train-mode-only behavior."""

        super().train(mode)

        self.target_encoder.eval()

        return self

    @torch.no_grad()
    def update_target(
        self,
        step: int,
        total_steps: int,
        base_momentum: float = 0.996,
    ):
        """EMA update with a cosine momentum schedule (BYOL-style): momentum
        ramps from `base_momentum` up to 1.0 over training, so the target
        network still moves early on (when it's close to a random encoder
        and would otherwise give a near-useless, slowly-updating target)
        and freezes late (once representations are non-trivial, a very
        stable target helps convergence).

        `step`/`total_steps` should be the current and total optimizer
        steps (not epochs) for a smooth ramp.
        """
        progress = min(max(step / max(total_steps, 1), 0.0), 1.0)

        momentum = 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * progress) + 1) / 2

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
        """Returns (z_global, tokens_mid, tokens_local)."""

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
        """Returns (z_global_target, tokens_mid_target, tokens_local_target)
        from the EMA target encoder.
        """

        return self.target_encoder(
            image,
            mask,
        )

    def _action_gates(self, action):
        """Given the discrete action (0-8 reveal a 3x3-grid cell, STOP_ACTION
        i.e. 9 = STOP), looks up the position gates the predictors use
        instead of broadcasting the action identically to every token:

          - is_stop: (B,) bool, True where the action is STOP.
          - mid_gate: (B, 9) 0/1 weights over the 3x3 mid-token grid.
          - local_gate: (B, 49) 0/1 weights over the 7x7 local-token grid.

        Both come from `build_gate_tables`: the cell's true pixel footprint,
        dilated by `gate_dilation` local tokens (conv receptive field), and
        max-pooled with the same windows the MidPooler averages over. All-zero
        for STOP.
        """
        is_stop = action == self.STOP_ACTION

        safe_action = action.masked_fill(is_stop, 0)

        keep = (~is_stop).unsqueeze(-1).to(self.mid_gate_table.dtype)

        mid_gate = self.mid_gate_table[safe_action] * keep

        local_gate = self.local_gate_table[safe_action] * keep

        return is_stop, mid_gate, local_gate

    @staticmethod
    def opened_mask_from_mask(mask):
        """(B, 1, 28, 28) pixel mask -> (B, 9) float 0/1: which reveal cells
        are open (row-major, same order as the action indices)."""

        rows = mask[:, 0, CELL_CENTERS, :]  # (B, 3, 28)

        cells = rows[:, :, CELL_CENTERS]  # (B, 3, 3)

        return cells.reshape(mask.shape[0], -1)

    def estimate_accuracy(self, z, opened_mask):
        """(B, 10) probabilities: P(correct) after revealing cell 0-8, and
        (index 9) P(correct) if we stop now. See AccuracyHead."""

        return torch.sigmoid(self.accuracy_head(z.detach(), opened_mask))

    def predict(
        self,
        z,
        tokens_mid,
        tokens_local,
        action,
    ):
        """Runs all three levels of prediction for a given action, including
        the top-down FiLM modulation and the position/STOP gating.

        Returns (z_pred, tokens_mid_pred, tokens_local_pred).
        """

        action_embedding = self.action_encoder(action)

        is_stop, mid_gate, local_gate = self._action_gates(action)

        z_pred = self.global_predictor(
            z,
            action_embedding,
            is_stop,
        )

        tokens_mid_ctx = self.global_to_mid_film(tokens_mid, z)

        tokens_mid_pred = self.mid_predictor(
            tokens_mid_ctx,
            tokens_mid,
            action_embedding,
            mid_gate,
            is_stop,
        )

        tokens_local_ctx = self.mid_to_local_film(
            tokens_local, tokens_mid_ctx.mean(dim=1)
        )

        tokens_local_pred = self.local_predictor(
            tokens_local_ctx,
            tokens_local,
            action_embedding,
            local_gate,
            is_stop,
        )

        return z_pred, tokens_mid_pred, tokens_local_pred

    @torch.no_grad()
    def select_action(
        self,
        z,
        tokens_mid,
        tokens_local,
        revealed_mask,
        step,
        budget,
        novelty_threshold=None,
    ):
        """Heuristic active-perception action selection (no extra training
        needed -- reuses the existing predictors): for each not-yet-revealed
        cell, predicts the resulting next-step global latent and uses the
        magnitude of the predicted change, ``||z_pred - z||``, as a proxy for
        how much revealing that cell is expected to update the model's
        belief. Picks the candidate with the largest predicted change.

        Falls back to STOP once `step >= budget`, or (if `novelty_threshold`
        is given) once the best candidate's predicted change is below that
        threshold -- i.e. the model doesn't expect any further reveal to
        teach it much.

        z, tokens_mid, tokens_local: current encoder outputs, batch size B.
        revealed_mask: (B, 9) bool/float, 1 where that cell is already revealed.
        step, budget: current step index and this episode's max steps.
        """
        batch_size = z.shape[0]
        device = z.device

        if step >= budget:
            return torch.full(
                (batch_size,), self.STOP_ACTION, device=device, dtype=torch.long
            )

        best_action = torch.full(
            (batch_size,), self.STOP_ACTION, device=device, dtype=torch.long
        )

        best_score = torch.full((batch_size,), float("-inf"), device=device)

        for candidate in range(NUM_MID_CELLS):
            candidate_action = torch.full(
                (batch_size,), candidate, device=device, dtype=torch.long
            )

            z_pred, _, _ = self.predict(z, tokens_mid, tokens_local, candidate_action)

            score = (z_pred - z).norm(dim=-1)

            already_revealed = revealed_mask[:, candidate].bool()

            score = score.masked_fill(already_revealed, float("-inf"))

            improve = score > best_score

            best_score = torch.where(improve, score, best_score)

            best_action = torch.where(improve, candidate_action, best_action)

        if novelty_threshold is not None:
            stop_worthy = best_score < novelty_threshold

            best_action = torch.where(
                stop_worthy,
                torch.full_like(best_action, self.STOP_ACTION),
                best_action,
            )

        return best_action

    @torch.no_grad()
    def rollout(self, image, mask, actions):
        """Multi-step latent-space rollout: encodes once from a real
        observation, then repeatedly calls `predict()` to imagine forward
        through a sequence of actions with no further access to ground
        truth. Returns the list of (z, tokens_mid, tokens_local) at every
        step, including the initial encode.

        Caution: errors compound at each step (autoregressive drift), since
        step k+1 is predicted from *predicted* (not real) step-k
        representations. This is fine for a handful of steps but degrades
        for long horizons. If long rollouts matter, prefer training with an
        explicit multi-step rollout loss (predict K steps ahead and compare
        each one to its own target-encoder target) over relying on 1-step
        prediction accuracy alone, and/or periodically re-encode from a real
        observation instead of rolling out indefinitely in latent space.
        """
        z, tokens_mid, tokens_local = self.encode(image, mask)

        trajectory = [(z, tokens_mid, tokens_local)]

        for action in actions:
            z, tokens_mid, tokens_local = self.predict(
                z, tokens_mid, tokens_local, action
            )

            trajectory.append((z, tokens_mid, tokens_local))

        return trajectory

    def forward(
        self,
        image,
        mask,
        next_image,
        next_mask,
        full_image,
        action,
    ):
        z, tokens_mid, tokens_local = self.encoder(
            image,
            mask,
        )

        with torch.no_grad():
            z_target, tokens_mid_target, tokens_local_target = self.target_encoder(
                next_image,
                next_mask,
            )

        z_pred, tokens_mid_pred, tokens_local_pred = self.predict(
            z,
            tokens_mid,
            tokens_local,
            action,
        )

        logits_current = self.classifier(z)

        logits_pred = self.classifier(z_pred)

        # Expected-accuracy head: trained (in train.py) against the classifier's
        # correctness on the REAL next observation, hence the extra no-grad
        # encode of next_image. z is detached inside the head input.
        voi_logits = self.accuracy_head(
            z.detach(),
            self.opened_mask_from_mask(mask),
        )

        with torch.no_grad():
            z_next, _, _ = self.encoder(
                next_image,
                next_mask,
            )

            logits_next = self.classifier(z_next)

        # Level 0: predict next partial observation (finest) from local tokens.
        reconstruction = self.decoder(tokens_local_pred)

        # Level 1: predict next partial observation (coarser) from region tokens.
        mid_reconstruction = self.mid_decoder(tokens_mid_pred)

        # Level 2: predict the complete hidden image from the global latent.
        full_reconstruction = self.full_image_decoder(z_pred)

        return {
            "z": z,
            "z_target": z_target,
            "z_pred": z_pred,
            "tokens_mid": tokens_mid,
            "tokens_mid_target": tokens_mid_target,
            "tokens_mid_pred": tokens_mid_pred,
            "tokens_local": tokens_local,
            "tokens_local_target": tokens_local_target,
            "tokens_local_pred": tokens_local_pred,
            "logits_current": logits_current,
            "logits_pred": logits_pred,
            "logits_next": logits_next,
            "voi_logits": voi_logits,
            "reconstruction": reconstruction,
            "mid_reconstruction": mid_reconstruction,
            "full_reconstruction": full_reconstruction,
        }
