import torch
import torch.nn.functional as F

from .data import (
    NUM_ACTIONS,
    NUM_CELLS,
    STOP_ACTION,  # noqa: F401
    reveal_cells,
)

NUM_CLASSES = 10
LATENT_DIM = 128

# --------------------------------------------------------------------------
# State layout -- the ONE place that knows it. rl.py (action masking),
# planner.py, play.py and eval_rl.py all import these instead of repeating
# magic offsets like `128 + 10 + 1 + 1 + 1`.
#
#   z                       LATENT_DIM   (128)
#   class probabilities     10
#   entropy                 1
#   confidence              1
#   opened ratio            1
#   opened mask             9
#   candidate features      18   per cell: [P(correct) after reveal,
#                                          that minus P(correct) now]
#                                          (from the AccuracyHead; 0 if open)
#                           ----
#                           168
# --------------------------------------------------------------------------

Z_SLICE = slice(0, LATENT_DIM)
PROB_SLICE = slice(LATENT_DIM, LATENT_DIM + NUM_CLASSES)
ENTROPY_IDX = PROB_SLICE.stop
CONFIDENCE_IDX = ENTROPY_IDX + 1
OPENED_RATIO_IDX = CONFIDENCE_IDX + 1
OPENED_MASK_SLICE = slice(OPENED_RATIO_IDX + 1, OPENED_RATIO_IDX + 1 + NUM_CELLS)
CANDIDATE_SLICE = slice(OPENED_MASK_SLICE.stop, OPENED_MASK_SLICE.stop + 2 * NUM_CELLS)
STATE_DIM = CANDIDATE_SLICE.stop  # 168


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)


@torch.no_grad()
def build_state(
    model,
    image,
    opened,
    device,
):
    """Builds the STATE_DIM-dim DQN state for `image` with the cells in
    `opened` revealed.

    Returns (state, info). `info` carries what other callers need without
    recomputing it (planner / play / eval):

        z, tokens_mid, tokens_local   encoder outputs, batch size 1
        probabilities                 (10,) classifier probabilities
        acc_now                       scalar tensor, P(correct) if we STOP now
        acc_after                     (9,) P(correct) after revealing each cell
    """

    current_image, current_mask = reveal_cells(
        image,
        opened,
    )

    image_batch = current_image.unsqueeze(0).to(device)

    mask_batch = current_mask.unsqueeze(0).to(device)

    z, tokens_mid, tokens_local = model.encode(
        image_batch,
        mask_batch,
    )

    probabilities = F.softmax(
        model.classifier(z),
        dim=-1,
    )

    current_entropy = entropy(probabilities)

    current_confidence = probabilities.max(dim=-1).values

    opened_ratio = torch.tensor(
        [len(opened) / NUM_CELLS],
        device=device,
        dtype=z.dtype,
    )

    opened_mask = torch.zeros(
        1,
        NUM_CELLS,
        device=device,
        dtype=z.dtype,
    )

    if opened:
        opened_mask[0, list(opened)] = 1.0

    # --------------------------------------------------
    # Expected accuracy per action (AccuracyHead)
    #
    # Replaces the old "imagined entropy / confidence" (classifier applied to
    # z_pred). That was confidence of an *averaged* latent, not the expected
    # value of information. This is trained against the real next observation.
    # --------------------------------------------------

    accuracy = model.estimate_accuracy(
        z,
        opened_mask,
    ).squeeze(0)  # (10,)

    acc_after = accuracy[:NUM_CELLS]

    acc_now = accuracy[STOP_ACTION]

    candidate_features = torch.stack(
        [
            acc_after,
            acc_after - acc_now,
        ],
        dim=-1,
    )  # (9, 2)

    # Opened cells should not provide useful candidate information.
    candidate_features = candidate_features * (1.0 - opened_mask.squeeze(0)).unsqueeze(
        -1
    )

    state = torch.cat(
        [
            z.squeeze(0),
            probabilities.squeeze(0),
            current_entropy,
            current_confidence,
            opened_ratio,
            opened_mask.squeeze(0),
            candidate_features.flatten(),
        ],
        dim=0,
    )

    info = {
        "z": z,
        "tokens_mid": tokens_mid,
        "tokens_local": tokens_local,
        "probabilities": probabilities.squeeze(0),
        "acc_now": acc_now,
        "acc_after": acc_after,
    }

    return state.detach(), info


class ActiveMNISTEnv:
    def __init__(
        self,
        model,
        dataset,
        device,
        reveal_cost=0.10,
        correct_reward=1.0,
        wrong_reward=-1.0,
        max_steps=9,
        info_gain_weight=0.0,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.reveal_cost = reveal_cost
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.max_steps = max_steps

        # Optional reward shaping, OFF by default.
        #
        # It used to be `w * max(H_before - H_after, 0)` on reveals only. That
        # (a) ignores entropy increases, (b) gives STOP nothing, so every
        # reveal was subsidised relative to STOP, and (c) silently cancelled
        # the reveal_cost breakeven analysis (0.5 * 0.4 nats == 0.20 == the
        # whole cost of a reveal), all while rewarding "confidently wrong".
        #
        # With w > 0 this is now POTENTIAL-BASED shaping with
        # Phi(s) = -w * H(s), Phi(terminal) = 0 and gamma = 1:
        #
        #   reveal -> non-terminal s':   + w * (H(s) - H(s'))
        #   any terminal transition:     + w * H(s)     (s = state acted from)
        #
        # Over a whole episode the bonuses telescope to exactly w * H(s_0), a
        # constant, so shaping can speed up credit assignment but cannot
        # change which policy is optimal. train_gamma must stay 1.0.
        self.info_gain_weight = info_gain_weight

        self.image = None
        self.label = None
        self.opened = []
        self.steps = 0

        self._state = None

    def reset(self, index=None):
        if index is None:
            index = torch.randint(
                0,
                len(self.dataset),
                (1,),
            ).item()

        image, label = self.dataset[index]

        self.image = image
        self.label = int(label)
        self.opened = []
        self.steps = 0

        self._state = self._get_state()

        return self._state

    @torch.no_grad()
    def _get_state(self):
        state, _ = build_state(
            self.model,
            self.image,
            self.opened,
            self.device,
        )

        return state

    def valid_actions(self):
        actions = [cell for cell in range(NUM_CELLS) if cell not in self.opened]

        # STOP is always available.
        actions.append(STOP_ACTION)

        return actions

    @torch.no_grad()
    def step(self, action):
        action = int(action)

        if action not in self.valid_actions():
            raise ValueError(f"Invalid action: {action}")

        # The state the agent acted from was already computed (reset / previous
        # step). Reusing it halves the JEPA forward passes per step.
        state_before = self._state

        entropy_before = state_before[ENTROPY_IDX].item()

        shaping = self.info_gain_weight

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action == STOP_ACTION:
            prediction = state_before[PROB_SLICE].argmax().item()

            correct = prediction == self.label

            reward = self.correct_reward if correct else self.wrong_reward

            reward += shaping * entropy_before

            info = {
                "type": "stop",
                "prediction": prediction,
                "label": self.label,
                "correct": correct,
                "opened": len(self.opened),
            }

            return (
                state_before,
                reward,
                True,
                info,
            )

        # --------------------------------------------------
        # REVEAL
        # --------------------------------------------------

        self.opened.append(action)
        self.steps += 1

        next_state = self._get_state()

        self._state = next_state

        reward = -self.reveal_cost

        entropy_after = next_state[ENTROPY_IDX].item()

        probabilities = next_state[PROB_SLICE]

        prediction = probabilities.argmax().item()

        # --------------------------------------------------
        # Maximum number of reveals (or nothing left to reveal): the episode
        # ends and the current belief is graded.
        # --------------------------------------------------

        if self.steps >= min(self.max_steps, NUM_CELLS):
            correct = prediction == self.label

            reward += self.correct_reward if correct else self.wrong_reward

            reward += shaping * entropy_before

            info = {
                "type": "max_steps",
                "prediction": prediction,
                "label": self.label,
                "correct": correct,
                "opened": len(self.opened),
                "info_gain": entropy_before - entropy_after,
            }

            return (
                next_state,
                reward,
                True,
                info,
            )

        # --------------------------------------------------
        # Continue
        # --------------------------------------------------

        reward += shaping * (entropy_before - entropy_after)

        info = {
            "type": "reveal",
            "cell": action,
            "prediction": prediction,
            "confidence": probabilities.max().item(),
            "label": self.label,
            "opened": len(self.opened),
            "info_gain": entropy_before - entropy_after,
        }

        return (
            next_state,
            reward,
            False,
            info,
        )
