import torch
import torch.nn.functional as F

from .data import reveal_cells

NUM_CELLS = 9
STOP_ACTION = 9
NUM_ACTIONS = 10


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)


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
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.reveal_cost = reveal_cost
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.max_steps = max_steps

        self.image = None
        self.label = None
        self.opened = []
        self.steps = 0

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

        return self._get_state()

    @torch.no_grad()
    def _get_state(self):
        current_image, current_mask = reveal_cells(
            self.image,
            self.opened,
        )

        image = current_image.unsqueeze(0).to(self.device)
        mask = current_mask.unsqueeze(0).to(self.device)

        # --------------------------------------------------
        # Current state
        #
        # Hierarchical encoder returns (z_global, tokens); the RL state is
        # still built only from z_global so state_dim / layout is unchanged.
        # --------------------------------------------------

        z, tokens = self.model.encode(image, mask)

        current_logits = self.model.classifier(z)

        current_probabilities = F.softmax(
            current_logits,
            dim=-1,
        )

        current_entropy = entropy(current_probabilities)

        current_confidence = current_probabilities.max(dim=-1).values

        opened_ratio = torch.tensor(
            [len(self.opened) / NUM_CELLS],
            device=self.device,
            dtype=z.dtype,
        )

        opened_mask = torch.zeros(
            NUM_CELLS,
            device=self.device,
            dtype=z.dtype,
        )

        for cell in self.opened:
            opened_mask[cell] = 1.0

        # --------------------------------------------------
        # JEPA imagination for every cell
        # --------------------------------------------------

        actions = torch.arange(
            NUM_CELLS,
            device=self.device,
            dtype=torch.long,
        )

        z_repeated = z.expand(
            NUM_CELLS,
            -1,
        )

        tokens_repeated = tokens.expand(
            NUM_CELLS,
            -1,
            -1,
        )

        z_pred, tokens_pred = self.model.predict(
            z_repeated,
            tokens_repeated,
            actions,
        )

        imagined_logits = self.model.classifier(z_pred)

        imagined_probabilities = F.softmax(
            imagined_logits,
            dim=-1,
        )

        imagined_entropy = entropy(imagined_probabilities)

        imagined_confidence = imagined_probabilities.max(dim=-1).values

        # Each cell gets:
        #
        #   imagined entropy
        #   imagined confidence
        #
        # shape = [9, 2]
        candidate_features = torch.stack(
            [
                imagined_entropy,
                imagined_confidence,
            ],
            dim=-1,
        )

        # Opened cells should not provide useful
        # candidate information.
        for cell in self.opened:
            candidate_features[cell] = 0.0

        candidate_features = candidate_features.flatten()

        # --------------------------------------------------
        # Final state
        # --------------------------------------------------

        state = torch.cat(
            [
                z.squeeze(0),
                current_probabilities.squeeze(0),
                current_entropy,
                current_confidence,
                opened_ratio,
                opened_mask,
                candidate_features,
            ],
            dim=0,
        )

        return state.detach()

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

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action == STOP_ACTION:
            state = self._get_state()

            probabilities = state[128 : 128 + 10]

            prediction = probabilities.argmax().item()

            correct = prediction == self.label

            reward = self.correct_reward if correct else self.wrong_reward

            info = {
                "type": "stop",
                "prediction": prediction,
                "label": self.label,
                "correct": correct,
                "opened": len(self.opened),
            }

            return (
                state,
                reward,
                True,
                info,
            )

        # --------------------------------------------------
        # REVEAL
        # --------------------------------------------------

        self.opened.append(action)
        self.steps += 1

        reward = -self.reveal_cost

        # --------------------------------------------------
        # Maximum number of reveals
        # --------------------------------------------------

        if self.steps >= self.max_steps:
            state = self._get_state()

            probabilities = state[128 : 128 + 10]

            prediction = probabilities.argmax().item()

            correct = prediction == self.label

            if correct:
                reward += self.correct_reward
            else:
                reward += self.wrong_reward

            info = {
                "type": "max_steps",
                "prediction": prediction,
                "label": self.label,
                "correct": correct,
                "opened": len(self.opened),
            }

            return (
                state,
                reward,
                True,
                info,
            )

        # --------------------------------------------------
        # Continue
        # --------------------------------------------------

        next_state = self._get_state()

        probabilities = next_state[128 : 128 + 10]

        prediction = probabilities.argmax().item()
        confidence = probabilities.max().item()

        info = {
            "type": "reveal",
            "cell": action,
            "prediction": prediction,
            "confidence": confidence,
            "label": self.label,
            "opened": len(self.opened),
        }

        return (
            next_state,
            reward,
            False,
            info,
        )
