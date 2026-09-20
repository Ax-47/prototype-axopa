import random
from collections import deque

import torch
import torch.nn as nn


NUM_ACTIONS = 10
STOP_ACTION = 9
NUM_CELLS = 9


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=256,
        num_actions=NUM_ACTIONS,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_dim,
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

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                state.detach().cpu(),
                int(action),
                float(reward),
                next_state.detach().cpu(),
                bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)

        actions = torch.tensor(
            actions,
            dtype=torch.long,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):
        return len(self.buffer)


def mask_invalid_actions(
    q_values,
    state,
):
    q_values = q_values.clone()

    # State layout:
    #
    # z              = 128
    # probabilities  = 10
    # entropy        = 1
    # confidence     = 1
    # opened ratio   = 1
    # opened mask    = 9
    # candidate      = 18
    #
    # opened mask starts at 141.

    opened_mask = state[:, 128 + 10 + 1 + 1 + 1 : 128 + 10 + 1 + 1 + 1 + 9]

    q_values[:, :NUM_CELLS] = q_values[:, :NUM_CELLS].masked_fill(
        opened_mask > 0.5,
        float("-inf"),
    )

    return q_values


def select_action(
    q_network,
    state,
    epsilon,
    device,
    stop_exploration_probability=0.30,
):
    # --------------------------------------------------
    # Exploration
    # --------------------------------------------------

    if random.random() < epsilon:
        opened_mask = state[128 + 10 + 1 + 1 + 1 : 128 + 10 + 1 + 1 + 1 + 9]

        valid_cells = [
            cell for cell in range(NUM_CELLS) if opened_mask[cell].item() < 0.5
        ]

        # Explicitly force STOP exploration sometimes.
        if random.random() < stop_exploration_probability:
            return STOP_ACTION

        return random.choice(valid_cells)

    # --------------------------------------------------
    # Exploitation
    # --------------------------------------------------

    state_batch = state.unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network(state_batch)

        q_values = mask_invalid_actions(
            q_values,
            state_batch,
        )

        action = q_values.argmax(dim=-1).item()

    return action


def train_dqn_step(
    q_network,
    target_network,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device,
):
    if len(replay_buffer) < batch_size:
        return None

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # --------------------------------------------------
    # Current Q
    # --------------------------------------------------

    q_values = q_network(states)

    chosen_q = q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    # --------------------------------------------------
    # Target Q
    # --------------------------------------------------

    with torch.no_grad():
        next_q_values = target_network(next_states)

        next_q_values = mask_invalid_actions(
            next_q_values,
            next_states,
        )

        max_next_q = next_q_values.max(dim=1).values

        target = rewards + gamma * (1.0 - dones) * max_next_q

    # --------------------------------------------------
    # Optimize
    # --------------------------------------------------

    loss = nn.functional.smooth_l1_loss(
        chosen_q,
        target,
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        q_network.parameters(),
        1.0,
    )

    optimizer.step()

    return loss.item()
