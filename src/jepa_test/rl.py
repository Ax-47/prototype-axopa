import random

import torch
import torch.nn as nn

from .data import NUM_ACTIONS, NUM_CELLS, STOP_ACTION
from .environment import OPENED_MASK_SLICE


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
    """Ring buffer of preallocated tensors.

    Replaces the old `deque` of tuples: `random.sample` on a deque indexes
    into the middle (O(n)), and every sample re-stacked 128 small CPU tensors
    and copied them to the GPU. Here storage lives on `device` (default CPU),
    pushes are in-place row writes, and sampling is one `randint` + indexing.
    """

    def __init__(
        self,
        capacity,
        state_dim,
        device="cpu",
    ):
        self.capacity = capacity
        self.device = torch.device(device)

        self.states = torch.zeros(capacity, state_dim, device=self.device)
        self.next_states = torch.zeros(capacity, state_dim, device=self.device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(capacity, device=self.device)
        self.dones = torch.zeros(capacity, device=self.device)

        self.position = 0
        self.size = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        i = self.position

        self.states[i] = state.detach().to(self.device)
        self.next_states[i] = next_state.detach().to(self.device)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(bool(done))

        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        index = torch.randint(
            0,
            self.size,
            (batch_size,),
            device=self.device,
        )

        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
        )

    def __len__(self):
        return self.size


def mask_invalid_actions(
    q_values,
    state,
):
    """-inf on reveal actions whose cell is already open. STOP is never masked."""

    q_values = q_values.clone()

    opened_mask = state[:, OPENED_MASK_SLICE]

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
):
    # --------------------------------------------------
    # Exploration
    #
    # Uniform over ALL valid actions, STOP included. With k unopened cells
    # STOP is picked with probability 1/(k+1), so exploratory episodes end at
    # every depth 0..9 with equal probability. (The old "force STOP with
    # probability 0.6*epsilon" made P(reach depth j) ~ 0.4^j: the replay
    # buffer had almost no states with >4 cells open early in training.)
    # --------------------------------------------------

    if random.random() < epsilon:
        opened_mask = state[OPENED_MASK_SLICE].tolist()

        valid_actions = [cell for cell in range(NUM_CELLS) if opened_mask[cell] < 0.5]

        valid_actions.append(STOP_ACTION)

        return random.choice(valid_actions)

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
    device=None,  # kept for call compatibility; the buffer already lives on its device
    double_dqn=True,
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
    #
    # Double DQN: the online network picks the best VALID next action, the
    # target network evaluates it (less over-estimation than max over the
    # target network's own noisy Q-values).
    # --------------------------------------------------

    with torch.no_grad():
        if double_dqn:
            online_next_q = mask_invalid_actions(
                q_network(next_states),
                next_states,
            )

            best_next_action = online_next_q.argmax(
                dim=1,
                keepdim=True,
            )

            next_q = (
                target_network(next_states)
                .gather(
                    1,
                    best_next_action,
                )
                .squeeze(1)
            )

        else:
            next_q = (
                mask_invalid_actions(
                    target_network(next_states),
                    next_states,
                )
                .max(dim=1)
                .values
            )

        target = rewards + gamma * (1.0 - dones) * next_q

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
