import os

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from .environment import ActiveMNISTEnv
from .model import ActiveJEPA
from .rl import QNetwork, ReplayBuffer, select_action, train_dqn_step


def linear_epsilon(
    step,
    start=1.0,
    end=0.05,
    decay_steps=300_000,
):
    progress = min(
        step / decay_steps,
        1.0,
    )

    return start + progress * (end - start)


def linear_stop_exploration(
    step,
    start=0.60,
    end=0.20,
    decay_steps=150_000,
):
    """
    Probability of forcing a random STOP action during epsilon-greedy
    exploration. Starts high so the replay buffer collects plenty of
    early-STOP transitions (otherwise the Q-network never sees enough
    examples to learn that stopping early can be worthwhile), then
    decays so that later training relies more on cells actually explored
    by the (increasingly competent) policy.
    """

    progress = min(
        step / decay_steps,
        1.0,
    )

    return start + progress * (end - start)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # --------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------

    num_episodes = 100_000

    batch_size = 128

    # No discount:
    #
    # return =
    #     - reveal_cost * number_of_reveals
    #     + terminal_reward
    #
    gamma = 1.0

    learning_rate = 1e-4

    target_update_frequency = 1000

    replay_capacity = 100_000

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # --------------------------------------------------
    # JEPA
    # --------------------------------------------------

    jepa = ActiveJEPA(latent_dim=128).to(device)

    checkpoint = torch.load(
        "checkpoints/latest.pt",
        map_location=device,
        weights_only=False,
    )

    jepa.load_state_dict(checkpoint["model"])

    jepa.eval()

    for parameter in jepa.parameters():
        parameter.requires_grad = False

    print("JEPA loaded and frozen.")

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------

    env = ActiveMNISTEnv(
        model=jepa,
        dataset=dataset,
        device=device,
        # Raised from 0.10: with the old cost, opening a cell only needed to
        # improve accuracy by ~5% to be "worth it" (reveal_cost / (correct_reward
        # - wrong_reward) = 0.10 / 2.0), so the agent almost always kept opening
        # cells instead of stopping. 0.20 raises that breakeven to ~10%.
        reveal_cost=0.20,
        correct_reward=1.0,
        wrong_reward=-1.0,
        max_steps=9,
    )

    state = env.reset()

    state_dim = state.shape[0]

    print(f"state_dim: {state_dim}")

    # Expected:
    #
    # 128 z
    # 10 probabilities
    # 1 entropy
    # 1 confidence
    # 1 opened ratio
    # 9 opened mask
    # 18 candidate features
    #
    # = 168

    # --------------------------------------------------
    # DQN
    # --------------------------------------------------

    q_network = QNetwork(
        state_dim=state_dim,
        hidden_dim=256,
        num_actions=10,
    ).to(device)

    target_network = QNetwork(
        state_dim=state_dim,
        hidden_dim=256,
        num_actions=10,
    ).to(device)

    target_network.load_state_dict(q_network.state_dict())

    target_network.eval()

    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    global_step = 0

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    progress = tqdm(
        range(num_episodes),
        desc="RL v2",
    )

    for episode in progress:
        state = env.reset()

        episode_reward = 0.0
        done = False
        info = {}

        while not done:
            epsilon = linear_epsilon(global_step)

            stop_exploration_probability = linear_stop_exploration(global_step)

            action = select_action(
                q_network,
                state,
                epsilon,
                device,
                stop_exploration_probability=stop_exploration_probability,
            )

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            episode_reward += reward

            global_step += 1

            loss = train_dqn_step(
                q_network,
                target_network,
                optimizer,
                replay_buffer,
                batch_size,
                gamma,
                device,
            )

            if global_step % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        opened = info.get(
            "opened",
            0,
        )

        correct = info.get(
            "correct",
            False,
        )

        progress.set_postfix(
            reward=f"{episode_reward:.3f}",
            opened=opened,
            correct=correct,
            epsilon=f"{epsilon:.3f}",
            stop_exp=f"{stop_exploration_probability:.3f}",
            replay=len(replay_buffer),
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        if (episode + 1) % 500 == 0:
            torch.save(
                {
                    "q_network": q_network.state_dict(),
                    "target_network": target_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": episode,
                    "state_dim": state_dim,
                },
                "checkpoints/rl_v2.pt",
            )

            print("\nSaved RL v2 checkpoint.")


if __name__ == "__main__":
    main()
