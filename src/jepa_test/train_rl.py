import os
from collections import deque

import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from .data import split_indices
from .environment import STATE_DIM, ActiveMNISTEnv
from .eval_rl import dqn_policy, load_jepa, run_policy
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


def main(
    num_episodes=100_000,
    batch_size=128,
    replay_capacity=100_000,
    learning_starts=5_000,
    train_every=4,
    target_update_frequency=1000,
    epsilon_decay_steps=300_000,
    eval_every=5_000,
    eval_episodes=500,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # --------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------

    # No discount:
    #
    # return =
    #     - reveal_cost * number_of_reveals
    #     + terminal_reward
    #
    # (gamma must stay 1.0 if info_gain_weight > 0, see environment.py)
    gamma = 1.0

    learning_rate = 1e-4

    # --------------------------------------------------
    # Dataset
    #
    # The JEPA (and so its classifier) is over-confident on the images it was
    # trained on. The DQN learns *when to stop* from that confidence, so it
    # trains on a DIFFERENT slice of MNIST-train (rl_train), and model
    # selection uses a third (rl_val). The real test set is only used by
    # eval_rl.py / play.py.
    # --------------------------------------------------

    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    _, rl_train_indices, rl_val_indices = split_indices()

    train_set = Subset(dataset, rl_train_indices)

    val_set = Subset(dataset, rl_val_indices)

    # --------------------------------------------------
    # JEPA
    # --------------------------------------------------

    jepa = load_jepa(
        "checkpoints/latest.pt",
        device,
    )

    print("JEPA loaded and frozen.")

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------

    env_kwargs = dict(
        model=jepa,
        device=device,
        # Breakeven for "is one more reveal worth it?" =
        #     reveal_cost / (correct_reward - wrong_reward)
        #   = 0.20 / 2.0 = 10% more expected accuracy.
        # This is now the WHOLE story: the old info_gain bonus is off
        # (info_gain_weight=0.0), because it subsidised every reveal and
        # silently made the effective cost far smaller than 0.20.
        reveal_cost=0.20,
        correct_reward=1.0,
        wrong_reward=-1.0,
        # No artificial cap: the agent may use the full grid if it needs to.
        max_steps=9,
        info_gain_weight=0.0,
    )

    env = ActiveMNISTEnv(dataset=train_set, **env_kwargs)

    val_env = ActiveMNISTEnv(dataset=val_set, **env_kwargs)

    state = env.reset()

    state_dim = state.shape[0]

    assert state_dim == STATE_DIM, f"state_dim {state_dim} != STATE_DIM {STATE_DIM}"

    print(f"state_dim: {state_dim}")

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

    replay_buffer = ReplayBuffer(
        capacity=replay_capacity,
        state_dim=state_dim,
        device=device,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    global_step = 0

    best_val_return = float("-inf")

    recent_reward = deque(maxlen=200)
    recent_correct = deque(maxlen=200)
    recent_opened = deque(maxlen=200)

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    def save(path, episode):
        torch.save(
            {
                "q_network": q_network.state_dict(),
                "target_network": target_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "state_dim": state_dim,
            },
            path,
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
        epsilon = linear_epsilon(
            global_step,
            decay_steps=epsilon_decay_steps,
        )

        while not done:
            epsilon = linear_epsilon(
                global_step,
                decay_steps=epsilon_decay_steps,
            )

            action = select_action(
                q_network,
                state,
                epsilon,
                device,
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

            # One gradient step per `train_every` environment steps (each env
            # step costs a JEPA forward pass, so 1:1 wasted most of the time).
            if global_step >= learning_starts and global_step % train_every == 0:
                train_dqn_step(
                    q_network,
                    target_network,
                    optimizer,
                    replay_buffer,
                    batch_size,
                    gamma,
                )

            if global_step % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        recent_reward.append(episode_reward)
        recent_correct.append(float(info.get("correct", False)))
        recent_opened.append(info.get("opened", 0))

        progress.set_postfix(
            reward=f"{sum(recent_reward) / len(recent_reward):.3f}",
            acc=f"{sum(recent_correct) / len(recent_correct):.3f}",
            opened=f"{sum(recent_opened) / len(recent_opened):.2f}",
            epsilon=f"{epsilon:.3f}",
            replay=len(replay_buffer),
        )

        # --------------------------------------------------
        # Greedy evaluation on held-out images (epsilon = 0)
        # --------------------------------------------------

        if (episode + 1) % eval_every == 0:
            metrics = run_policy(
                val_env,
                dqn_policy(q_network, device),
                list(range(min(eval_episodes, len(val_set)))),
            )

            tqdm.write(
                f"[eval @ episode {episode + 1}] "
                f"acc={metrics['accuracy']:.3f} "
                f"cells={metrics['cells']:.2f} "
                f"return={metrics['return']:+.3f}"
            )

            if metrics["return"] > best_val_return:
                best_val_return = metrics["return"]

                save("checkpoints/rl_v2_best.pt", episode)

                tqdm.write("  new best -> checkpoints/rl_v2_best.pt")

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        if (episode + 1) % 500 == 0:
            save("checkpoints/rl_v2.pt", episode)

            tqdm.write("Saved RL v2 checkpoint.")


if __name__ == "__main__":
    main()
