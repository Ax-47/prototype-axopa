"""
Evaluation + baselines for the active-perception agent.

    python -m active_jepa.eval_rl                # uses checkpoints/latest.pt + rl_v2.pt
    python -m active_jepa.eval_rl --episodes 2000

For the same MNIST test images it reports accuracy, average number of cells
revealed and average return (with the environment's economics:
-reveal_cost per reveal, +1 correct / -1 wrong) for

  * dqn        the trained Q-network, greedy (epsilon = 0)
  * hybrid     greedy expected-value planner + DQN tie-break (what play.py runs)
  * greedy     the planner alone (AccuracyHead only, no DQN)
  * fixed@t    reveal cells in a fixed order (centre first) until the
               classifier's confidence >= t   (t in THRESHOLDS)
  * random@t   same, but in a random order

If the DQN cannot clearly beat the best fixed@t / random@t row, it has not
learned "where to look" -- only "when to stop", which a threshold already does.

It also prints a calibration report for the AccuracyHead and for the raw
softmax confidence, so you can see whether the features fed to the DQN
actually predict the value of revealing a cell.
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from .data import NUM_CELLS, reveal_cells
from .environment import ActiveMNISTEnv, build_state
from .model import ActiveJEPA
from .planner import CORRECT_REWARD, REVEAL_COST, WRONG_REWARD, plan_action
from .rl import QNetwork, select_action

# Centre first, then edges, then corners.
FIXED_ORDER = [4, 1, 3, 5, 7, 0, 2, 6, 8]

THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]


# --------------------------------------------------
# Policies (closed-loop, run through the real environment)
# --------------------------------------------------


def dqn_policy(q_network, device):
    def policy(env, state):
        return select_action(
            q_network,
            state,
            0.0,
            device,
        )

    return policy


def planner_policy(q_network, device):
    """q_network=None -> purely greedy planner."""

    def policy(env, state):
        return plan_action(
            env.model,
            q_network,
            env.image,
            env.opened,
            device,
            reveal_cost=env.reveal_cost,
            correct_reward=env.correct_reward,
            wrong_reward=env.wrong_reward,
        )["action"]

    return policy


def run_policy(env, policy, indices):
    """Plays one episode per dataset index. Returns accuracy / cells / return."""

    correct = 0
    opened_total = 0
    return_total = 0.0

    for index in indices:
        state = env.reset(index)

        done = False

        episode_return = 0.0

        while not done:
            action = policy(env, state)

            state, reward, done, info = env.step(action)

            episode_return += reward

        correct += int(info["correct"])
        opened_total += info["opened"]
        return_total += episode_return

    count = len(indices)

    return {
        "accuracy": correct / count,
        "cells": opened_total / count,
        "return": return_total / count,
    }


# --------------------------------------------------
# Threshold baselines (open-loop order, closed-loop stop rule)
# --------------------------------------------------


@torch.no_grad()
def prefix_probabilities(model, image, order, device):
    """Class probabilities after revealing 0, 1, ..., 9 cells of `order`
    (10 rows), computed in one batched encode."""

    images = []
    masks = []

    for k in range(NUM_CELLS + 1):
        current_image, current_mask = reveal_cells(
            image,
            order[:k],
        )

        images.append(current_image)
        masks.append(current_mask)

    z, _, _ = model.encode(
        torch.stack(images).to(device),
        torch.stack(masks).to(device),
    )

    return F.softmax(
        model.classifier(z),
        dim=-1,
    )


def threshold_baselines(
    model,
    dataset,
    indices,
    device,
    rng,
    reveal_cost=REVEAL_COST,
    correct_reward=CORRECT_REWARD,
    wrong_reward=WRONG_REWARD,
    thresholds=THRESHOLDS,
):
    orders = ("fixed", "random")

    totals = {(name, t): [0, 0, 0.0] for name in orders for t in thresholds}

    for index in indices:
        image, label = dataset[index]

        label = int(label)

        for name in orders:
            if name == "fixed":
                order = FIXED_ORDER
            else:
                order = rng.sample(range(NUM_CELLS), NUM_CELLS)

            confidence, prediction = prefix_probabilities(
                model,
                image,
                order,
                device,
            ).max(dim=-1)

            confidence = confidence.tolist()
            prediction = prediction.tolist()

            for t in thresholds:
                # First prefix that is confident enough; otherwise all 9 cells.
                k = next(
                    (i for i, c in enumerate(confidence) if c >= t),
                    NUM_CELLS,
                )

                ok = prediction[k] == label

                entry = totals[(name, t)]

                entry[0] += int(ok)
                entry[1] += k
                entry[2] += -reveal_cost * k + (correct_reward if ok else wrong_reward)

    count = len(indices)

    return {
        f"{name}@{t:.2f}": {
            "accuracy": entry[0] / count,
            "cells": entry[1] / count,
            "return": entry[2] / count,
        }
        for (name, t), entry in totals.items()
    }


# --------------------------------------------------
# Diagnostics: is the AccuracyHead informative and calibrated?
# --------------------------------------------------


@torch.no_grad()
def head_calibration(
    model,
    dataset,
    indices,
    device,
    rng,
    samples=1500,
):
    """Samples random partial observations (0-8 cells open) and one random
    unopened cell, then compares what the head predicted with what really
    happened after revealing that cell."""

    pred_after, true_after = [], []
    pred_now, true_now, confidence_now = [], [], []

    for _ in range(samples):
        image, label = dataset[rng.choice(indices)]

        label = int(label)

        opened = rng.sample(range(NUM_CELLS), rng.randint(0, NUM_CELLS - 1))

        cell = rng.choice([c for c in range(NUM_CELLS) if c not in opened])

        _, info = build_state(model, image, opened, device)

        _, info_next = build_state(model, image, opened + [cell], device)

        pred_after.append(info["acc_after"][cell].item())
        true_after.append(float(info_next["probabilities"].argmax().item() == label))

        pred_now.append(info["acc_now"].item())
        true_now.append(float(info["probabilities"].argmax().item() == label))
        confidence_now.append(info["probabilities"].max().item())

    def brier(pred, true):
        return sum((p - y) ** 2 for p, y in zip(pred, true)) / len(true)

    def mean(values):
        return sum(values) / len(values)

    def skill(pred, true):
        # 1 - Brier / Brier(constant mean). > 0 means the head explains
        # variance the base rate does not.
        base = mean(true)

        reference = brier([base] * len(true), true)

        return 1.0 - brier(pred, true) / reference if reference > 0 else float("nan")

    print()
    print("Calibration (random partial observations, 0-8 cells open):")
    print(
        f"  P(correct) now,   head:    mean pred={mean(pred_now):.3f}  "
        f"actual={mean(true_now):.3f}  skill={skill(pred_now, true_now):+.3f}"
    )
    print(
        f"  P(correct) now,   softmax: mean conf={mean(confidence_now):.3f}  "
        f"actual={mean(true_now):.3f}  skill={skill(confidence_now, true_now):+.3f}"
    )
    print(
        f"  P(correct) after, head:    mean pred={mean(pred_after):.3f}  "
        f"actual={mean(true_after):.3f}  skill={skill(pred_after, true_after):+.3f}"
    )


# --------------------------------------------------
# Reporting
# --------------------------------------------------


def print_table(results):
    print()
    print(f"{'policy':<22}{'accuracy':>10}{'cells':>9}{'return':>10}")
    print("-" * 51)

    for name, m in results.items():
        print(
            f"{name:<22}{m['accuracy']:>10.4f}{m['cells']:>9.2f}{m['return']:>+10.4f}"
        )


def load_jepa(path, device):
    model = ActiveJEPA(latent_dim=128).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def load_dqn(path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    q_network = QNetwork(
        state_dim=checkpoint["state_dim"],
        hidden_dim=256,
        num_actions=10,
    ).to(device)

    q_network.load_state_dict(checkpoint["q_network"])

    q_network.eval()

    return q_network


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jepa-checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--rl-checkpoint", default="checkpoints/rl_v2.pt")
    parser.add_argument("--no-diagnostics", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    model = load_jepa(args.jepa_checkpoint, device)

    dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    indices = rng.sample(range(len(dataset)), args.episodes)

    env = ActiveMNISTEnv(
        model=model,
        dataset=dataset,
        device=device,
        reveal_cost=REVEAL_COST,
        correct_reward=CORRECT_REWARD,
        wrong_reward=WRONG_REWARD,
        max_steps=9,
        info_gain_weight=0.0,
    )

    results = {}

    if os.path.exists(args.rl_checkpoint):
        q_network = load_dqn(args.rl_checkpoint, device)

        results["dqn"] = run_policy(env, dqn_policy(q_network, device), indices)

        results["hybrid (greedy+dqn)"] = run_policy(
            env,
            planner_policy(q_network, device),
            indices,
        )

    else:
        print(f"(no DQN checkpoint at {args.rl_checkpoint}; skipping dqn / hybrid)")

    results["greedy (head only)"] = run_policy(
        env,
        planner_policy(None, device),
        indices,
    )

    results.update(
        threshold_baselines(
            model,
            dataset,
            indices,
            device,
            rng,
        )
    )

    print_table(results)

    if not args.no_diagnostics:
        head_calibration(
            model,
            dataset,
            list(range(len(dataset))),
            device,
            rng,
        )


if __name__ == "__main__":
    main()
