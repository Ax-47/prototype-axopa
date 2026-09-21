"""
Hybrid planner -- greedy JEPA planner + DQN tie-breaker (interactive viewer).

The decision logic lives in planner.py (shared with eval_rl.py); this file
only plays MNIST test images one after another and draws what the model
believes at every step.

For every step, planner.plan_action:

  1. Builds the same state used to train the DQN (environment.build_state).
  2. Computes an expected value for every possible action (STOP, or reveal
     cell c) from the AccuracyHead's P(correct) -- see planner.py.
  3. If exactly one action has (approximately) the best value, that's the
     move. If several are within TIE_THRESHOLD, the DQN picks among ONLY
     those tied actions.

Needs both checkpoints/latest.pt (JEPA) and checkpoints/rl_v2.pt (DQN).
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

from .data import NUM_CELLS, STOP_ACTION, reveal_cells
from .environment import STATE_DIM, entropy
from .eval_rl import load_dqn, load_jepa
from .planner import TIE_THRESHOLD, plan_action

# How many images to play through automatically.
# Set to None to keep going forever (until the plot window is closed).
NUM_EPISODES = None


def create_visualizer():
    plt.ion()

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    ax_current = axes[0]
    ax_local = axes[1]
    ax_mid = axes[2]
    ax_global = axes[3]
    ax_real = axes[4]

    current_display = ax_current.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    local_display = ax_local.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    mid_display = ax_mid.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    global_display = ax_global.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    real_display = ax_real.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax_current.set_title("Current Observation")

    ax_local.set_title("Local Decoder\n(chosen cell)")

    ax_mid.set_title("Mid Decoder\n(chosen cell)")

    ax_global.set_title("Global Decoder\n(chosen cell)")

    ax_real.set_title("Real Full Image")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    return (
        fig,
        current_display,
        local_display,
        mid_display,
        global_display,
        real_display,
    )


def update_visualizer(
    fig,
    current_display,
    local_display,
    mid_display,
    global_display,
    real_display,
    current_image,
    local_imagined_image,
    mid_imagined_image,
    global_imagined_image,
    real_image,
    title,
):
    current_display.set_data(current_image.squeeze(0).cpu())

    local_display.set_data(local_imagined_image.squeeze(0).cpu())

    mid_display.set_data(mid_imagined_image.squeeze(0).cpu())

    global_display.set_data(global_imagined_image.squeeze(0).cpu())

    real_display.set_data(real_image.squeeze(0).cpu())

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    plt.pause(3)


@torch.no_grad()
def imagine(model, info, action, device):
    """What the three decoders expect to see after `action` (STOP is an exact
    identity map in the predictors, so this also works for STOP). Only the
    chosen action is imagined -- the planner no longer needs the other eight
    (it reads the AccuracyHead instead)."""

    action_tensor = torch.tensor(
        [action],
        device=device,
        dtype=torch.long,
    )

    z_pred, tokens_mid_pred, tokens_local_pred = model.predict(
        info["z"],
        info["tokens_mid"],
        info["tokens_local"],
        action_tensor,
    )

    local_image = model.decoder(tokens_local_pred).squeeze(0)

    mid_image = model.mid_decoder(tokens_mid_pred).squeeze(0)

    global_image = model.full_image_decoder(z_pred).squeeze(0)

    imagined_probabilities = F.softmax(
        model.classifier(z_pred),
        dim=-1,
    ).squeeze(0)

    return (
        local_image,
        mid_image,
        global_image,
        imagined_probabilities,
    )


@torch.no_grad()
def run_episode(
    model,
    q_network,
    dataset,
    device,
    fig,
    current_display,
    local_display,
    mid_display,
    global_display,
    real_display,
    episode_index,
):
    index = torch.randint(
        0,
        len(dataset),
        (1,),
    ).item()

    image, label = dataset[index]

    opened = []

    # At most NUM_CELLS reveals, then the planner has only STOP left, so the
    # loop always ends with a STOP inside it.
    for step in range(NUM_CELLS + 1):
        plan = plan_action(
            model,
            q_network,
            image,
            opened,
            device,
        )

        action = plan["action"]

        info = plan["info"]

        probabilities = info["probabilities"]

        prediction = probabilities.argmax().item()

        confidence = probabilities.max().item()

        current_entropy = entropy(probabilities).item()

        print()
        print(f"[Image {episode_index}] Step {step + 1}")

        print(f"Opened: {opened}")

        print(f"Current prediction: {prediction}")

        print(f"Confidence: {confidence:.4f}")

        print(f"Entropy: {current_entropy:.4f}")
        print()
        print("Candidate values:")
        for candidate_action, value in sorted(
            plan["values"].items(),
            key=lambda item: -item[1],
        ):
            label_text = (
                "STOP"
                if candidate_action == STOP_ACTION
                else f"cell {candidate_action}"
            )

            tie_marker = " (tied)" if candidate_action in plan["tied_actions"] else ""

            chosen_marker = " <-- chosen" if candidate_action == action else ""

            print(
                f"  {label_text}: value={value:+.4f} "
                f"P(correct)={plan['accuracy'][candidate_action]:.3f}"
                f"{tie_marker}{chosen_marker}"
            )

        if plan["tie_broken_by_dqn"]:
            print(
                f"  -> tie among {plan['tied_actions']} within {TIE_THRESHOLD}, "
                f"DQN broke the tie -> chose {action}"
            )

        (
            local_image,
            mid_image,
            global_image,
            imagined_probabilities,
        ) = imagine(
            model,
            info,
            action,
            device,
        )

        current_image, _ = reveal_cells(
            image,
            opened,
        )

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action == STOP_ACTION:
            print()
            print(">>> STOP")

            print(f"Prediction: {prediction}")

            print(f"Real label: {label}")

            print(f"Correct: {prediction == label}")

            print(f"Opened {len(opened)}/9 cells")

            update_visualizer(
                fig,
                current_display,
                local_display,
                mid_display,
                global_display,
                real_display,
                current_image,
                local_image,
                mid_image,
                global_image,
                image,
                (
                    f"[Image {episode_index}] STOP | "
                    f"prediction={prediction} | "
                    f"real={label} | "
                    f"opened={len(opened)}/9"
                ),
            )

            return

        # --------------------------------------------------
        # Reveal the chosen cell
        # --------------------------------------------------

        print(f">>> Reveal cell {action}")

        imagined_prediction = imagined_probabilities.argmax().item()

        update_visualizer(
            fig,
            current_display,
            local_display,
            mid_display,
            global_display,
            real_display,
            current_image,
            local_image,
            mid_image,
            global_image,
            image,
            (
                f"[Image {episode_index}] Step {step + 1} | "
                f"action=cell {action} | "
                f"JEPA predicts {imagined_prediction}"
            ),
        )

        opened.append(action)

        current_image, _ = reveal_cells(
            image,
            opened,
        )

        update_visualizer(
            fig,
            current_display,
            local_display,
            mid_display,
            global_display,
            real_display,
            current_image,
            local_image,
            mid_image,
            global_image,
            image,
            (
                f"[Image {episode_index}] Revealed cell {action} | "
                f"opened={len(opened)}/9"
            ),
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # --------------------------------------------------
    # JEPA (frozen, eval mode)
    # --------------------------------------------------

    model = load_jepa(
        "checkpoints/latest.pt",
        device,
    )

    # --------------------------------------------------
    # DQN (used only as a tie-breaker)
    # --------------------------------------------------

    q_network = load_dqn(
        "checkpoints/rl_v2.pt",
        device,
    )

    assert q_network.net[0].in_features == STATE_DIM, (
        "DQN checkpoint was trained with a different state layout; "
        "retrain it with train_rl.py"
    )

    print("JEPA + DQN (tie-breaker) loaded.")

    dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    (
        fig,
        current_display,
        local_display,
        mid_display,
        global_display,
        real_display,
    ) = create_visualizer()

    window_closed = {"value": False}

    def _on_close(event):
        window_closed["value"] = True

    fig.canvas.mpl_connect("close_event", _on_close)

    episode_index = 0

    while (NUM_EPISODES is None or episode_index < NUM_EPISODES) and not window_closed[
        "value"
    ]:
        episode_index += 1

        run_episode(
            model,
            q_network,
            dataset,
            device,
            fig,
            current_display,
            local_display,
            mid_display,
            global_display,
            real_display,
            episode_index,
        )

    print()

    if window_closed["value"]:
        print("Plot window closed — stopping.")
    else:
        print(
            f"Finished {episode_index} image(s). Close the matplotlib window to exit."
        )

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
