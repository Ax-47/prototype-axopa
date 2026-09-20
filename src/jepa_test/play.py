import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

from .data import reveal_cells
from .model import ActiveJEPA
from .rl import QNetwork, mask_invalid_actions

NUM_CELLS = 9
STOP_ACTION = 9

# How many images to play through automatically.
# Set to None to keep going forever (until the plot window is closed).
NUM_EPISODES = None


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)


@torch.no_grad()
def imagine_full_image(
    model,
    image,
    opened,
    action,
    device,
):
    current_image, current_mask = reveal_cells(
        image,
        opened,
    )

    image_batch = current_image.unsqueeze(0).to(device)

    mask_batch = current_mask.unsqueeze(0).to(device)

    z = model.encode(
        image_batch,
        mask_batch,
    )

    action_tensor = torch.tensor(
        [action],
        device=device,
        dtype=torch.long,
    )

    z_pred = model.predict(
        z,
        action_tensor,
    )

    imagined_image = model.full_image_decoder(z_pred)

    imagined_logits = model.classifier(z_pred)

    imagined_probabilities = F.softmax(
        imagined_logits,
        dim=-1,
    )

    return (
        imagined_image.squeeze(0),
        imagined_probabilities.squeeze(0),
    )


@torch.no_grad()
def build_state(
    model,
    image,
    opened,
    device,
):
    current_image, current_mask = reveal_cells(
        image,
        opened,
    )

    image_batch = current_image.unsqueeze(0).to(device)

    mask_batch = current_mask.unsqueeze(0).to(device)

    z = model.encode(
        image_batch,
        mask_batch,
    )

    logits = model.classifier(z)

    probabilities = F.softmax(
        logits,
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
        NUM_CELLS,
        device=device,
        dtype=z.dtype,
    )

    for cell in opened:
        opened_mask[cell] = 1.0

    # --------------------------------------------------
    # Imagine every possible action
    # --------------------------------------------------

    actions = torch.arange(
        NUM_CELLS,
        device=device,
        dtype=torch.long,
    )

    z_repeated = z.expand(
        NUM_CELLS,
        -1,
    )

    z_pred = model.predict(
        z_repeated,
        actions,
    )

    imagined_logits = model.classifier(z_pred)

    imagined_probabilities = F.softmax(
        imagined_logits,
        dim=-1,
    )

    imagined_entropy = entropy(imagined_probabilities)

    imagined_confidence = imagined_probabilities.max(dim=-1).values

    candidate_features = torch.stack(
        [
            imagined_entropy,
            imagined_confidence,
        ],
        dim=-1,
    )

    for cell in opened:
        candidate_features[cell] = 0.0

    candidate_features = candidate_features.flatten()

    state = torch.cat(
        [
            z.squeeze(0),
            probabilities.squeeze(0),
            current_entropy,
            current_confidence,
            opened_ratio,
            opened_mask,
            candidate_features,
        ],
        dim=0,
    )

    return (
        state,
        probabilities.squeeze(0),
        imagined_probabilities,
        imagined_entropy,
        imagined_confidence,
    )


def create_visualizer():
    plt.ion()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    ax_current = axes[0]
    ax_imagined = axes[1]
    ax_real = axes[2]

    current_display = ax_current.imshow(
        torch.zeros(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    imagined_display = ax_imagined.imshow(
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

    ax_imagined.set_title("JEPA Imagination")

    ax_real.set_title("Real Full Image")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    return (
        fig,
        ax_current,
        ax_imagined,
        ax_real,
        current_display,
        imagined_display,
        real_display,
    )


def update_visualizer(
    fig,
    current_display,
    imagined_display,
    real_display,
    current_image,
    imagined_image,
    real_image,
    title,
):
    current_display.set_data(current_image.squeeze(0).cpu())

    imagined_display.set_data(imagined_image.squeeze(0).cpu())

    real_display.set_data(real_image.squeeze(0).cpu())

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Small pause so matplotlib actually renders.
    plt.pause(3)


def run_episode(
    model,
    q_network,
    dataset,
    device,
    fig,
    current_display,
    imagined_display,
    real_display,
    episode_index,
):
    """Play one image end-to-end (STOP action or step cap), then return."""

    index = torch.randint(
        0,
        len(dataset),
        (1,),
    ).item()

    image, label = dataset[index]

    opened = []

    for step in range(10):
        (
            state,
            probabilities,
            imagined_probabilities,
            imagined_entropy,
            imagined_confidence,
        ) = build_state(
            model,
            image,
            opened,
            device,
        )

        state_batch = state.unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = q_network(state_batch)

            q_values = mask_invalid_actions(
                q_values,
                state_batch,
            )

        q = q_values.squeeze(0)

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
        print("Candidate cells:")

        for cell in range(NUM_CELLS):
            if cell in opened:
                continue

            imagined_prediction = imagined_probabilities[cell].argmax().item()

            imagined_conf = imagined_confidence[cell].item()

            imagined_ent = imagined_entropy[cell].item()

            print(
                f"  cell {cell}: "
                f"Q={q[cell].item():+.4f} | "
                f"digit={imagined_prediction} | "
                f"conf={imagined_conf:.4f} | "
                f"entropy={imagined_ent:.4f}"
            )

        print(f"  STOP: Q={q[STOP_ACTION].item():+.4f}")

        action = q.argmax().item()

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

            # Show final current image.
            current_image, _ = reveal_cells(
                image,
                opened,
            )

            final_imagination, _ = imagine_full_image(
                model,
                image,
                opened,
                STOP_ACTION,
                device,
            )

            update_visualizer(
                fig,
                current_display,
                imagined_display,
                real_display,
                current_image,
                final_imagination,
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
        # Reveal
        # --------------------------------------------------

        print(f">>> Reveal cell {action}")

        imagined_image, imagined_probs = imagine_full_image(
            model,
            image,
            opened,
            action,
            device,
        )

        imagined_prediction = imagined_probs.argmax().item()

        imagined_confidence_value = imagined_probs.max().item()

        print("JEPA imagined full image:")

        print(f"  digit = {imagined_prediction}")

        print(f"  confidence = {imagined_confidence_value:.4f}")

        # Current image BEFORE opening cell.
        current_image, _ = reveal_cells(
            image,
            opened,
        )

        update_visualizer(
            fig,
            current_display,
            imagined_display,
            real_display,
            current_image,
            imagined_image,
            image,
            (
                f"[Image {episode_index}] Step {step + 1} | "
                f"action=cell {action} | "
                f"JEPA predicts "
                f"{imagined_prediction}"
            ),
        )

        opened.append(action)

        # Update immediately after actual reveal.
        current_image, _ = reveal_cells(
            image,
            opened,
        )

        update_visualizer(
            fig,
            current_display,
            imagined_display,
            real_display,
            current_image,
            imagined_image,
            image,
            (
                f"[Image {episode_index}] Revealed cell {action} | "
                f"opened={len(opened)}/9"
            ),
        )

    # Step cap reached without a STOP action.
    print()
    print(">>> Step cap reached (no STOP)")

    print(f"Prediction: {prediction}")

    print(f"Real label: {label}")

    print(f"Correct: {prediction == label}")

    print(f"Opened {len(opened)}/9 cells")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # --------------------------------------------------
    # JEPA
    # --------------------------------------------------

    model = ActiveJEPA(latent_dim=128).to(device)

    checkpoint = torch.load(
        "checkpoints/latest.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    # --------------------------------------------------
    # DQN
    # --------------------------------------------------

    q_network = QNetwork(
        state_dim=168,
        hidden_dim=256,
        num_actions=10,
    ).to(device)

    rl_checkpoint = torch.load(
        "checkpoints/rl_v2.pt",
        map_location=device,
        weights_only=False,
    )

    q_network.load_state_dict(rl_checkpoint["q_network"])

    q_network.eval()

    print("JEPA + DQN v2 loaded.")

    # --------------------------------------------------
    # MNIST
    # --------------------------------------------------

    dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # --------------------------------------------------
    # Visualizer (created once, reused for every image)
    # --------------------------------------------------

    (
        fig,
        ax_current,
        ax_imagined,
        ax_real,
        current_display,
        imagined_display,
        real_display,
    ) = create_visualizer()

    # Track whether the user closed the plot window, so we can stop
    # the auto-advance loop instead of erroring out on a dead figure.
    window_closed = {"value": False}

    def _on_close(event):
        window_closed["value"] = True

    fig.canvas.mpl_connect("close_event", _on_close)

    # --------------------------------------------------
    # Play through image after image automatically
    # --------------------------------------------------

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
            imagined_display,
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
