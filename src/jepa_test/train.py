import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ActiveMNIST
from .model import ActiveJEPA


def compute_loss(
    output,
    next_image,
    full_image,
    label,
):
    # --------------------------------------------------
    # JEPA latent prediction
    # --------------------------------------------------

    jepa_loss = F.smooth_l1_loss(
        output["z_pred"],
        output["z_target"],
    )

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    classification_current = F.cross_entropy(
        output["logits_current"],
        label,
    )

    classification_pred = F.cross_entropy(
        output["logits_pred"],
        label,
    )

    classification_loss = 0.5 * classification_current + 0.5 * classification_pred

    # --------------------------------------------------
    # Next partial observation
    # --------------------------------------------------

    reconstruction_loss = F.mse_loss(
        output["reconstruction"],
        next_image,
    )

    # --------------------------------------------------
    # Full hidden image
    # --------------------------------------------------

    full_image_loss = F.mse_loss(
        output["full_reconstruction"],
        full_image,
    )

    # --------------------------------------------------
    # Total
    # --------------------------------------------------

    loss = (
        1.0 * jepa_loss
        + 1.0 * classification_loss
        + 0.1 * reconstruction_loss
        + 0.5 * full_image_loss
    )

    return loss, {
        "jepa": jepa_loss.item(),
        "classification": classification_loss.item(),
        "reconstruction": reconstruction_loss.item(),
        "full_image": full_image_loss.item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    dataset = ActiveMNIST(
        root="./data",
        train=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = ActiveJEPA(latent_dim=128).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    epochs = 10

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    for epoch in range(epochs):
        model.train()

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        total_loss = 0.0

        for batch in progress:
            image = batch["image"].to(device)

            mask = batch["mask"].to(device)

            next_image = batch["next_image"].to(device)

            next_mask = batch["next_mask"].to(device)

            full_image = batch["full_image"].to(device)

            action = batch["action"].to(device)

            label = batch["label"].to(device)

            output = model(
                image=image,
                mask=mask,
                next_image=next_image,
                next_mask=next_mask,
                full_image=full_image,
                action=action,
            )

            loss, metrics = compute_loss(
                output,
                next_image,
                full_image,
                label,
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            model.update_target(momentum=0.996)

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                jepa=f"{metrics['jepa']:.4f}",
                cls=f"{metrics['classification']:.4f}",
                full=f"{metrics['full_image']:.4f}",
            )

        average_loss = total_loss / len(loader)

        print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            "checkpoints/latest.pt",
        )


if __name__ == "__main__":
    main()
