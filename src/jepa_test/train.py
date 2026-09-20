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
    # JEPA latent prediction — GLOBAL level (level 1)
    # --------------------------------------------------

    jepa_loss_global = F.smooth_l1_loss(
        output["z_pred"],
        output["z_target"],
    )

    # --------------------------------------------------
    # JEPA latent prediction — LOCAL level (level 0)
    #
    # Without this term the local predictor / local tokens only ever get a
    # gradient indirectly through the pixel-level reconstruction_loss below,
    # so the "hierarchical" half of the model would never actually be doing
    # latent-space prediction (the whole point of JEPA), only autoencoding.
    # --------------------------------------------------

    jepa_loss_local = F.smooth_l1_loss(
        output["tokens_pred"],
        output["tokens_target"],
    )

    jepa_loss = 0.5 * jepa_loss_global + 0.5 * jepa_loss_local

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
    # Next partial observation (decoded from local tokens)
    #
    # BCE instead of MSE: the decoder ends in Sigmoid, and MSE's gradient
    # w.r.t. the pre-Sigmoid logit carries an extra sigmoid'(z) factor that
    # vanishes once the output saturates near 0 or 1 — so once the decoder
    # starts predicting "all black" (a decent local minimum on MNIST, which
    # is mostly background), gradient descent can get stuck there with a
    # gradient that's ~0 no matter how wrong it is. BCE's gradient reduces
    # to (prediction - target) with no extra vanishing factor, so it can
    # still escape that trap.
    # --------------------------------------------------

    reconstruction_loss = F.binary_cross_entropy(
        output["reconstruction"].clamp(1e-6, 1 - 1e-6),
        next_image,
    )

    # --------------------------------------------------
    # Full hidden image (decoded from global latent) — same Sigmoid/BCE
    # reasoning applies here too.
    # --------------------------------------------------

    full_image_loss = F.binary_cross_entropy(
        output["full_reconstruction"].clamp(1e-6, 1 - 1e-6),
        full_image,
    )

    # --------------------------------------------------
    # Total
    # --------------------------------------------------

    loss = (
        1.0 * jepa_loss
        + 1.0 * classification_loss
        + 0.5 * reconstruction_loss
        + 0.5 * full_image_loss
    )

    return loss, {
        "jepa": jepa_loss.item(),
        "jepa_global": jepa_loss_global.item(),
        "jepa_local": jepa_loss_local.item(),
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
                jepa_g=f"{metrics['jepa_global']:.4f}",
                jepa_l=f"{metrics['jepa_local']:.4f}",
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
