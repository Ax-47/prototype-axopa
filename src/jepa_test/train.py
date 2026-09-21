import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import STOP_ACTION, ActiveMNIST, split_indices
from .model import ActiveJEPA


def compute_loss(
    output,
    next_image,
    full_image,
    label,
    action,
):
    # --------------------------------------------------
    # JEPA latent prediction — GLOBAL level (level 2)
    # --------------------------------------------------

    jepa_loss_global = F.smooth_l1_loss(
        output["z_pred"],
        output["z_target"],
    )

    # --------------------------------------------------
    # JEPA latent prediction — MID level (level 1)
    # --------------------------------------------------

    jepa_loss_mid = F.smooth_l1_loss(
        output["tokens_mid_pred"],
        output["tokens_mid_target"],
    )

    # --------------------------------------------------
    # JEPA latent prediction — LOCAL level (level 0)
    #
    # Without these per-level terms, the mid/local predictors and tokens
    # only ever get a gradient indirectly through the pixel-level
    # reconstruction losses below, so the "hierarchical" part of the model
    # would never actually be doing latent-space prediction (the whole
    # point of JEPA) at those levels, only autoencoding.
    # --------------------------------------------------

    jepa_loss_local = F.smooth_l1_loss(
        output["tokens_local_pred"],
        output["tokens_local_target"],
    )

    jepa_loss = (
        (1.0 / 3.0) * jepa_loss_global
        + (1.0 / 3.0) * jepa_loss_mid
        + (1.0 / 3.0) * jepa_loss_local
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
    # Expected-accuracy head (feeds the RL state + the greedy planner)
    #
    # Target = "is the classifier actually CORRECT on the real next
    # observation?" for the action that was really taken, plus "is it correct
    # right now?" for the STOP output (index 9). Binary targets + BCE make
    # the sigmoid output an estimate of E[correct | z, opened cells, action],
    # i.e. the value of information of that action. The head reads z.detach(),
    # so this loss only trains the head, never the representation.
    # --------------------------------------------------

    correct_now = (output["logits_current"].argmax(dim=-1) == label).float()

    correct_next = (output["logits_next"].argmax(dim=-1) == label).float()

    voi_logits = output["voi_logits"]

    voi_action = voi_logits.gather(
        1,
        action.unsqueeze(1),
    ).squeeze(1)

    voi_loss = F.binary_cross_entropy_with_logits(
        voi_action,
        correct_next,
    ) + F.binary_cross_entropy_with_logits(
        voi_logits[:, STOP_ACTION],
        correct_now,
    )

    # --------------------------------------------------
    # Next partial observation, finest resolution (decoded from local tokens)
    #
    # BCE instead of MSE: the decoder ends in Sigmoid, and MSE's gradient
    # w.r.t. the pre-Sigmoid logit carries an extra sigmoid'(z) factor that
    # vanishes once the output saturates near 0 or 1 — so once a decoder
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
    # Next partial observation, coarser resolution (decoded from region
    # tokens) — same Sigmoid/BCE reasoning applies.
    # --------------------------------------------------

    mid_reconstruction_loss = F.binary_cross_entropy(
        output["mid_reconstruction"].clamp(1e-6, 1 - 1e-6),
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
        + 0.5 * voi_loss
        + 0.5 * reconstruction_loss
        + 0.3 * mid_reconstruction_loss
        + 0.5 * full_image_loss
    )

    return loss, {
        "jepa": jepa_loss.item(),
        "jepa_global": jepa_loss_global.item(),
        "jepa_mid": jepa_loss_mid.item(),
        "jepa_local": jepa_loss_local.item(),
        "classification": classification_loss.item(),
        "voi": voi_loss.item(),
        "reconstruction": reconstruction_loss.item(),
        "mid_reconstruction": mid_reconstruction_loss.item(),
        "full_image": full_image_loss.item(),
    }


def unpack_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Held-out pass (images the JEPA never trains on). Returns classifier
    accuracy on the current / predicted latent and the accuracy-head loss."""

    model.eval()

    correct_current = 0.0
    correct_pred = 0.0
    voi_total = 0.0
    seen = 0

    for batch in loader:
        batch = unpack_batch(batch, device)

        output = model(
            image=batch["image"],
            mask=batch["mask"],
            next_image=batch["next_image"],
            next_mask=batch["next_mask"],
            full_image=batch["full_image"],
            action=batch["action"],
        )

        _, metrics = compute_loss(
            output,
            batch["next_image"],
            batch["full_image"],
            batch["label"],
            batch["action"],
        )

        label = batch["label"]

        size = label.shape[0]

        correct_current += (output["logits_current"].argmax(-1) == label).sum().item()

        correct_pred += (output["logits_pred"].argmax(-1) == label).sum().item()

        voi_total += metrics["voi"] * size

        seen += size

    return {
        "acc_current": correct_current / seen,
        "acc_pred": correct_pred / seen,
        "voi": voi_total / seen,
    }


def main(
    epochs=10,
    batch_size=128,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # The JEPA only ever sees its own split; the DQN trains on a different
    # one and validation on a third (see data.split_indices).
    jepa_indices, _, val_indices = split_indices()

    dataset = ActiveMNIST(
        root="./data",
        train=True,
        indices=jepa_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ActiveMNIST(
            root="./data",
            train=True,
            indices=val_indices,
        ),
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
    )

    model = ActiveJEPA(latent_dim=128).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # update_target() uses a cosine EMA schedule (momentum ramps from
    # base_momentum up to 1.0 over training), so it needs the current and
    # total *optimizer* step count rather than a single fixed momentum.
    total_steps = epochs * len(loader)

    global_step = 0

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
            batch = unpack_batch(batch, device)

            image = batch["image"]

            mask = batch["mask"]

            next_image = batch["next_image"]

            next_mask = batch["next_mask"]

            full_image = batch["full_image"]

            action = batch["action"]

            label = batch["label"]

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
                action,
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            model.update_target(
                step=global_step,
                total_steps=total_steps,
            )

            global_step += 1

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                jepa_g=f"{metrics['jepa_global']:.4f}",
                jepa_m=f"{metrics['jepa_mid']:.4f}",
                jepa_l=f"{metrics['jepa_local']:.4f}",
                cls=f"{metrics['classification']:.4f}",
                voi=f"{metrics['voi']:.4f}",
                full=f"{metrics['full_image']:.4f}",
            )

        average_loss = total_loss / len(loader)

        val = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}: loss={average_loss:.4f} | "
            f"val acc(current)={val['acc_current']:.4f} "
            f"acc(pred)={val['acc_pred']:.4f} "
            f"voi={val['voi']:.4f}"
        )

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
