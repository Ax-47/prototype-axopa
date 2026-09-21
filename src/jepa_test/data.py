import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

NUM_CELLS = 9
STOP_ACTION = 9
NUM_ACTIONS = 10

CELL_ROWS = [
    (0, 9),
    (9, 18),
    (18, 28),
]

CELL_COLS = [
    (0, 9),
    (9, 18),
    (18, 28),
]


def split_indices(
    total=60_000,
    jepa_size=50_000,
    val_size=2_000,
    seed=0,
):
    """Deterministic split of the 60k MNIST *train* images into three
    disjoint parts:

      jepa      -- trains the JEPA (train.py)
      rl_train  -- episodes the DQN trains on (train_rl.py)
      rl_val    -- held out from BOTH; used for validation / model selection

    The JEPA classifier is over-confident on the images it was trained on, so
    the DQN must learn *when to stop* from images the JEPA has never seen.
    (The real MNIST test set stays untouched for eval_rl.py / play.py.)
    """

    generator = torch.Generator().manual_seed(seed)

    permutation = torch.randperm(
        total,
        generator=generator,
    ).tolist()

    jepa = permutation[:jepa_size]

    rl_train = permutation[jepa_size : total - val_size]

    rl_val = permutation[total - val_size :]

    return jepa, rl_train, rl_val


def reveal_cells(
    image: torch.Tensor,
    opened: list[int],
):
    visible = torch.zeros_like(image)
    mask = torch.zeros_like(image)

    for cell in opened:
        row = cell // 3
        col = cell % 3

        r0, r1 = CELL_ROWS[row]
        c0, c1 = CELL_COLS[col]

        visible[
            :,
            r0:r1,
            c0:c1,
        ] = image[
            :,
            r0:r1,
            c0:c1,
        ]

        mask[
            :,
            r0:r1,
            c0:c1,
        ] = 1.0

    return visible, mask


class ActiveMNIST(Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        indices=None,
    ):
        self.dataset = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Optional subset (see split_indices).
        self.indices = None if indices is None else list(indices)

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)

        return len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        image, label = self.dataset[index]

        # 0..9 cells already open. It used to be 0..8, so the encoder never
        # saw a fully revealed image as INPUT -- but the RL environment ends
        # with all 9 cells open (max_steps=9) and classifies exactly that.
        num_opened = random.randint(
            0,
            NUM_CELLS,
        )

        opened = random.sample(
            range(NUM_CELLS),
            num_opened,
        )

        available = [cell for cell in range(NUM_CELLS) if cell not in opened]

        if available:
            action = random.choice(available)

            next_opened = opened + [action]

        else:
            # Everything is already open: the only legal action is STOP, and
            # STOP leaves the observation unchanged (the predictors are exact
            # identity maps for STOP, see model.py).
            action = STOP_ACTION

            next_opened = opened

        current_image, current_mask = reveal_cells(
            image,
            opened,
        )

        next_image, next_mask = reveal_cells(
            image,
            next_opened,
        )

        return {
            "image": current_image,
            "mask": current_mask,
            "next_image": next_image,
            "next_mask": next_mask,
            # complete original image
            "full_image": image,
            "action": torch.tensor(
                action,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }
