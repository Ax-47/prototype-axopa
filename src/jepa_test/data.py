import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

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
    ):
        self.dataset = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        num_opened = random.randint(
            0,
            8,
        )

        opened = random.sample(
            range(9),
            num_opened,
        )

        available = [cell for cell in range(9) if cell not in opened]

        action = random.choice(available)

        next_opened = opened + [action]

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
            # NEW:
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
