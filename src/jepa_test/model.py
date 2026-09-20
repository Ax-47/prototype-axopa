import copy

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                128 * 7 * 7,
                latent_dim,
            ),
        )

    def forward(self, image, mask):
        x = torch.cat(
            [image, mask],
            dim=1,
        )

        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(
        self,
        num_actions: int = 10,
        action_dim: int = 32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_actions,
            action_dim,
        )

    def forward(self, action):
        return self.embedding(action)


class Predictor(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        action_dim=32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                latent_dim,
            ),
        )

    def forward(
        self,
        z,
        action_embedding,
    ):
        x = torch.cat(
            [z, action_embedding],
            dim=-1,
        )

        return self.net(x)


class Classifier(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        num_classes=10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    """
    Predict next partial observation.
    """

    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                128 * 7 * 7,
            ),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)

        x = x.view(
            -1,
            128,
            7,
            7,
        )

        return self.deconv(x)


class FullImageDecoder(nn.Module):
    """
    Predict the complete hidden MNIST image.
    """

    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                128 * 7 * 7,
            ),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)

        x = x.view(
            -1,
            128,
            7,
            7,
        )

        return self.deconv(x)


class ActiveJEPA(nn.Module):
    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.encoder = Encoder(latent_dim)

        self.target_encoder = copy.deepcopy(self.encoder)

        self.action_encoder = ActionEncoder(
            num_actions=10,
            action_dim=32,
        )

        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=32,
        )

        self.classifier = Classifier(
            latent_dim=latent_dim,
            num_classes=10,
        )

        # Predict next partial observation.
        self.decoder = Decoder(latent_dim=latent_dim)

        # Predict complete original image.
        self.full_image_decoder = FullImageDecoder(latent_dim=latent_dim)

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_target(
        self,
        momentum=0.996,
    ):
        for online, target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data.mul_(momentum)
            target.data.add_(
                online.data,
                alpha=1.0 - momentum,
            )

    def encode(
        self,
        image,
        mask,
    ):
        return self.encoder(
            image,
            mask,
        )

    @torch.no_grad()
    def encode_target(
        self,
        image,
        mask,
    ):
        return self.target_encoder(
            image,
            mask,
        )

    def predict(
        self,
        z,
        action,
    ):
        action_embedding = self.action_encoder(action)

        return self.predictor(
            z,
            action_embedding,
        )

    def forward(
        self,
        image,
        mask,
        next_image,
        next_mask,
        full_image,
        action,
    ):
        z = self.encoder(
            image,
            mask,
        )

        with torch.no_grad():
            z_target = self.target_encoder(
                next_image,
                next_mask,
            )

        action_embedding = self.action_encoder(action)

        z_pred = self.predictor(
            z,
            action_embedding,
        )

        logits_current = self.classifier(z)

        logits_pred = self.classifier(z_pred)

        # Predict next partial observation.
        reconstruction = self.decoder(z_pred)

        # Predict complete hidden image.
        full_reconstruction = self.full_image_decoder(z_pred)

        return {
            "z": z,
            "z_target": z_target,
            "z_pred": z_pred,
            "logits_current": logits_current,
            "logits_pred": logits_pred,
            "reconstruction": reconstruction,
            "full_reconstruction": full_reconstruction,
        }
