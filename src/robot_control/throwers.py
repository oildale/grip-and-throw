import torch
import torch.nn as nn
from typing import Tuple, cast
import numpy as np

from robot_control.types import ThrowerModule


class SimpleEmbeddingThrowerModule(ThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        self.embedding = nn.Embedding(num_embeddings=16, embedding_dim=1)
        super().__init__(name="emb")

    def models(self) -> dict:
        return {"embedding": self.embedding}

    def is_trainable(self) -> bool:
        return True

    def forward(self, targets: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        if self.is_cuda:
            targets = targets.cuda()
        return self.embedding(targets[:, 0] * 4 + targets[:, 1])[:, 0]

    def choose_throw(self, *, target: Tuple[int, int], sensors: np.ndarray) -> float:
        target2 = torch.tensor([target[0] * 4 + target[1]])
        if self.is_cuda:
            target2 = target2.cuda()
        throw_vel = self.embedding(target2)[0][0].item()
        return cast(float, throw_vel)


class TargetAndSensorsThrowerModule(ThrowerModule):
    def __init__(self, ff: nn.Module, name: str):
        super().__init__(name=name)
        self.embedding = nn.Embedding(num_embeddings=16, embedding_dim=4)
        self.ff = ff

    def is_trainable(self) -> bool:
        return True

    def models(self) -> dict:
        return {"embedding": self.embedding, "ff": self.ff}

    def forward(self, targets: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        batch_size = sensors.shape[0]

        if self.is_cuda:
            targets = targets.cuda()
            sensors = sensors.cuda()

        target_emb = self.embedding(targets[:, 0] * 4 + targets[:, 1])
        sensors = sensors.reshape((batch_size, -1))
        ff_input = torch.cat([target_emb, sensors], dim=1)

        return self.ff(ff_input)[:, 0]

    def choose_throw(self, *, target: Tuple[int, int], sensors: np.ndarray) -> float:
        out = self.forward(
            targets=torch.tensor([target]), sensors=torch.tensor([sensors]).float(),
        )
        return cast(float, out[0].item())


class FF1ThrowerModule(TargetAndSensorsThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        ff = nn.Sequential(nn.Linear(32, 1), nn.ReLU())
        super().__init__(ff, "ff1")


class FF2ThrowerModule(TargetAndSensorsThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        ff = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )
        super().__init__(ff, "ff2")


class FF3ThrowerModule(TargetAndSensorsThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        ff = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        super().__init__(ff, "ff3")


class FF4ThrowerModule(TargetAndSensorsThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        ff = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        super().__init__(ff, "ff4")


class FF5ThrowerModule(TargetAndSensorsThrowerModule):
    def __init__(self):
        torch.manual_seed(42)
        ff = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        super().__init__(ff, "ff5")
