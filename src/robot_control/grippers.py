import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, cast

from robot_control.types import GripperModule
from transforms import get_rotated_imgs, get_grip_pos


class MLGripperModule(GripperModule):
    """
        Podklasa GripperModule oparta na modelu MLowym.
        MLGripperModule oparty jest na podejściu, że robimy
        jeden forward na jeden obrót oryginalnego obrazu.
        Cała logika obrotów i wyboru ostatecznej pozycji
        chwytu jest zawarta w tej klasie. Dzięki temu
        implementacja docelowych modułów będzie mogła
        się ograniczyć do zdefiniowania modelu
        i przekazania go do konstruktora MLGripperModule.
    """

    def __init__(self, model: nn.Module, name: str):
        super().__init__(name=name)
        self.model = model

    def is_trainable(self):
        return True

    def models(self):
        return {"model": self.model}

    def choose_grip(self, img: np.ndarray) -> Tuple[float, float, float]:
        angles, rot_imgs = get_rotated_imgs(img=img)

        if self.is_cuda:
            rot_imgs = rot_imgs.cuda()
        grip_probs = self.model.forward(rot_imgs)

        grip_probs = grip_probs.cpu()
        rot_idx, _, grip_x, grip_y = np.unravel_index(
            np.argmax(grip_probs), grip_probs.shape
        )

        angle = cast(float, angles[rot_idx].item())
        pos = get_grip_pos(angle=angle, pixel=[grip_x, grip_y])
        grip_x, grip_y = pos
        return grip_x, grip_y, angle

    def forward(
        self,
        *,
        grip_pixel_xs: torch.Tensor,
        grip_pixel_ys: torch.Tensor,
        imgs: torch.Tensor,
    ) -> torch.FloatTensor:
        if self.is_cuda:
            grip_pixel_xs = grip_pixel_xs.cuda()
            grip_pixel_ys = grip_pixel_ys.cuda()
            imgs = imgs.cuda()

        grip_probs = self.model.forward(imgs)
        bs = imgs.shape[0]
        indices = (
            torch.arange(bs).long(),
            torch.zeros(bs).long(),
            grip_pixel_xs,
            grip_pixel_ys,
        )
        return grip_probs[indices]


class ConvGripperModule(MLGripperModule):
    def __init__(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        super().__init__(model, name="conv")
