import os
import numpy as np
import numpy.random as nr
from typing import Tuple, cast

import consts
from robot_control.types import GripperModule, ThrowerModule, AuxGripperModule
from transforms import get_rotated_imgs, get_grip_pos, default_angles


class MissingGripperModule(GripperModule):
    """
        Moduł, który zawsze zwraca pozycje poza tacą,
        ale odpowiadającą jakiemuś pixelowi w obróconym obrazie.
        Moduł ten jest używany do generowania przykładów,
        z których modele mają się nauczyć, że pozycje poza
        tacą są kiepskie. Używając jedynie przykładów
        wygenerowanych przez gripper zawsze trafiający
        w tacę (np. BaselineGripperModule) istnieje
        ryzyko, że model nie nauczy się nic o pozycjach
        poza tacą.
    """

    def __init__(self):
        super().__init__(name="miss")

    def choose_grip(self, img: np.ndarray) -> Tuple[float, float, float]:
        angles, rot_imgs = get_rotated_imgs(img=img)
        num_angles = angles.shape[0]
        side = rot_imgs.shape[3]

        rng = _rng_from_array(arr=img)
        rot_idx = rng.randint(0, num_angles)
        angle = cast(float, angles[rot_idx].item())
        while True:
            grip_x, grip_y = rng.randint(0, side), rng.randint(0, side)

            pos = get_grip_pos(angle=angle, pixel=[grip_x, grip_y])

            if (
                pos[0] < 0
                or pos[0] > consts.WIDTH
                or pos[1] < 0
                or pos[1] > consts.WIDTH
            ):
                return pos[0], pos[1], angle


class BaselineGripperModule(GripperModule):
    def __init__(self):
        super().__init__(name="base")

    def choose_grip(self, img: np.ndarray) -> Tuple[float, float, float]:
        grip_x, grip_y = np.unravel_index(np.argmax(img[:, :, 3]), img.shape[:2])
        rng = _rng_from_array(arr=img)
        grip_x = np.clip(grip_x + rng.randint(-5, 6), a_min=0, a_max=consts.WIDTH - 1)
        grip_y = np.clip(grip_y + rng.randint(-5, 6), a_min=0, a_max=consts.WIDTH - 1)
        angle = default_angles()
        angle = angle[rng.randint(0, len(angle))]
        return grip_x, grip_y, angle


class BaselineAuxGripperModule(AuxGripperModule):
    def __init__(self, *, force: float = -1.0):
        super().__init__(name="base")
        self.force = force

    def choose_grip(
        self, img: np.ndarray, grip_x: float, grip_y: float, angle: float
    ) -> float:
        if self.force < 0.0:
            raise Exception("Uninitialized model!")
        return self.force

    def load(self, model_dir: str):
        with open(os.path.join(model_dir, "model"), "r") as f:
            self.force = float(f.read())

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=False)
        with open(os.path.join(model_dir, "model"), "w") as f:
            f.write(str(self.force))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"BaselineAuxGripperModule(force={self.force})"


class RandomAuxGripperModule(AuxGripperModule):
    def __init__(self):
        super().__init__(name="random")

    def choose_grip(
        self, img: np.ndarray, grip_x: float, grip_y: float, angle: float
    ) -> float:
        rng = _rng_from_array(arr=img)
        return 21 * rng.random() + 4


class BaselineThrowerModule(ThrowerModule):
    def __init__(self):
        super().__init__(name="base")

    def choose_throw(self, *, target: Tuple[int, int], sensors: np.ndarray) -> float:
        rng = _rng_from_array(arr=sensors)
        vel = rng.random() * 3 + 3
        return vel


def _rng_from_array(arr: np.ndarray) -> nr.RandomState:
    seed = arr.flatten()
    seed = seed * np.arange(seed.shape[0])
    seed = int(seed.sum()) % (1 << 30)
    return nr.RandomState(seed=seed)
