import json
import torch
import kornia
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Union, Tuple, TYPE_CHECKING

import consts

if TYPE_CHECKING:
    from data import DataPoints


class TransformingDataset(Dataset):
    """
        Wrapper na dataset, który wykonuje 
        transformacje surowych danych.
    """

    def __init__(
        self, wrapped: "DataPoints", image_transformer: "ImageTransformer" = None
    ):
        self.wrapped = wrapped
        self.image_transformer = (
            image_transformer if image_transformer else NopImageTransformer()
        )

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, idx: int) -> dict:
        item = self.wrapped.__getitem__(idx)

        rot_img = get_rotated_imgs(
            img=self.image_transformer.transform(item.img), angles=[item.angle]
        )[1][0]

        rot_grip_x, rot_grip_y = get_grip_pixel(
            angle=item.angle, pos=[item.grip_x, item.grip_y]
        )

        return {
            "idx": idx,
            "rot_img": rot_img,
            "rot_grip_x": torch.tensor(rot_grip_x),
            "rot_grip_y": torch.tensor(rot_grip_y),
            "grip_forces": torch.tensor(item.grip_force),
            "grip_success": torch.tensor(item.grip_success).float(),
            "hit": torch.tensor(item.hit),
            "throw_vel": torch.tensor(item.throw_vel),
            "sensors": torch.tensor(item.sensors),
            "all_grip_results": json.dumps(item.all_grip_results),
        }


class ImageTransformer:
    def transform(self, image: np.ndarray) -> np.ndarray:
        raise Exception("Unsupported operation!")

    def name(self) -> str:
        raise Exception("Unsupported operation!")


class NopImageTransformer(ImageTransformer):
    def transform(self, image: np.ndarray) -> np.ndarray:
        return image

    def name(self) -> str:
        return "nop"


class ScaleImageTransformer(ImageTransformer):
    def transform(self, image: np.ndarray) -> np.ndarray:
        image = image.copy()
        image[:, :, :3] /= 255.0
        return image

    def name(self) -> str:
        return "scale"


def default_angles() -> List[float]:
    step = (2 * np.pi) / consts.NUM_ROTS
    return [step * i for i in range(consts.NUM_ROTS)]


def get_rotated_imgs(
    *,
    img: Union[np.ndarray, torch.Tensor],
    angles: Union[List[float], None] = None,
    in_radians: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(img):
        img = torch.tensor(img)

    if len(img.shape) < 3 or len(img.shape) > 4:
        raise Exception(f"Given image has bad shape: {img.shape}")

    if len(img.shape) == 3:
        img = img.reshape([1] + list(img.shape))

    if img.shape[3] in [3, 4]:
        img = img.permute(0, 3, 1, 2)

    angles2 = angles if angles else default_angles()
    if in_radians:
        step = (2 * np.pi) / consts.NUM_ROTS
        angles2 = [(ang / step) for ang in angles2]
        angles2 = torch.tensor(angles2)
        angles2.round_()
        angles2 *= 360 / consts.NUM_ROTS
    angles2 = torch.tensor(angles2)

    if img.shape[2:] == (consts.WIDTH, consts.WIDTH):
        pad = (consts.TRANS_WIDTH - consts.WIDTH) // 2
        img = nn.functional.pad(img, [pad, pad, pad, pad])
    assert img.shape[2:] == (consts.TRANS_WIDTH, consts.TRANS_WIDTH), img.shape

    img = img.expand([angles2.shape[0]] + list(img.shape)[1:])

    rotated = kornia.geometry.transform.rotate(tensor=img, angle=(-angles2))
    angles2 *= (2 * np.pi) / 360.0
    return angles2, rotated


def get_grip_pos(
    *, angle: float, pixel: Union[Tuple[int, int], List[int], np.ndarray]
) -> Tuple[float, float]:
    """
        Zwraca pozycje w oryginalnym układzie odniesienia
        odpowiadającą danemu pixelowi na obrazie obróconym
        o dany kąt.
        Zwrócona pozycja ma niecałkowite współrzędne.
    """
    np_pixel = np.array(pixel, dtype=np.float32)
    if np_pixel.shape != (2,):
        raise Exception(f"Bad pixel: {np_pixel}")
    np_pixel -= (consts.TRANS_WIDTH - 1) / 2.0
    s = np.sin(angle).item()
    c = np.cos(angle).item()
    x = np_pixel[0] * c - np_pixel[1] * s
    y = np_pixel[0] * s + np_pixel[1] * c
    x += (consts.WIDTH - 1) / 2.0
    y += (consts.WIDTH - 1) / 2.0
    return x, y


def get_grip_pixel(
    *,
    angle: float,
    pos: Union[List[float], Tuple[float, float], np.ndarray],
    in_radians: bool = True,
    width: int = consts.WIDTH,
) -> Tuple[int, int]:
    """
        Zwraca współrzędne pixela, na obrazie obróconym
        o dany kąt, odpowiadającego danej współrzędnej
        w oryginalnym układzie odniesienia.
        Zwrócone współrzędne są całkowite.
    """
    np_pos = np.array(pos, dtype=np.float32)
    if np_pos.shape != (2,):
        raise Exception(f"Bad position: {np_pos}")

    if not in_radians:
        angle = (angle / 180.0) * np.pi

    np_pos -= float(width - 1) / 2
    s = np.sin(-angle)
    c = np.cos(-angle)
    x = np_pos[0] * c - np_pos[1] * s
    y = np_pos[0] * s + np_pos[1] * c
    x += (consts.TRANS_WIDTH - 1) / 2.0
    y += (consts.TRANS_WIDTH - 1) / 2.0
    x = int(round(x))
    y = int(round(y))

    if (x < 0) or (x >= consts.TRANS_WIDTH) or (y < 0) or (y >= consts.TRANS_WIDTH):
        raise Exception(f"Conversion exception! (x={x}, y={y})")

    return x, y
