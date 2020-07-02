import json
import numpy as np
import torch
import torch.nn as nn
from typing import List

from robot_control.types import AuxGripperModule
from transforms import get_rotated_imgs, get_grip_pixel


class BaseContAuxGripperModule(AuxGripperModule):
    def __init__(self, model: nn.Module, name: str):
        super().__init__(name=name)
        self.model = model
        self.loss_fn = nn.MSELoss()

    def is_trainable(self) -> bool:
        return True

    def models(self) -> dict:
        return {"model": self.model}

    def choose_grip(
        self, img: np.ndarray, grip_x: float, grip_y: float, angle: float
    ) -> float:
        _, rot_imgs = get_rotated_imgs(img=img, angles=[angle])
        grip_x, grip_y = get_grip_pixel(angle=angle, pos=[grip_x, grip_y])
        if self.is_cuda:
            rot_imgs = rot_imgs.cuda()
        grip_forces = self.model.forward(rot_imgs).cpu()
        grip_force = grip_forces[0, 0, grip_x, grip_y].item()
        return grip_force

    def forward(self, batch: dict) -> torch.Tensor:
        imgs = batch["rot_img"]
        grip_xs = batch["rot_grip_x"]
        grip_ys = batch["rot_grip_y"]
        grip_forces = batch["grip_forces"].float()
        grip_succ = batch["grip_success"]

        if self.is_cuda:
            imgs = imgs.cuda()
            grip_xs = grip_xs.cuda()
            grip_ys = grip_ys.cuda()
            grip_forces = grip_forces.cuda()
            grip_succ = grip_succ.cuda()

        bs = imgs.shape[0]
        preds = self.model.forward(imgs)
        indices = (torch.arange(bs).long(), torch.zeros(bs).long(), grip_xs, grip_ys)
        preds2 = preds[indices]

        bad_grips = grip_succ < 0.5
        loss = self.loss_fn(preds2[~bad_grips], grip_forces[~bad_grips])
        loss *= (bs - bad_grips.float().sum()) / bs

        return loss


class ContAuxGripperModule(BaseContAuxGripperModule):
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
        )
        super().__init__(model, "cont")


class BaseDiscAuxGripperModule(AuxGripperModule):
    def __init__(self, model: nn.Module, forces: List[float], name: str):
        super().__init__(name=name)
        self.model = model
        self.forces = forces
        self.emb_dim = 4
        self.emb = nn.Embedding(num_embeddings=len(forces), embedding_dim=self.emb_dim)

    def is_trainable(self) -> bool:
        return True

    def models(self) -> dict:
        return {"model": self.model, "emb": self.emb}

    def choose_grip(
        self, img: np.ndarray, grip_x: float, grip_y: float, angle: float
    ) -> float:
        _, rot_img = get_rotated_imgs(img=img, angles=[angle])
        grip_x, grip_y = get_grip_pixel(angle=angle, pos=[grip_x, grip_y])
        if self.is_cuda:
            rot_img = rot_img.cuda()
        rot_imgs = rot_img.expand((len(self.forces), -1, -1, -1))
        w = rot_imgs.shape[2]
        force_idxs = torch.arange(len(self.forces))
        if self.is_cuda:
            force_idxs = force_idxs.cuda()
        embs = self.emb(force_idxs)
        embs = embs.reshape((len(self.forces), self.emb_dim, 1, 1))
        embs = embs.expand((len(self.forces), self.emb_dim, w, w))
        inp = torch.cat([rot_imgs, embs], dim=1)
        score_maps = self.model(inp)
        scores = score_maps[
            (
                torch.arange(len(self.forces)).long(),
                torch.zeros(len(self.forces)).long(),
                (torch.ones(len(self.forces)) * grip_x).long(),
                (torch.ones(len(self.forces)) * grip_y).long(),
            )
        ].cpu()
        force_idx = np.unravel_index(np.argmax(scores), scores.shape)
        force_idx = force_idx[0]
        return self.forces[force_idx]

    def forward(self, batch: dict) -> torch.Tensor:
        imgs = batch["rot_img"]
        grip_xs = batch["rot_grip_x"]
        grip_ys = batch["rot_grip_y"]

        bs = imgs.shape[0]
        nf = len(self.forces)
        w = imgs.shape[2]

        imgs = imgs.repeat_interleave(nf, dim=0)
        grip_xs = grip_xs.repeat_interleave(nf, dim=0)
        grip_ys = grip_ys.repeat_interleave(nf, dim=0)
        force_idxs = torch.arange(nf).repeat(bs)
        if self.is_cuda:
            imgs = imgs.cuda()
            grip_xs = grip_xs.cuda()
            grip_ys = grip_ys.cuda()
            force_idxs = force_idxs.cuda()
        embs = self.emb(force_idxs)
        embs2 = embs.reshape((bs * nf, self.emb_dim, 1, 1))
        embs3 = embs2.expand((bs * nf, self.emb_dim, w, w))
        inp = torch.cat([imgs, embs3], dim=1)
        score_maps = self.model(inp)
        scores = score_maps[
            (
                torch.arange(bs * nf).long(),
                torch.zeros(bs * nf).long(),
                grip_xs,
                grip_ys,
            )
        ]
        weights = []
        for sample_idx in range(bs):
            all_grip_results = json.loads(batch["all_grip_results"][sample_idx])
            if len(all_grip_results) < 2:
                raise Exception("DataPoint contains only one grip force!")
            pos_cnt = sum([1 for r in all_grip_results.values() if r])
            neg_cnt = sum([1 for r in all_grip_results.values() if not r])
            for force_idx in range(nf):
                force = self.forces[force_idx]
                if all_grip_results[str(force)]:
                    weights.append(-neg_cnt)
                else:
                    weights.append(pos_cnt)
        weights = torch.tensor(weights).float()
        if self.is_cuda:
            weights = weights.cuda()
        loss = (scores * weights).sum()
        return loss


class Disc1AuxGripperModule(BaseDiscAuxGripperModule):
    def __init__(self):
        torch.manual_seed(42)
        super().__init__(
            model=nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32, track_running_stats=True),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            ),
            forces=[4.0, 6.0, 9.0, 16.0, 20.0, 25.0],
            name="disc1",
        )


class Disc2AuxGripperModule(BaseDiscAuxGripperModule):
    def __init__(self):
        torch.manual_seed(42)
        super().__init__(
            model=nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=32, track_running_stats=True),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            ),
            forces=[4.0, 6.0, 9.0, 16.0, 20.0, 25.0],
            name="disc2",
        )
