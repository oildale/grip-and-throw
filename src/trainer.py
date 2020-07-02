import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, cast, TYPE_CHECKING

import consts
from utils import trace_num_samples
from problem import (
    get_camera_snapshot,
    grip,
    throw,
    drop,
    get_sensors,
    get_picked_objs,
    get_obj_box,
    remove_obj,
)

if TYPE_CHECKING:
    from problem import Ctx
    from transforms import ImageTransformer
    from robot_control import GripperModule, AuxGripperModule, ThrowerModule
    from sampling import Sampler


def evaluate_one_step(
    *,
    ctx: "Ctx",
    image_transformer: "ImageTransformer",
    gripper: "GripperModule",
    aux_gripper: "AuxGripperModule",
    thrower: "ThrowerModule",
    target: Tuple[int, int],
):
    from data import DataPoint

    img = get_camera_snapshot(ctx)
    transformed_img = image_transformer.transform(img)
    grip_x, grip_y, angle = gripper.choose_grip(img=transformed_img)

    if (
        (grip_x < 0)
        or (grip_x >= consts.WIDTH)
        or (grip_y < 0)
        or (grip_y >= consts.WIDTH)
    ):
        return DataPoint(
            img=img,
            grip_x=grip_x,
            grip_y=grip_y,
            angle=angle,
            grip_force=-0.5,
            throw_vel=-1.3,
            sensors=np.zeros((4, 7)),
            target=target,
            grip_success=False,
            throw_success=False,
            hit=(-1, -1),
            out_of_tray=True,
            multi_grip=False,
            obj_type="???",
        )

    grip_force = aux_gripper.choose_grip(transformed_img, grip_x, grip_y, angle)

    grip(
        ctx=ctx,
        grip_x=grip_x,
        grip_y=grip_y,
        img=img,
        grip_angle=angle,
        grip_force=grip_force,
        target=target,
    )

    picked_objs = get_picked_objs(ctx)
    if picked_objs == []:
        return DataPoint(
            img=img,
            grip_x=grip_x,
            grip_y=grip_y,
            angle=angle,
            grip_force=grip_force,
            throw_vel=-1.1,
            sensors=np.zeros((4, 7)),
            target=target,
            grip_success=False,
            throw_success=False,
            hit=(-1, -1),
            multi_grip=False,
            out_of_tray=False,
            obj_type="???",
        )
    if len(picked_objs) > 1:
        drop(ctx)
        return DataPoint(
            img=img,
            grip_x=grip_x,
            grip_y=grip_y,
            angle=angle,
            grip_force=grip_force,
            throw_vel=-1.2,
            sensors=np.zeros((4, 7)),
            target=target,
            grip_success=False,
            throw_success=False,
            hit=(-1, -1),
            multi_grip=True,
            out_of_tray=False,
            obj_type="???",
        )

    assert len(picked_objs) == 1
    picked_obj = picked_objs[0]["id"]
    picked_obj_type = picked_objs[0]["type"]

    sensors = get_sensors(ctx)
    throw_vel = thrower.choose_throw(sensors=sensors, target=target,)

    throw(ctx=ctx, vel=throw_vel)
    hit = get_obj_box(ctx, picked_obj)
    hit = hit if hit else (-1, -1)
    remove_obj(ctx, picked_obj)

    return DataPoint(
        img=img,
        grip_x=grip_x,
        grip_y=grip_y,
        angle=angle,
        grip_force=grip_force,
        throw_vel=throw_vel,
        sensors=sensors,
        target=target,
        grip_success=True,
        throw_success=(hit == target),
        hit=hit,
        obj_type=picked_obj_type,
        out_of_tray=False,
        multi_grip=False,
    )


def run_offline(
    *,
    sampler: "Sampler",
    gripper: "GripperModule",
    aux_gripper: "AuxGripperModule",
    thrower: "ThrowerModule",
    train: bool,
    lr: float = 0.001,
) -> float:
    is_cuda = False
    try:
        is_cuda = next(gripper.parameters()).is_cuda
    except:
        try:
            is_cuda = next(aux_gripper.parameters()).is_cuda
        except:
            try:
                is_cuda = next(thrower.parameters()).is_cuda
            except:
                pass

    if train:
        gripper.train()
        aux_gripper.train()
        thrower.train()
    else:
        gripper.evaluate()
        aux_gripper.evaluate()
        thrower.evaluate()

    grip_loss_fn = nn.BCELoss()
    throw_loss_fn = nn.MSELoss()

    optimizer = None
    if train:
        parameters = []
        if gripper.is_trainable():
            parameters.append(gripper.parameters())
        if thrower.is_trainable():
            parameters.append(thrower.parameters())
        if aux_gripper.is_trainable():
            parameters.append(aux_gripper.parameters())
        if len(parameters) == 0:
            raise Exception("Nothing to train!")
        parameters = itertools.chain(*parameters)
        optimizer = optim.Adam(parameters, lr=lr)

    avg_loss = 0.0
    num_samples = len(sampler)

    for batch in trace_num_samples(sampler, step=10):
        if optimizer is not None:
            optimizer.zero_grad()

        idxs = batch["idx"]
        imgs = batch["rot_img"]
        grip_xs = batch["rot_grip_x"]
        grip_ys = batch["rot_grip_y"]
        grip_success = batch["grip_success"].float()
        hits = batch["hit"]
        throw_vels = batch["throw_vel"].float()
        sensors = batch["sensors"].float()

        batch_size = idxs.shape[0]

        if gripper.is_trainable():
            if is_cuda:
                grip_success = grip_success.cuda()
            grip_probs = gripper.forward(
                grip_pixel_xs=grip_xs, grip_pixel_ys=grip_ys, imgs=imgs
            )
            grip_loss = grip_loss_fn(grip_probs, grip_success)

            for sample_idx, glob_sample_idx in enumerate(idxs):
                sampler.update_weight(
                    sample_idx=glob_sample_idx.item(),
                    target_prob=grip_success[sample_idx].item(),
                    pred_prob=cast(float, grip_probs[sample_idx].item()),
                )

            grip_loss = grip_loss.mean()
        else:
            grip_loss = torch.tensor(0)
            if is_cuda:
                grip_loss = grip_loss.cuda()

        if aux_gripper.is_trainable():
            aux_grip_loss = aux_gripper.forward(batch)
        else:
            aux_grip_loss = torch.tensor(0.0)
            if is_cuda:
                aux_grip_loss = aux_grip_loss.cuda()

        if thrower.is_trainable():
            bad_throw_mask = (hits[:, 0] < 0) | (hits[:, 1] < 0)
            if is_cuda:
                bad_throw_mask = bad_throw_mask.cuda()
                throw_vels = throw_vels.cuda()

            hits.clamp_(0, 3)
            pred_throw_vels = thrower.forward(targets=hits, sensors=sensors,)

            throw_loss = throw_loss_fn(
                pred_throw_vels[~bad_throw_mask], throw_vels[~bad_throw_mask]
            )
            throw_loss *= (batch_size - bad_throw_mask.float().sum()) / batch_size
        else:
            throw_loss = torch.tensor(0.0)
            if is_cuda:
                throw_loss = throw_loss.cuda()

        loss = grip_loss + aux_grip_loss + throw_loss

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += (loss.item() * batch_size) / num_samples

    return avg_loss
