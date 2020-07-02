# Przykładowy program pokazujący jak
# wygenerować zbiór, wytrenować model
# i zrobić jego ewaluację.

import os

from data import generate_data
from robot_control import (
    get_gripper_by_name,
    get_thrower_by_name,
    get_aux_gripper_by_name,
)
from transforms import ScaleImageTransformer, TransformingDataset
from trainer import run_offline
import metrics
from utils import cprint
from sampling import construct_sampler


if __name__ == "__main__":
    heur_gripper = get_gripper_by_name("base")
    gripper = get_gripper_by_name("conv")
    aux_gripper = get_aux_gripper_by_name("base", force=4)
    heur_thrower = get_thrower_by_name("base")
    thrower = get_thrower_by_name("emb")
    image_transformer = ScaleImageTransformer()

    os.system("rm -rf /tmp/grip-and-throw")
    dataset = generate_data(
        gripper=heur_gripper,
        aux_gripper=aux_gripper,
        thrower=heur_thrower,
        image_transformer=image_transformer,
        seeds=[6],
        length=2,
        on_cuda=False,
        workdir_path="/tmp/grip-and-throw/example_dataset",
    )
    tra_dataset = TransformingDataset(
        wrapped=dataset, image_transformer=image_transformer
    )

    for epoch_idx in range(200):
        grip_loss = run_offline(
            sampler=construct_sampler(
                sampling_name="shuf", dataset=tra_dataset, batch_size=2, num_workers=0
            ),
            gripper=gripper,
            aux_gripper=aux_gripper,
            thrower=heur_thrower,
            train=True,
        )
        throw_loss = run_offline(
            sampler=construct_sampler(
                sampling_name="shuf", dataset=tra_dataset, batch_size=2, num_workers=0
            ),
            gripper=heur_gripper,
            aux_gripper=aux_gripper,
            thrower=thrower,
            train=True,
            lr=(0.1 if epoch_idx < 100 else 0.001),
        )
        cprint(
            "epoch_idx=",
            epoch_idx,
            "| grip_loss=",
            grip_loss,
            "| throw_loss=",
            throw_loss,
        )

    gripper.save("/tmp/grip-and-throw/example_gripper")
    thrower.save("/tmp/grip-and-throw/example_thrower")

    gripper.load("/tmp/grip-and-throw/example_gripper")
    thrower.load("/tmp/grip-and-throw/example_thrower")

    eval_dataset = generate_data(
        gripper=gripper,
        aux_gripper=aux_gripper,
        thrower=thrower,
        image_transformer=image_transformer,
        seeds=[7],
        length=2,
        on_cuda=False,
        workdir_path="/tmp/grip-and-throw/example_eval_dataset",
    )
    cprint(metrics.calculate_metrics(eval_dataset))
