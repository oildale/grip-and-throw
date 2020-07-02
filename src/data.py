import os
import time
import json
import pickle
import argparse
import numpy as np
import numpy.random as nr
from typing import Tuple, List, Union, Dict, TYPE_CHECKING
from torch.utils.data import Dataset

import consts

if TYPE_CHECKING:
    from robot_control import GripperModule, AuxGripperModule, ThrowerModule
    from transforms import ImageTransformer
from trainer import evaluate_one_step
from problem import init_problem, spawn_objects, deinit_problem
from utils import save_obj, load_obj, trace_num_samples, cprint


class _TestCasesGenerator:
    def __init__(
        self,
        seed: int,
        length: int,
        repeat: int = 1,  # Ile razy powtórzyć każdy z przypadków
    ):
        self.seed = seed
        self.length = length
        self.repeat = repeat

    @staticmethod
    def _next_obj_desc(rng):
        return [
            rng.randint(consts.WIDTH // 3, 2 * (consts.WIDTH // 3)),
            rng.randint(consts.WIDTH // 3, 2 * (consts.WIDTH // 3)),
            rng.rand() * np.pi * 2,
            rng.rand() * np.pi * 2,
            rng.rand() * np.pi * 2,
            rng.choice(["cube", "hammer", "cross"]),
        ]

    @staticmethod
    def _next_target(rng) -> Tuple[int, int]:
        return (rng.randint(0, 4), rng.randint(0, 4))

    def __iter__(self):
        od_rng = nr.RandomState(seed=self.seed)
        t_rng = nr.RandomState(seed=self.seed)

        for _ in range(self.length):
            target = _TestCasesGenerator._next_target(t_rng)
            obj_desc = _TestCasesGenerator._next_obj_desc(od_rng)
            for _ in range(self.repeat):
                ctx = init_problem()
                spawn_objects(ctx, [obj_desc])
                yield (ctx, target)
                deinit_problem(ctx)


class DataPoint:
    def __init__(
        self,
        *,
        img: np.ndarray,
        grip_x: float,
        grip_y: float,
        angle: float,
        grip_force: float,
        throw_vel: float,
        sensors: np.ndarray,
        target: Tuple[int, int],
        hit: Tuple[int, int],
        grip_success: bool,
        throw_success: bool,
        obj_type: str,
        out_of_tray: bool,
        multi_grip: bool,
    ):
        self.img = img
        self.grip_x = grip_x
        self.grip_y = grip_y
        self.angle = angle
        self.grip_force = grip_force
        self.throw_vel = throw_vel
        self.sensors = sensors
        self.target = target
        self.hit = hit
        self.grip_success = grip_success
        self.throw_success = throw_success
        self.obj_type = obj_type
        self.out_of_tray = out_of_tray
        self.multi_grip = multi_grip
        # Jeżeli do generowania przykładów podaliśmy
        # listę sił chwytu do przetestowania,
        # to all_grip_results zawiera mapę
        # z siły chwytu w rezultat.
        # Pozostałe pola DataPoint'a zawierają
        # dane dotyczące próby z użyciem pierwszej siły.
        self.all_grip_results: Dict[float, bool] = {}

    def __str__(self) -> str:
        s = "DataPoint(\n"
        s += f"  img=[{self.img.shape}, {type(self.img)}],\n"
        s += f"  grip_x={self.grip_x},\n"
        s += f"  grip_y={self.grip_y},\n"
        s += f"  angle={self.angle},\n"
        s += f"  grip_force={self.grip_force},\n"
        s += f"  throw_vel={self.throw_vel},\n"
        s += f"  sensors=[{self.sensors.shape}, {type(self.sensors)}],\n"
        s += f"  target={self.target},\n"
        s += f"  hit={self.hit},\n"
        s += f"  grip_success={self.grip_success},\n"
        s += f"  throw_success={self.throw_success},\n"
        s += f"  obj_type={self.obj_type},\n"
        s += f"  out_of_tray={self.out_of_tray}\n"
        s += f"  multi_grip={self.multi_grip}\n"
        s += f"  all_grip_results={self.all_grip_results}\n"
        s += ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def deserialize(filepath: str) -> "DataPoint":
        with open(filepath, "rb") as f:
            data_point = pickle.load(f)

        with open(filepath + ".rgb_img", "rb") as f:
            rgb_img = np.load(f).astype(np.float32)
        with open(filepath + ".dep_img", "rb") as f:
            dep_img = np.load(f)
        dep_img = dep_img.reshape(list(dep_img.shape) + [1])
        data_point.img = np.concatenate([rgb_img, dep_img], axis=2)

        return data_point

    def serialize(self, filepath: str):
        img, self.img = self.img, None
        dep_img = img[:, :, 3]
        rgb_img = img[:, :, :3].astype(np.uint8)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        with open(filepath + ".dep_img", "wb") as f:
            np.save(f, dep_img)
        with open(filepath + ".rgb_img", "wb") as f:
            np.save(f, rgb_img)
        self.img = img


class DataPoints(Dataset):
    def __len__(self) -> int:
        raise Exception("Unsupported operation!")

    def __getitem__(self, index: int) -> DataPoint:
        raise Exception("Unsupported operation!")

    def take(self, n: int) -> "DataPoints":
        return _CroppedDataPoints(self, n=n, take=True)

    def drop(self, n: int) -> "DataPoints":
        return _CroppedDataPoints(self, n=n, take=False)


def construct_dataset(dataset_dirpath: str) -> DataPoints:
    with open(os.path.join(dataset_dirpath, "metadata"), "r") as f:
        metadata = json.load(f)
    if metadata["dataset_type"] == "composed":
        return _ComposedDataPoints(
            datasets=[
                construct_dataset(os.path.join(dataset_dirpath, part_dirpath))
                for part_dirpath in metadata["part_dirpaths"]
            ]
        )
    elif metadata["dataset_type"] == "simple":
        return _SimpleDataPoints(dataset_dirpath)
    else:
        raise Exception(f"Unknown dataset_type `{metadata['dataset_type']}`!")


def concat_datasets(datasets: List[DataPoints]) -> DataPoints:
    return _ComposedDataPoints(datasets)


class _SimpleDataPoints(DataPoints):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.l = len(os.listdir(dataset_dir)) // 3

    def __str__(self) -> str:
        return f"SimpleDataPoints('{self.dataset_dir}')"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.l

    def __getitem__(self, idx: int) -> DataPoint:
        return DataPoint.deserialize(os.path.join(self.dataset_dir, str(idx)))


class _ComposedDataPoints(DataPoints):
    def __init__(self, datasets: List[DataPoints]):
        self.datasets = datasets
        self.l = sum([len(d) for d in self.datasets])

    def __str__(self) -> str:
        return f"ComposedDataPoints({','.join([str(d) for d in self.datasets])})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.l

    def __getitem__(self, idx: int) -> DataPoint:
        l = 0
        d, d_idx = None, 0
        while True:
            if d_idx == len(self.datasets):
                break
            d = self.datasets[d_idx]
            d_idx += 1
            l += len(d)
            if idx < l:
                return d.__getitem__(idx - l + len(d))
        raise Exception("Bad idx!")


class _CroppedDataPoints(DataPoints):
    def __init__(self, dataset: DataPoints, n: int, take: bool):
        self.dataset = dataset
        self.n = n
        self._take = take

    def __len__(self) -> int:
        if self._take:
            return min(self.n, len(self.dataset))
        else:
            return max(0, len(self.dataset) - self.n)

    def __getitem__(self, idx: int) -> DataPoint:
        if self._take:
            return self.dataset.__getitem__(idx)
        else:
            return self.dataset.__getitem__(idx + self.n)

    def __str__(self) -> str:
        if self._take:
            return str(self.dataset) + f"[:{self.n}]"
        else:
            return str(self.dataset) + f"[{self.n}:]"

    def __repr__(self) -> str:
        return str(self.dataset)


def generate_data(
    *,
    gripper: "GripperModule",
    aux_gripper: Union["AuxGripperModule", None] = None,
    thrower: "ThrowerModule",
    image_transformer: "ImageTransformer",
    seeds: List[int],
    length: int,
    on_cuda: bool,
    workdir_path: str,
    forces: Union[None, List[float]] = None,
) -> DataPoints:
    """
        Generuje zbiór przykładów o zadanych parametrach:
        (seeds, length) z użyciem zadanych modułów.
        Parametr on_cuda kontroluje użycie akceleracji
        przy pomocy karty graficznej.
        Zwracany jest obiekt typu Dataset 
        zawierający wygenerowane przykłady.

        Jeśli podano parametr forces, to jest on
        użyty jako lista sił chwytu do przetestowania.
    """

    if (forces is None and aux_gripper is None) or (
        forces is not None and aux_gripper is not None
    ):
        raise Exception("Pass exactly one of the parameters: aux_gripper, forces")

    cprint("Generating data...")

    gripper.save(os.path.join(workdir_path, "gripper"))
    if aux_gripper is not None:
        aux_gripper.save(os.path.join(workdir_path, "aux-gripper"))
    thrower.save(os.path.join(workdir_path, "thrower"))
    save_obj(os.path.join(workdir_path, "image_transformer"), image_transformer)

    # Dla każdego ziarna tworzymy nowy proces Python'owy.
    # Nie jest to optymalne rozwiązanie, ale za to
    # omijamy problem z tworzeniem procesów
    # potomnych po zainicjalizowaniu internalowych
    # struktur PyTorcha.
    for seed_idx, seed in enumerate(seeds):
        src_dir = os.path.dirname(__file__)
        os.system(
            f"""
cd {src_dir} ; 
python3 ./data.py \
    --seed {seed} \
    --length {length} \
    {"--on_cuda" if on_cuda else ""} \
    --gripper {gripper.name()} \
    {"--aux_gripper " + aux_gripper.name() if aux_gripper is not None else ""} \
    {"--forces '" + json.dumps(forces) + "'" if forces is not None else ""} \
    --thrower {thrower.name()} \
    --workdir_path {workdir_path} \
    --output_path {os.path.join(workdir_path, str(seed_idx))} &
        """
        )

    while True:
        time.sleep(5)
        if all(
            [
                os.path.isfile(os.path.join(workdir_path, str(seed_idx), "metadata"))
                for seed_idx in range(len(seeds))
            ]
        ):
            break

    metadata = {
        "dataset_type": "composed",
        "part_dirpaths": [str(seed_idx) for seed_idx in range(len(seeds))],
        "seeds": seeds,
        "length": length,
        "thrower": thrower.name(),
        "gripper": gripper.name(),
        "image_transformer": image_transformer.name(),
    }
    if forces:
        metadata["forces"] = forces
    else:
        metadata["aux_gripper"] = aux_gripper.name()

    with open(os.path.join(workdir_path, "metadata"), "w") as f:
        json.dump(metadata, f)

    cprint("Data generated!")

    return construct_dataset(workdir_path)


if __name__ == "__main__":
    from robot_control import (
        get_aux_gripper_by_name,
        get_thrower_by_name,
        get_gripper_by_name,
        BaselineAuxGripperModule,
        AuxGripperModule,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--on_cuda", type=bool, default=False, const=True, nargs="?")
    parser.add_argument("--gripper", type=str)
    parser.add_argument("--aux_gripper", type=str)
    parser.add_argument("--forces", type=str)
    parser.add_argument("--thrower", type=str)
    parser.add_argument("--workdir_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    if args.forces is not None:
        args.forces = json.loads(args.forces)
    cprint(args)

    os.makedirs(args.output_path, exist_ok=False)

    gripper = get_gripper_by_name(args.gripper)
    thrower = get_thrower_by_name(args.thrower)
    gripper.load(os.path.join(args.workdir_path, "gripper"))
    thrower.load(os.path.join(args.workdir_path, "thrower"))
    image_transformer = load_obj(os.path.join(args.workdir_path, "image_transformer"))

    aux_grippers: List[AuxGripperModule] = []
    if not args.forces:
        aux_gripper = get_aux_gripper_by_name(args.aux_gripper)
        aux_gripper.load(os.path.join(args.workdir_path, "aux-gripper"))
        aux_grippers.append(aux_gripper)
    else:
        for grip_force in args.forces:
            aux_grippers.append(BaselineAuxGripperModule(force=grip_force))

    gripper.evaluate()
    thrower.evaluate()
    for ag in aux_grippers:
        ag.evaluate()

    if args.on_cuda:
        gripper.cuda()
        thrower.cuda()
        for ag in aux_grippers:
            ag.cuda()

    test_cases_generator = _TestCasesGenerator(
        seed=args.seed, length=args.length, repeat=len(aux_grippers),
    )

    first_dp: Union[DataPoint, None] = None
    for sample_idx, (ctx, target) in trace_num_samples(
        enumerate(test_cases_generator), step=100
    ):
        data_point = evaluate_one_step(
            ctx=ctx,
            image_transformer=image_transformer,
            gripper=gripper,
            aux_gripper=aux_grippers[sample_idx % len(aux_grippers)],
            thrower=thrower,
            target=target,
        )
        if first_dp is None:
            first_dp = data_point
        first_dp.all_grip_results[
            float(data_point.grip_force)
        ] = data_point.grip_success
        if (sample_idx + 1) % len(aux_grippers) == 0:
            first_dp.serialize(
                os.path.join(args.output_path, str(sample_idx // len(aux_grippers)))
            )
            first_dp = None

    metadata = {
        "dataset_type": "simple",
        "seeds": [args.seed],
        "length": args.length,
        "thrower": thrower.name(),
        "gripper": gripper.name(),
        "aux_grippers": [ag.name() for ag in aux_grippers],
        "image_transformer": image_transformer.name(),
    }
    with open(os.path.join(args.output_path, "metadata"), "w") as f:
        json.dump(metadata, f)
