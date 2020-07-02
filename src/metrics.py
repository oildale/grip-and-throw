from utils import trace_num_samples
from data import DataPoints


def calculate_metrics(dataset: DataPoints) -> dict:
    """
        Oblicza metryki dla danego zbioru przykładów.
    """

    metrics = {
        "num_samples": 0.0,
        "succ_grip": 0.0,
        "succ_throw": 0.0,
        "multi_grip": 0.0,
        "out_of_tray": 0.0,
        "per_obj": {},
    }

    for idx in trace_num_samples(range(len(dataset)), step=1000):
        dp = dataset[idx]

        metrics["num_samples"] += 1
        metrics["succ_grip"] += dp.grip_success
        metrics["succ_throw"] += dp.throw_success
        metrics["multi_grip"] += dp.multi_grip
        metrics["out_of_tray"] += dp.out_of_tray

        if dp.grip_success:
            assert dp.obj_type
            if dp.obj_type not in metrics["per_obj"]:
                metrics["per_obj"][dp.obj_type] = {"num": 0, "succ_throw": 0}
            metrics["per_obj"][dp.obj_type]["num"] += 1
            if dp.throw_success:
                metrics["per_obj"][dp.obj_type]["succ_throw"] += 1

    metrics["succ_throw"] /= max(metrics["succ_grip"], 1)
    metrics["succ_grip"] /= metrics["num_samples"]
    metrics["multi_grip"] /= metrics["num_samples"]
    metrics["out_of_tray"] /= metrics["num_samples"]

    return metrics
