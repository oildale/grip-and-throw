from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Callable, Union, List, cast

from transforms import TransformingDataset
from utils import cprint


class Sampler:
    def __iter__(self):
        raise Exception("Unsupported operation!")

    def update_weight(self, *, sample_idx: int, target_prob: float, pred_prob: float):
        raise Exception("Unsupported operation!")

    def weights(self) -> Union[None, "_SamplesWeights"]:
        raise Exception("Unsupported operation!")

    def __len__(self) -> int:
        raise Exception("Unsupported operation!")


class DataLoaderWrapper(Sampler):
    def __init__(self, *, wrapped: DataLoader, weights: Union["_SamplesWeights", None]):
        self.wrapped = wrapped
        self._weights = weights

    def __iter__(self):
        return iter(self.wrapped)

    def update_weight(self, *, sample_idx: int, target_prob: float, pred_prob: float):
        if self._weights:
            self._weights.update_weight(
                sample_idx=sample_idx, target_prob=target_prob, pred_prob=pred_prob
            )

    def weights(self) -> Union[None, "_SamplesWeights"]:
        return self._weights

    def __len__(self) -> int:
        return len(self.wrapped)


class WeightedDataLoader(Sampler):
    def __init__(
        self,
        *,
        dataset: TransformingDataset,
        weights: "_SamplesWeights",
        batch_size: int,
        num_samples: int,
        num_workers: int,
    ):
        self._dataset = dataset
        self._weights = weights
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._num_workers = num_workers

    def __iter__(self):
        """
            Mało wydajna implementacja, która tworzy
            nowy DataLoader dla każdego batcha.
            W przeciwnym razie ciężko byłoby zagwarantować,
            aby kolejne batch'e były samplowane
            z zaktualizowanego rozkładu (a nie na przykład
            lekko opóźnionego).
        """
        num_samples = self._num_samples
        while num_samples > 0:
            batch_size = min(num_samples, self._batch_size)
            sampler = WeightedRandomSampler(
                weights=self._weights.get_weights(),
                num_samples=batch_size,
                replacement=True,
            )
            dataloader = DataLoader(
                dataset=self._dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=self._num_workers,
            )
            for batch in dataloader:
                num_samples -= len(batch["idx"])
                yield batch

    def update_weight(self, sample_idx, target_prob, pred_prob):
        self.weights.update_weight(sample_idx, target_prob, pred_prob)

    def weights(self) -> Union[None, "_SamplesWeights"]:
        return self._weights

    def __len__(self) -> int:
        return self._num_samples


class _SamplesWeights:
    def __init__(
        self,
        *,
        density_fn: Callable,
        num_samples: int,
        init_weights: List[float] = None,
    ):
        self.density_fn = density_fn
        if init_weights is None:
            cprint("Initializing samples weights...")
            self.weights = [density_fn(1, 0.5) for _ in range(num_samples)]
        else:
            cprint("Using obtained weights...")
            self.weights = init_weights
        if len(self.weights) < num_samples:
            diff = num_samples - len(self.weights)
            cprint(f"Extending weights by {diff} samples...")
            self.weights += [density_fn(1, 0.5) for _ in range(diff)]
        if len(self.weights) > num_samples:
            diff = len(self.weights) - num_samples
            cprint(f"Shrinking weights by {diff} samples...")
            self.weights = self.weights[:num_samples]
        cprint("Samples weights initialized!")

    def update_weight(self, *, sample_idx: int, target_prob: float, pred_prob: float):
        self.weights[sample_idx] = self.density_fn(target_prob, pred_prob)

    def get_weights(self) -> List[float]:
        return self.weights


def construct_sampler(
    *,
    sampling_name: str,
    dataset: TransformingDataset,
    batch_size: int,
    num_workers: int,
    prev_sampler: Union[Sampler, None] = None,
) -> Sampler:
    if sampling_name not in ["seq", "shuf", "w-pow"]:
        raise Exception(f"Unknown sampling name: {sampling_name}")

    num_samples = len(dataset)
    prev_weights = None if prev_sampler is None else prev_sampler.weights()
    prev_weights = None if prev_weights is None else prev_weights.get_weights()
    if sampling_name in ["seq", "shuf"] and not prev_weights:
        weights = None
    else:
        weights = _SamplesWeights(
            density_fn=lambda t, p: (t - p) ** 4,
            num_samples=num_samples,
            init_weights=prev_weights,
        )

    if sampling_name == "seq":
        return DataLoaderWrapper(
            wrapped=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            ),
            weights=weights,
        )
    if sampling_name == "shuf":
        return DataLoaderWrapper(
            wrapped=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            weights=weights,
        )

    return WeightedDataLoader(
        dataset=dataset,
        weights=cast(_SamplesWeights, weights),
        batch_size=batch_size,
        num_samples=num_samples,
        num_workers=num_workers,
    )
