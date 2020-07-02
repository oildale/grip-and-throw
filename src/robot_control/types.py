"""
    Zawiera interfejsy modułów rzucających i łapiących.
    Wszystkie implementacje powinny być w pełni deterministyczne,
    aby ułatwić debugowanie.
"""

import os
import torch
import itertools
import numpy as np
from typing import Tuple


class Module:
    def __init__(self, name: str):
        self.is_cuda = False
        self._name = name

    def is_trainable(self) -> bool:
        return False

    # Zwraca słownik zawierający wszystkie PyTorch'owe
    # modele używane przez moduł.
    def models(self) -> dict:
        return {}

    def load(self, model_dir: str):
        for (model_name, model) in self.models().items():
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=False)
        for (model_name, model) in self.models().items():
            model.cpu()
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
            if self.is_cuda:
                model.cuda()

    def parameters(self):
        parameters = []
        for model in self.models().values():
            parameters.append(model.parameters())
        return itertools.chain(*parameters)

    def cpu(self):
        self.is_cuda = False
        for model in self.models().values():
            model.cpu()

    def cuda(self):
        self.is_cuda = True
        for model in self.models().values():
            model.cuda()

    def train(self):
        for p in self.parameters():
            p.requires_grad = True
        for model in self.models().values():
            model.train()

    def evaluate(self):
        for p in self.parameters():
            p.requires_grad = False
        for model in self.models().values():
            model.eval()

    def name(self) -> str:
        return self._name


class GripperModule(Module):
    def __init__(self, name: str):
        super().__init__(name=name)

    # Działa dla pojedyńczego przykładu.
    # Zwraca (grip_x, grip_y, angle).
    def choose_grip(self, img: np.ndarray) -> Tuple[float, float, float]:
        raise Exception("Unsupported operation!")

    # Powinno być zaimplementowane tylko wtedy,
    # gdy is_trainable() == True.
    # Działa dla batch'a przykładów.
    # imgs zawiera obrazki już obrócone.
    # grip_pixel_xs, grip_pixel_ys są współrzędnymi
    # pixela na obróconym obrazie, w którym mamy
    # przewidzieć prawdopodobieństwo udanego chwytu.
    def forward(
        self,
        *,
        grip_pixel_xs: torch.Tensor,
        grip_pixel_ys: torch.Tensor,
        imgs: torch.Tensor,
    ) -> torch.FloatTensor:
        raise Exception("Unsupported operation!")


class AuxGripperModule(Module):
    def __init__(self, name: str):
        super().__init__(name=name)

    def choose_grip(
        self, img: np.ndarray, grip_x: float, grip_y: float, angle: float
    ) -> float:
        raise Exception("Unsupported operation!")

    # Powinno być zaimplementowane tylko wtedy,
    # gdy is_trainable() == True.
    # Przyjmuje jeden parametr batch,
    # z którego wyciąga potrzebne pola
    # i oblicza loss na danym batchu.
    # Takie podejście jest wymuszone
    # faktem, że poszczególne implementacje
    # mogą się bardzo różnić sposobem
    # trenowania.
    def forward(self, batch: dict) -> torch.Tensor:
        raise Exception("Unsupported operation!")


class ThrowerModule(Module):
    def __init__(self, name: str):
        super().__init__(name=name)

    # Działa dla pojedyńczego przykładu.
    # Zwraca throw_vel.
    def choose_throw(self, *, target: Tuple[int, int], sensors: np.ndarray) -> float:
        raise Exception("Unsupported operation!")

    # Powinno być zaimplementowane tylko wtedy,
    # gdy is_trainable() == True.
    # Działa dla batch'a przykładów.
    # Zwraca predykcje wartości throw_vel, używane
    # podczas treningu.
    def forward(self, *, targets: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        raise Exception("Unsupported operation!")
