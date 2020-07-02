from robot_control.baseline import (
    BaselineAuxGripperModule,
    BaselineGripperModule,
    BaselineThrowerModule,
    MissingGripperModule,
    RandomAuxGripperModule,
)
from robot_control.grippers import ConvGripperModule
from robot_control.aux_grippers import (
    Disc1AuxGripperModule,
    Disc2AuxGripperModule,
    ContAuxGripperModule,
)
from robot_control.throwers import (
    FF1ThrowerModule,
    FF2ThrowerModule,
    FF3ThrowerModule,
    FF4ThrowerModule,
    FF5ThrowerModule,
    SimpleEmbeddingThrowerModule,
)
from robot_control.types import GripperModule, AuxGripperModule, ThrowerModule


def get_gripper_by_name(name: str) -> GripperModule:
    model = None
    if name == "miss":
        model = MissingGripperModule()
    if name == "base":
        model = BaselineGripperModule()
    if name == "conv":
        model = ConvGripperModule()
    if model is None or model.name() != name:
        raise Exception(f"Unknown gripper `{name}`!")
    return model


def get_aux_gripper_by_name(name: str, **kwargs) -> AuxGripperModule:
    model = None
    if name == "base":
        model = BaselineAuxGripperModule(**kwargs)
    if name == "rand":
        model = RandomAuxGripperModule()
    if name == "disc1":
        model = Disc1AuxGripperModule()
    if name == "cont":
        model = ContAuxGripperModule()
    if name == "disc2":
        model = Disc2AuxGripperModule()
    if model is None or model.name() != name:
        raise Exception(f"Unknown aux_gripper `{name}`!")
    return model


def get_thrower_by_name(name: str) -> ThrowerModule:
    model = None
    if name == "base":
        model = BaselineThrowerModule()
    if name == "emb":
        model = SimpleEmbeddingThrowerModule()
    if name == "ff1":
        model = FF1ThrowerModule()
    if name == "ff2":
        model = FF2ThrowerModule()
    if name == "ff3":
        model = FF3ThrowerModule()
    if name == "ff4":
        model = FF4ThrowerModule()
    if name == "ff5":
        model = FF5ThrowerModule()
    if model is None or model.name() != name:
        raise Exception(f"Unknown thrower `{name}`!")
    return model
