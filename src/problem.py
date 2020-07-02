from typing import Tuple, List, Union
import pyrobolearn as prl
import pybullet
import os
import numpy as np

import consts


class Ctx:
    def __init__(
        self, *, sim, world, robot, render, objects, tray_id, targets_box_id,
    ):
        self.sim = sim
        self.world = world
        self.robot = robot
        self.render = render
        self.objects = objects
        self.tray_id = tray_id
        self.targets_box_id = targets_box_id
        self.sleep_dt = 1 / 240.0 if render else 0.0

    def step(self, sleep_dt=None):
        sleep_dt = self.sleep_dt if sleep_dt is None else sleep_dt
        self.world.step(sleep_dt=sleep_dt)


def init_problem(render: bool = False) -> Ctx:
    sim = prl.simulators.Bullet(render=render)
    world = prl.worlds.BasicWorld(sim)

    static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../static"))

    r = prl.robots.robot.Robot(
        sim,
        static_dir + "/kuka_with_gripper2.sdf",
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        fixed_base=True,
        scale=1.0,
    )

    r.enable_joint_force_torque_sensor(joint_ids=[10, 13, 8, 11])

    tray_id = world.load_urdf(
        static_dir + "/traybox.urdf",
        [consts.TRAY_OFFSET, 0.0, 0.0],
        scale=0.7,
        fixed_base=True,
    )

    targets_box_id = world.load_urdf(
        static_dir + "/targets_box.urdf",
        [consts.TARGET_OFFSET, 0, 0],
        scale=1.5,
        fixed_base=True,
    )

    ctx = Ctx(
        sim=sim,
        world=world,
        robot=r,
        render=render,
        objects=[],
        tray_id=tray_id,
        targets_box_id=targets_box_id,
    )

    return ctx


def spawn_objects(ctx: Ctx, objs):
    static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../static"))

    ctx.objects = []
    for obj in objs:
        if len(obj) == 6:
            x, y, r, p, y2, t = obj
            orientation = prl.utils.transformation.get_quaternion_from_rpy((r, p, y2))
        else:
            x, y, t = obj
            orientation = (0, 0, 0, 1)

        obj_id = ctx.world.load_urdf(
            filename=f"{static_dir}/{t}.urdf",
            position=_snapshot_to_world_coords((x, y)) + [0.15],
            orientation=orientation,
            scale=1.0,
        )
        ctx.objects.append({"id": obj_id, "type": t})

        for _ in range(25):
            ctx.step()

    for _ in range(200):
        ctx.step()


def _rep(x, n):
    return [x for _ in range(n)]


def has_objs_on_tray(ctx):
    for obj in ctx.objects:
        obj_id = obj["id"]
        try:
            pos = ctx.world.get_body_position(obj_id)
            if np.abs(pos[:2] - np.array([consts.TRAY_OFFSET, 0])).max() <= 0.21:
                return True
        except:
            pass
    return False


def _get_target_position(x: int, y: int) -> np.ndarray:
    box_center = np.array([consts.TARGET_OFFSET, 0])
    scale = 1.5
    box_radius = 0.25 * scale
    border_width = 0.005 * scale
    small_box_side = (2 * box_radius - 5 * border_width) / 4.0
    orig = box_center + (1.5 * (border_width + small_box_side)) * np.ones(2)
    return orig + (-1.0 * (border_width + small_box_side)) * np.array([x, y])


def get_obj_box(ctx: Ctx, obj_id) -> Union[None, Tuple[int, int]]:
    try:
        scale = 1.5
        box_radius = 0.25 * scale
        border_width = 0.005 * scale
        small_box_side = (2 * box_radius - 5 * border_width) / 4.0
        pos = ctx.world.get_body_position(obj_id)
        for x in range(4):
            for y in range(4):
                if np.abs(np.array(pos[:2]) - _get_target_position(x, y)).max() <= (
                    0.5 * small_box_side
                ):
                    if pos[2] <= 0.14:
                        return (x, y)
    except:
        pass
    return None


def get_picked_objs(ctx: Ctx) -> List:
    res = []
    for obj in ctx.objects:
        obj_id = obj["id"]
        try:
            pos = ctx.world.get_body_position(obj_id)
            if pos[2] > 0.25:
                contacts = ctx.sim.get_contact_points(body1=obj_id, body2=ctx.robot.id)
                if len(contacts) > 0:
                    res.append(obj)
        except:
            pass
    return res


def get_sensors(ctx: Ctx) -> np.ndarray:
    return np.concatenate(
        [
            ctx.robot.get_joint_reaction_forces(joint_ids=[10, 13, 8, 11]),
            ctx.robot.get_joint_torques(joint_ids=[10, 13, 8, 11]).reshape(-1, 1),
        ],
        axis=1,
    )


def reset_robot_position(ctx: Ctx):
    ctx.robot.set_joint_positions([0 for _ in range(12)])
    for _ in range(300):
        ctx.step()


def remove_obj(ctx: Ctx, obj_id):
    try:
        ctx.world.remove(obj_id)
    except:
        pass


def remove_objs(ctx: Ctx):
    for obj in ctx.objects:
        obj_id = obj["id"]
        remove_obj(ctx, obj_id)
    ctx.objects = []


def _pregrip(*, ctx: Ctx, grip_angle: float, grip_pos: Tuple[float, float, float]):
    # Poprawka aby uwzględnić niesymetrię chwytaka
    grip_pos_x = grip_pos[0] - 0.024 * np.sin(grip_angle)
    grip_pos_y = grip_pos[1] + 0.024 * np.cos(grip_angle)
    grip_pos_z = grip_pos[2]
    grip_pos = (grip_pos_x, grip_pos_y, grip_pos_z)

    link_id = ctx.robot.get_link_ids("base_link")
    init_h = ctx.robot.get_link_world_positions(link_id)[2]

    for grip_angle in [grip_angle]:
        for i in range(200):
            if (i % 10 == 0) and (i < 150):
                pos = [
                    grip_pos[0],
                    grip_pos[1],
                    (init_h * (140 - i) + grip_pos[2] * (i + 10)) / 150.0,
                ]
                if i == 140:
                    orient = prl.utils.transformation.get_quaternion_from_rpy(
                        [-np.pi, 0, grip_angle]
                    )
                    joint_positions = ctx.robot.calculate_inverse_kinematics(
                        link_id, position=pos, orientation=orient, max_iters=300
                    )
                else:
                    joint_positions = ctx.robot.calculate_inverse_kinematics(
                        link_id, position=pos, max_iters=300
                    )
                    joint_positions[-5] = np.pi - grip_angle
                ctx.robot.set_joint_positions(joint_positions)

            ctx.step()


def _grip(*, ctx: Ctx, force: float):
    ctx.robot.set_joint_positions(
        [0, 0, 0, 0], joint_ids=[10, 13, 8, 11], forces=_rep(force, 4)
    )
    for _ in range(200):
        ctx.step()


def _prethrow(*, ctx: Ctx, target: Tuple[int, int]):
    joint_positions = np.array(
        [
            [
                [0.15, 0.05, -0.05, -0.15],
                [0.17, 0.06, -0.06, -0.17],
                [0.18, 0.06, -0.06, -0.18],
                [0.21, 0.08, -0.08, -0.21],
            ][target[0]][target[1]]
        ]
        + [0.53, 0, -1.28, 0, 1.32, 0, 1.54, -0.007, -0.05, 0.007, 0.05]
    )

    ctx.robot.set_joint_positions(
        joint_positions,
        kp=(_rep(0.01, 12)),
        forces=(_rep(200, 4) + _rep(50, 4) + _rep(15, 4)),
    )

    for _ in range(600):
        ctx.step()


def grip(
    *,
    ctx: Ctx,
    grip_x: float,
    grip_y: float,
    img: np.ndarray,
    grip_angle: float,
    grip_force: float,
    target: Tuple[int, int],
):
    # Wyznaczamy pozycję chwytu na podstawie danego piksela.
    grip_z = max(0.25, 0.19 + img[int(grip_x)][int(grip_y)][3])
    grip_pos = _snapshot_to_world_coords((grip_x, grip_y))
    grip_pos = (grip_pos[0], grip_pos[1], grip_z)

    ctx.robot.set_joint_positions([0, 0, 0, -0.3, 0.3], joint_ids=[7, 10, 13, 8, 11])
    for _ in range(100):
        ctx.step()

    _pregrip(ctx=ctx, grip_angle=grip_angle, grip_pos=grip_pos)
    _grip(ctx=ctx, force=grip_force)
    _prethrow(ctx=ctx, target=target)


def drop(ctx: Ctx):
    ctx.robot.set_joint_positions([0, 0, -0.3, 0.3], joint_ids=[10, 13, 8, 11])
    for _ in range(100):
        ctx.step()


def throw(*, ctx: Ctx, vel: float):
    robot, sim = ctx.robot, ctx.sim

    # Przeguby 8, 10, 11, 13 to przeguby palców (po 2 na każdą stronę).
    jp = sim.get_joint_positions(robot.id, joint_ids=[8, 10, 11, 13])
    target = -0.8

    for _ in range(350):
        # Robimy zamach poprzez wprowadzenie w ruch pojedyńczego przegubu.
        # Jest to mniej precyzyjna metoda, gdyż nie możemy w ten sposób
        # zagwarantować ustalonej pozycji wypuszczenia przedmiotu.
        # Gdybyśmy chcieli osiągnąć konkretną pozycję powinniśmy użyć IK.
        # Niestety robot wtedy bardziej "wije się" w kierunku
        # danego punktu co skutkuje nieprzewidywalną trajektorią
        # i prędkością ramienia. Należy zwrócić uwagę, że sterowanie
        # NIE polega na zadaniu prędkości z jaką przedmiot zostanie
        # wypuszczony, a jedynie na zadaniu prędkości z jaką
        # ma zostać wykonany ruch. Powinno jednak być tak, że
        # obie te wartości są związane pewną monotoniczną funkcję,
        # którą model będzie w stanie się nauczyć.
        pybullet.setJointMotorControl2(
            bodyUniqueId=robot.id,
            jointIndex=3,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=target,
            targetVelocity=vel,
            force=200.0,
            maxVelocity=vel,
            positionGain=0.05,
            velocityGain=1,
        )

        # Iterujemy się po przegubach w palcach
        for (idx, (mult, i)) in enumerate([[-1, 8], [-1, 10], [1, 11], [1, 13]]):
            # Każemy palcom naciskać na przedmiot aby ten nie wypadł
            new_jp = jp[idx]
            new_jp = new_jp + mult * (-0.05)

            # Jeśli jesteśmy blisko celu, to zaczynamy wypuszczać przedmiot
            if sim.get_joint_positions(robot.id, joint_ids=[3])[0] > (target - 0.1):
                new_jp = mult * 0.3

            pybullet.setJointMotorControl2(
                robot.id, i, pybullet.POSITION_CONTROL, targetPosition=new_jp, force=2.0
            )

        ctx.step()


def _get_view_matrix(ctx: Ctx):
    return ctx.sim.compute_view_matrix(
        eye_position=[consts.TRAY_OFFSET, 0.0, 10],
        target_position=[consts.TRAY_OFFSET, 0.0, 0.0],
        up_vector=[0.0, -1.0, 0.0],
    )


def _get_projection_matrix(ctx: Ctx):
    s = 0.165
    return ctx.sim.compute_projection_matrix(
        left=-s, right=s, bottom=-s, top=s, near=8, far=12
    )


# Zwraca obraz RGB-D tacy. Kanał D jest
# tak przetransformowany, aby zawierał odległość
# obserwowanego punktu od powierzchni.
def get_camera_snapshot(ctx: Ctx) -> np.ndarray:
    sim = ctx.sim
    _, _, rgb_img, dep_img, _ = sim.get_camera_image(
        width=consts.WIDTH,
        height=consts.WIDTH,
        view_matrix=_get_view_matrix(ctx),
        projection_matrix=_get_projection_matrix(ctx),
    )
    rgb_img = rgb_img[:, :, :3]
    dep_img = dep_img.reshape(consts.WIDTH, consts.WIDTH, 1)

    dep_img = ((dep_img - 0.5974774) / (0.5853995 - 0.5974774)) * 0.05 + 0.02

    img = np.concatenate([rgb_img, dep_img], axis=2)

    return img


def _snapshot_to_world_coords(pixel: Tuple[float, float]) -> List[float]:
    hw = consts.WIDTH / 2.0
    x = (pixel[0] - hw) / hw
    y = (pixel[1] - hw) / hw
    x *= 0.21
    y *= 0.21
    return [consts.TRAY_OFFSET - y, x]


def deinit_problem(ctx: Ctx):
    ctx.sim.close()


if __name__ == "__main__":
    ctx = init_problem(render=True)
    spawn_objects(ctx, [[75, 75, np.pi, np.pi, np.pi / 8.0, "cube"]])
    grip(
        ctx=ctx,
        grip_x=75,
        grip_y=78,
        img=get_camera_snapshot(ctx),
        grip_angle=np.pi / 2.0,
        grip_force=4,
        target=(0, 0),
    )
    throw(ctx=ctx, vel=6)
    reset_robot_position(ctx)
    remove_objs(ctx)
