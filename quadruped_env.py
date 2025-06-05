from typing import Tuple

import gym
import mujoco
import numpy as np
from gym.core import ActType, ObsType


class QuadrupedEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("model/world_unconstrained.xml")
        self.data = mujoco.MjData(self.model)

        self.joint_names = ["FL_HFE", "FL_KFE", "FR_HFE", "FR_KFE", "HL_HFE", "HL_KFE", "HR_HFE", "HR_KFE"]
        self.joint_qpos_idx = [self.model.joint(name).qposadr for name in self.joint_names]
        self.joint_qvel_idx = [self.model.joint(name).dofadr for name in self.joint_names]

        self.accel_idx = self.model.sensor("accelerometer").adr
        self.gyro_idx = self.model.sensor("gyroscope").adr
        self.framequat_idx = self.model.sensor("orientation_sensor").adr

        # Init observation and action spaces
        obs_length = 4*2 + 8*3 + 3 + 3 + 4 # 4 legs * 2 joint angles + 8 joints * 3d velocities + 3d base accel + 3d base gyro + 4d base orientation
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_length,), dtype=np.float32)
        acs_length = 4*2 # 4 legs * 2 joint angles
        self.action_space = gym.spaces.Box(low=-3.14, high=3.14, shape=(acs_length,), dtype=np.float32)

        self.sim_steps_per_action = 5

        self.robot_base_body_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    def _get_obs(self):
        qpos = self.data.qpos[self.joint_qpos_idx]
        qvel = self.data.qvel[self.joint_qvel_idx]

        acc = self.data.sensordata[self.accel_idx:self.accel_idx + 3]
        gyro = self.data.sensordata[self.gyro_idx:self.gyro_idx + 3]
        quat = self.data.sensordata[self.framequat_idx:self.framequat_idx + 4]

        return np.concatenate([qpos, qvel, acc, gyro, quat])

    def _compute_reward(self):
        vx = self.data.qvel[self.robot_base_body_idx][0] # Velocity of the robot base in x direction
        target_vx = 0.3
        r_vx = -abs(vx - target_vx)

        com = self.data.subtree_com[self.robot_base_body_idx]
        zmp_ref = np.array([0.0, 0.0])
        r_zmp = -np.linalg.norm(com[:2] - zmp_ref)

        return 1.0 * r_vx + 0.1 * r_zmp
