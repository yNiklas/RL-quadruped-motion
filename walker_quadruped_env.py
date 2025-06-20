from quadruped_env import QuadrupedEnv
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import numpy as np
import math

class WalkerQuadrupedEnv(QuadrupedEnv):
    def __init__(self):
        super().__init__()

    def _init_spaces(self):
        self.obs_length = 4 * 2 + 8 + 3 + 3 + 2 + 8  # 4 legs * 2 joint angles + 8 joints velocities + 3d base accel + 3d base gyro + (roll, pitch) + 8 previous action
        self.action_length = 8  # 4 legs * 2 joint torques
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_length,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(self.action_length,),
            dtype=np.float32
        )

    def _get_obs(self):
        qpos = self.data.qpos[self.joint_qpos_idx]
        qvel = self.data.qvel[self.joint_qvel_idx]

        acc = self.data.sensordata[self.accel_idx:self.accel_idx + 3]
        gyro = self.data.sensordata[self.gyro_idx:self.gyro_idx + 3]
        quat = self.data.sensordata[self.framequat_idx:self.framequat_idx + 4]
        if all([i == 0 for i in quat]):
            roll, pitch = 0, 0
        else:
            roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')

        prev_action = self.last_action if self.last_action is not None else np.zeros_like(qpos)
        return np.concatenate([qpos, qvel, acc, gyro, [roll, pitch], prev_action])

    def _compute_reward(self, action):
        v_ref = 0.5  # Reference velocity in x direction
        r_vx = 3 * math.exp(-abs(v_ref - self.data.qvel[0]) ** 2)  # Velocity of the robot base in x direction

        upright_z = 0.26  # Z position of the robot base when upright
        r_z = np.exp(-((self.data.qpos[2] - upright_z) ** 2) / (
                2 * 0.04 ** 2))  # -25 * abs(self.data.qpos[2] - upright_z) # Vertical position of the robot base

        joint_angles = self.data.qpos[self.joint_qpos_idx]
        sigma_homing = 1
        r_homing_similarity = np.exp(-np.sum((joint_angles - self.homing_qpos) ** 2) / (2 * sigma_homing ** 2))

        # Action similarity
        sigma_action = 1.7
        if self.last_action is not None:
            r_action_similarity = np.exp(-np.sum((action - self.last_action) ** 2) / (2 * sigma_action ** 2))
        else:
            r_action_similarity = 0

        r_vz = -0.05 * self.data.qvel[2] ** 2  # Vertical velocity of the robot base

        quat = self.data.qpos[3:7]  # Orientation of the robot base in quaternion
        roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        r_orientation = -3 * (3 * roll ** 2 + 3 * pitch ** 2)

        self.log_info = {
            "state/body_v": self.data.qvel[0],
            "state/body_z": self.data.qpos[2],
            "state/roll": roll,
            "state/pitch": pitch,
            "reward/r_vx": r_vx,
            "reward/r_z": r_z,
            "reward/r_homing_similarity": r_homing_similarity,
            "reward/r_action_similarity": r_action_similarity,
            "reward/r_vz": r_vz,
            "reward/r_orientation": r_orientation
        }

        return r_vx + r_z + r_homing_similarity + r_action_similarity + r_vz + r_orientation

    def _is_collapsed(self, state):
        roll, pitch = (state[-10], state[-9])

        z_body = self.data.qpos[2]  # Z position of the robot base
        z_bound = z_body < 0.1 or z_body > 0.40  # Robot-height is 0.32 with full stretched legs
        roll_bound = abs(roll) > 0.6
        pitch_bound = abs(pitch) > 0.6
        collapse_reasons = {
            "collapse/z_bound": z_body,
            "collapse/roll_bound": abs(roll),
            "collapse/pitch_bound": abs(pitch)
        }
        return z_bound or roll_bound or pitch_bound, collapse_reasons