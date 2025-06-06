import math

from gymnasium.utils.seeding import np_random
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import mujoco
import numpy as np
import mujoco.viewer


class QuadrupedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.render_mode = "human"

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("model/world_unconstrained.xml")
        self.data = mujoco.MjData(self.model)
        self.data.qpos = self.model.keyframe("initial_joint_positions").qpos # Set XML-defined initial positions

        self.joint_names = ["FL_HFE", "FL_KFE", "FR_HFE", "FR_KFE", "HL_HFE", "HL_KFE", "HR_HFE", "HR_KFE"]
        self.joint_qpos_idx = np.array([self.model.joint(name).qposadr for name in self.joint_names]).flatten()
        self.joint_qvel_idx = np.array([self.model.joint(name).dofadr for name in self.joint_names]).flatten()

        self.homing_qpos = self.model.keyframe("initial_joint_positions").qpos[self.joint_qpos_idx]

        self.accel_idx = self.model.sensor("accelerometer").adr[0]
        self.gyro_idx = self.model.sensor("gyroscope").adr[0]
        self.framequat_idx = self.model.sensor("orientation_sensor").adr[0]

        # Init observation and action spaces
        obs_length = 4*2 + 8 + 3 + 3 + 2 + 8 # 4 legs * 2 joint angles + 8 joints velocities + 3d base accel + 3d base gyro + (roll, pitch) + 8 previous action
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_length,), dtype=np.float32)
        acs_length = 4*2 # 4 legs * 2 joint angles
        self.action_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(acs_length,), dtype=np.float32)

        self.sim_steps_per_action = 5
        self.viewer = None
        self.last_action = None
        self.log_info = None

        self.sim_robot_id = self.model.body("sim_robot").id

    def step(self, action):
        self.data.ctrl[:] = self.homing_qpos + action # Learn relative joint angles from the homing position
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(action)
        roll, pitch = (obs[-10], obs[-9])
        terminated, reasons = self._is_collapsed(roll, pitch)
        truncated = False
        info = {
            **self.log_info
        }
        if terminated:
            info.update(reasons)
        self.last_action = action
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = self.model.keyframe("initial_joint_positions").qpos # Set XML-defined initial positions
        for qpos_i in self.joint_qpos_idx:
            self.data.qpos[qpos_i] += self.np_random.uniform(-0.1, 0.1, 1)
        for qvel_i in self.joint_qvel_idx:
            self.data.qvel[qvel_i] = self.np_random.uniform(-0.1, 0.1, 1)
        info = {}
        return self._get_obs(), info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

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
        v_ref = 1.5 # Reference velocity in x direction
        r_vx = 6 * math.exp(-abs(v_ref-self.data.qvel[0])**2) # Velocity of the robot base in x direction

        upright_z = 0.26 # Z position of the robot base when upright
        r_z = -10 * abs(self.data.qpos[2] - upright_z) # Vertical position of the robot base

        joint_angles = self.data.qpos[self.joint_qpos_idx]
        r_homing_similarity = -0.2 * np.sum(np.abs(joint_angles - self.homing_qpos)**2) # Similarity to the homing position

        r_action_similarity = -0.005 * np.sum(np.abs(action - self.last_action)**2) if self.last_action is not None else 0

        r_vz = -0.1 * self.data.qvel[2]**2 # Vertical velocity of the robot base

        quat = self.data.qpos[3:7] # Orientation of the robot base in quaternion
        roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        r_orientation = -(roll**2 + 3 * pitch**2)

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

        return r_vx + r_z + r_homing_similarity + r_action_similarity + r_vz + r_orientation + 1

    def _is_collapsed(self, roll, pitch):
        z_body = self.data.qpos[2] # Z position of the robot base
        z_bound = z_body < 0.1 or z_body > 0.40 # Robot-height is 0.32 with full stretched legs
        roll_bound = abs(roll) > 0.785  # 45°
        pitch_bound = abs(pitch) > 0.785  # 45°
        collapse_reasons = {
            "collapse/z_bound": z_body,
            "collapse/roll_bound": abs(roll),
            "collapse/pitch_bound": abs(pitch)
        }
        return z_bound or roll_bound or pitch_bound, collapse_reasons
