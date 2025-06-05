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
        self.joint_qpos_idx = [self.model.joint(name).qposadr for name in self.joint_names]
        self.joint_qvel_idx = [self.model.joint(name).dofadr for name in self.joint_names]

        self.accel_idx = self.model.sensor("accelerometer").adr[0]
        self.gyro_idx = self.model.sensor("gyroscope").adr[0]
        self.framequat_idx = self.model.sensor("orientation_sensor").adr[0]

        # Init observation and action spaces
        obs_length = 4*2 + 8 + 3 + 3 + 4 # 4 legs * 2 joint angles + 8 joints velocities + 3d base accel + 3d base gyro + 4d base orientation
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_length,), dtype=np.float32)
        acs_length = 4*2 # 4 legs * 2 joint angles
        self.action_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(acs_length,), dtype=np.float32)

        self.sim_steps_per_action = 5
        self.viewer = None

        self.sim_robot_id = self.model.body("sim_robot").id

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_collapsed()
        truncated = False
        info = {
            "z_body": self.data.qpos[2],
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = self.model.keyframe("initial_joint_positions").qpos # Set XML-defined initial positions
        info = {}
        return self._get_obs(), info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def _get_obs(self):
        qpos = self.data.qpos[self.joint_qpos_idx].flatten()
        qvel = self.data.qvel[self.joint_qvel_idx].flatten()

        acc = self.data.sensordata[self.accel_idx:self.accel_idx + 3]
        gyro = self.data.sensordata[self.gyro_idx:self.gyro_idx + 3]
        quat = self.data.sensordata[self.framequat_idx:self.framequat_idx + 4]

        return np.concatenate([qpos, qvel, acc, gyro, quat])

    def _compute_reward(self):
        vx = self.data.qvel[0] # Velocity of the robot base in x direction
        target_vx = 0.3
        r_vx = -abs(vx - target_vx)

        quat = self.data.qpos[3:7] # Orientation of the robot base in quaternion
        roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        r_orientation = - (roll**2 + pitch**2)

        r_vz = -abs(self.data.qvel[2]) # Vertical velocity of the robot base

        joint_limit_penalty = -np.sum(np.maximum(np.abs(self.data.qpos[self.joint_qpos_idx].flatten()) -1.5, 0))

        return 1.0 * r_vx + 0.75 * r_orientation + 0.75 * r_vz + 10*joint_limit_penalty

    def _is_collapsed(self):
        z_body = self.data.qpos[2] # Z position of the robot base
        return z_body < 0.1 or z_body > 0.40 # Robot-height is 0.32 with full stretched legs
