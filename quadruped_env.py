from abc import abstractmethod

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
        self.fl_foot_contact_idx = self.model.sensor("FL_contact_sensor").adr[0]
        self.fr_foot_contact_idx = self.model.sensor("FR_contact_sensor").adr[0]
        self.hl_foot_contact_idx = self.model.sensor("HL_contact_sensor").adr[0]
        self.hr_foot_contact_idx = self.model.sensor("HR_contact_sensor").adr[0]

        # Init observation and action spaces
        self._init_spaces()

        self.sim_steps_per_action = 5
        self.viewer = None
        self.last_action = None
        self.log_info = None

        self.sim_robot_id = self.model.body("sim_robot").id

    @abstractmethod
    def _init_spaces(self):
        pass

    def step(self, action):
        self.data.ctrl[:] = self.homing_qpos + action # Learn relative joint angles from the homing position
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, reasons = self._is_collapsed(obs)
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

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _compute_reward(self, action):
        pass

    @abstractmethod
    def _is_collapsed(self, state):
        pass
