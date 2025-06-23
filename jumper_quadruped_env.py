import math
from scipy.spatial.transform import Rotation as R
from quadruped_env import QuadrupedEnv
import numpy as np
import gymnasium as gym

class JumperQuadrupedEnv(QuadrupedEnv):
    def __init__(self):
        super().__init__()

    def _init_spaces(self):
        self.obs_length = 4 * 2 + 8 + 3 + 3 + 4  # 4 legs * 2 joint angles + 8 joints velocities + 3d base lin velocity + 3d base angular velocity + 4d base orientation (quaternion
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
        joint_angles = self.data.qpos[self.joint_qpos_idx]
        joint_vel = self.data.qvel[self.joint_qvel_idx]
        lin_vel = self.data.qvel[0:3]  # Linear velocity of the robot base
        ang_vel = self.data.qvel[3:6] # Angular velocity of the robot base
        orient_quat = self.data.sensordata[self.framequat_idx:self.framequat_idx + 4]  # Orientation of the robot base in quaternion
        return np.concatenate([joint_angles, joint_vel, lin_vel, ang_vel, orient_quat])

    def _compute_reward(self, action):
        z_ref = 0.5 # Reference height for the robot base (standing robot is ~0.3)
        r_z = 4*max(0.0, self.data.qpos[2] - 0.28) / 0.2 #r_z = math.exp(-abs(z_ref - self.data.qpos[2]) ** 2 / (2 * 0.04 ** 2))

        quat = self.data.qpos[3:7]
        roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        r_orient = math.exp(-(abs(roll) + abs(pitch)) / 0.2)  # Penalize roll and pitch angles

        r_effort = -0.001 * np.sum(np.square(action))  # Penalize large actions

        self.log_info = {
            "state/body_z": self.data.qpos[2],
            "state/roll": roll,
            "state/pitch": pitch,
            "reward/r_z": r_z,
            "reward/r_orientation": r_orient
        }

        return r_z + 0.3 * r_orient + r_effort

    def _is_collapsed(self, state):
        roll, pitch = (state[-10], state[-9])
        z_body = self.data.qpos[2]
        z_bound = z_body < 0.1
        roll_bound = abs(roll) > 0.6
        pitch_bound = abs(pitch) > 0.8
        collapse_reasons = {
            "collapse/z": z_body,
            "collapse/roll": abs(roll),
            "collapse/pitch": abs(pitch)
        }
        return z_bound or roll_bound or pitch_bound, collapse_reasons
