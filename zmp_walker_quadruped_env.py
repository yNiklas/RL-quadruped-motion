import mujoco
from scipy.spatial import ConvexHull

from quadruped_env import QuadrupedEnv
from scipy.spatial.transform import Rotation as R
from matplotlib.path import Path

import gymnasium as gym
import numpy as np
import math

class ZMPWalkerQuadrupedEnv(QuadrupedEnv):
    def __init__(self):
        super().__init__()
        self.foot_geom_ids = [self.model.geom(name).id for name in ["FL_FOOT_GEOM", "FR_FOOT_GEOM", "HL_FOOT_GEOM", "HR_FOOT_GEOM"]]

    def _init_spaces(self):
        self.obs_length = 4 * 2 + 8 + 3 + 2 + 4 + 8  # 4 legs * 2 joint angles + 8 joints velocities + 3d base gyro + (roll, pitch) + 4 foot contacts + 8 previous action
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

        gyro = self.data.sensordata[self.gyro_idx:self.gyro_idx + 3]
        quat = self.data.sensordata[self.framequat_idx:self.framequat_idx + 4]
        if all([i == 0 for i in quat]):
            roll, pitch = 0, 0
        else:
            roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')

        foot_contacts = np.array([
            self.data.sensordata[self.fl_foot_contact_idx],
            self.data.sensordata[self.fr_foot_contact_idx],
            self.data.sensordata[self.hl_foot_contact_idx],
            self.data.sensordata[self.hr_foot_contact_idx]
        ])
        foot_contacts = np.clip(foot_contacts/100, -1, 1)

        prev_action = self.last_action if self.last_action is not None else np.zeros_like(qpos)
        return np.concatenate([qpos, qvel, gyro, foot_contacts, [roll, pitch], prev_action])

    def _compute_reward(self, action):
        #v_ref = 0.5  # Reference velocity in x direction
        #r_vx = 0 * math.exp(-(v_ref - self.data.qvel[0]) ** 2 / (2*0.3**2))  # Velocity of the robot base in x direction
        #p_vx = -4*abs(v_ref-self.data.qvel[0])
        r_vx = max(0, 3*min(self.data.qvel[0], 0.5))

        joint_angles = self.data.qpos[self.joint_qpos_idx]
        sigma_homing = 1
        r_homing_similarity = 0.75*np.exp(-np.sum((joint_angles - self.homing_qpos) ** 2) / (2 * sigma_homing ** 2))

        # Action similarity
        sigma_action = 1
        if self.last_action is not None:
            r_action_similarity = 0.75*np.exp(-np.sum((action - self.last_action) ** 2) / (2 * sigma_action ** 2))
        else:
            r_action_similarity = 0

        quat = self.data.qpos[3:7]  # Orientation of the robot base in quaternion
        roll, pitch, _ = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        r_orientation = 0#5*(roll**2 + pitch**2)

        cur_z = self.data.qpos[2]
        desired_z = 0.26
        r_upright = 0.8*math.exp(-(desired_z - cur_z) ** 2 / (2*0.04**2))

        r_zmp = self._calculate_zmp_reward()

        self.log_info = {
            "state/body_v": self.data.qvel[0],
            "state/body_z": self.data.qpos[2],
            "state/roll": roll,
            "state/pitch": pitch,
            "reward/r_vx": r_vx,
            "reward/r_homing_similarity": r_homing_similarity,
            "reward/r_action_similarity": r_action_similarity,
            "reward/r_zmp": r_zmp,
            "reward/r_orientation": r_orientation,
            "reward/r_upright": r_upright
        }

        r_x = 1*self.data.qpos[0] # Reward for achieved walking distance in x direction

        return r_vx + r_zmp + r_homing_similarity + r_action_similarity + r_x + r_orientation + r_upright

    def _is_collapsed(self, state):
        roll, pitch = (state[-10], state[-9])

        z_body = self.data.qpos[2]  # Z position of the robot base
        z_bound = z_body < 0.12 or z_body > 0.40  # Robot-height is 0.32 with full stretched legs
        roll_bound = abs(roll) > 0.3
        pitch_bound = abs(pitch) > 0.4
        collapse_reasons = {
            "collapse/z_bound": z_body,
            "collapse/roll_bound": abs(roll),
            "collapse/pitch_bound": abs(pitch)
        }
        return z_bound or roll_bound or pitch_bound, collapse_reasons

    def _calculate_zmp_reward(self):
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        foot_contact_pos_2d = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Only foot contacts
            if contact.geom1 in self.foot_geom_ids or contact.geom2 in self.foot_geom_ids:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                normal_force = force[:3]  # Extract normal forces (first three components)
                p = contact.pos
                foot_contact_pos_2d.append(p.copy()[:2])
                r = p - self.data.subtree_com[0]
                total_force += normal_force
                total_moment += np.cross(r, normal_force)

        fcs = np.array(foot_contact_pos_2d)
        zmp = self._calculate_zmp(total_force, total_moment)
        if zmp is None:
            # Reward foot contact
            return len(foot_contact_pos_2d)-1
        if len(foot_contact_pos_2d) >= 3:
            hull = ConvexHull(foot_contact_pos_2d)
            path = Path(fcs[hull.vertices])
            zmp_in_sp = path.contains_point(zmp)
            return 10.0 if zmp_in_sp else -1.0
        elif len(foot_contact_pos_2d) == 2:
            distance_to_line = np.abs(np.cross(fcs[1] - fcs[0], fcs[0] - zmp)) / np.linalg.norm(fcs[1] - fcs[0])
            # Only a small difference is okay, since the robot is small
            return np.clip(1 - 20*distance_to_line, -1.0, 1.0) + 2 #reward for having 2 feet in contact
        else:
            # Penalize if less than 2 feet are in contact
            return -1


    def _calculate_zmp(self, total_force, total_moment):
        if abs(total_force[2]) >= 1e-5:
            zmp_x = -total_moment[1] / total_force[2] + self.data.subtree_com[0][0]
            zmp_y = total_moment[0] / total_force[2] + self.data.subtree_com[0][1]
            return (zmp_x, zmp_y)
        else:
            return None
