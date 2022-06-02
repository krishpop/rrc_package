"""Example policies, demonstrating how to control the robot."""
import collections
import logging
import os.path as osp

import numpy as np
import pybullet as p
import trifinger_simulation.finger_types_data

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None
    print("ROS not imported, testing mode")
from scipy.spatial.transform import Rotation

from rrc_example_package import pinocchio_utils, utils
from rrc_example_package.quaternions import Quaternion


class PDControlPolicy:
    """PD control policy which just points at the goal positions with one finger."""

    position_gains = np.array([15.0, 15.0, 9.0] * 3)
    velocity_gains = np.array([0.5, 1.0, 0.5] * 3)

    def __init__(self, action_space, trajectory):
        self.action_space = action_space
        self.trajectory = trajectory
        if get_package_share_directory is not None:
            robot_properties_path = get_package_share_directory(
                "robot_properties_fingers"
            )
        else:
            robot_properties_path = osp.join(
                osp.split(osp.abspath(trifinger_simulation.__file__))[0],
                "robot_properties_fingers",
            )
        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
            "trifingerpro"
        )
        finger_urdf_path = osp.join(robot_properties_path, "urdf", urdf_file)
        self.kinematics = pinocchio_utils.Kinematics(
            finger_urdf_path,
            [
                "finger_tip_link_0",
                "finger_tip_link_120",
                "finger_tip_link_240",
            ],
        )

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array([0, 1.5, -2.7] * 3)
        self.joint_torques = np.zeros(9)
        self.dt = 0.001
        self._tip_jacobians = []
        self._prev_obs = collections.deque([], 5)

    def clip_to_space(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def compute_grasp_points_normals(self, observation, overwrite_z=True):
        curr_pose = {
            "position": np.mean(self.prev_obj_position, axis=0),
            "orientation": Rotation.from_euler(
                "xyz", self.prev_obj_orientation.mean(axis=0)
            ).as_quat(),
        }
        grasp_points = np.asarray(utils.closest_face_centers(curr_pose)).reshape(3, 3)
        if overwrite_z:
            grasp_points[:, 2] = utils.CUBE_HALF_SIZE
        grasp_normals = grasp_points - observation["object_observation"]["position"]
        grasp_normals[:, 2] *= 0.0
        grasp_normals = grasp_normals / np.linalg.norm(grasp_normals, axis=1)
        return grasp_points, grasp_normals

    def compute_tip_jacobians(self, q0):
        jacobians = []
        for fid in self.kinematics.tip_link_ids:
            Ji = self.kinematics.compute_jacobian(fid, q0)
            jacobians.append(Ji)
        return jacobians

    def compute_tip_velocity(self, observation):
        dq = observation["robot_observation"]["velocity"]
        dx = []
        for Ji in self.tip_jacobians:
            dx.append((Ji[:3, :] @ np.expand_dims(dq, -1)).squeeze())
        dx = np.concatenate(dx)  # dx.shape: (9,)
        return dx

    def compute_object_velocity(self, window_shift=4):
        if len(self._prev_obs) > window_shift:
            dx = np.mean(
                self.prev_obj_position[-window_shift:]
                - self.prev_obj_position[-window_shift - 1 : -1],
                axis=0,
            )
            dtheta = np.mean(
                self.prev_obj_orientation[-window_shift:]
                - self.prev_obj_orientation[-window_shift - 1 : -1],
                axis=0,
            )
        else:
            i = min(2, len(self._prev_obs))
            dx = self.prev_obj_position[-1] - self.prev_obj_position[-i]
            dtheta = self.prev_obj_orientation[-1] - self.prev_obj_orientation[-i]
        vel = dx / self.dt
        angular_vel = dtheta / self.dt
        return vel, angular_vel

    def compute_fingertip_forces(self, tip_forces_wf):
        tip_forces_wf = tip_forces_wf.reshape(3, 3)
        torque = np.zeros(9)
        for tip_force, Ji in zip(tip_forces_wf, self.tip_jacobians):
            torque += np.squeeze(Ji[:3, :].T @ tip_force)
        return torque

    def compute_pd_control_torques(
        self, observation, desired_position, kp=None, kd=None
    ):
        """
        Compute torque command to reach given target position using a PD
        controller.

        Args:
            desired_position (array-like, shape=(n,)):  Desired joint positions.
            current_position (array-like, shape=(n,)):  Current joint positions.
            current_velocity (array-like, shape=(n,)):  Current joint velocities.
            kp (array-like, shape=(n,)): P-gains, one for each joint.
            kd (array-like, shape=(n,)): D-gains, one for each joint.

        Returns:
            List of torques to be sent to the joints of the finger in order to
            reach the specified joint_positions.
        """
        current_position = observation["robot_observation"]["position"]
        current_velocity = observation["robot_observation"]["velocity"]
        position_error = desired_position - current_position
        if kp is None:
            kp = self.position_gains
        if kd is None:
            kd = self.velocity_gains

        position_feedback = np.asarray(kp) * position_error
        velocity_feedback = np.asarray(kd) * current_velocity

        joint_torques = position_feedback - velocity_feedback

        # set nan entries to zero (nans occur on joints for which the target
        # position was set to nan)
        joint_torques[np.isnan(joint_torques)] = 0.0

        return joint_torques

    def position_control(self, observation, grasp_points):
        applied_torque = np.zeros(9, dtype="float32")
        tip_velocities = self.compute_tip_velocity(observation).reshape((3, 3))
        tip_positions = np.asarray(
            self.kinematics.forward_kinematics(
                observation["robot_observation"]["position"]
            )
        ).reshape((3, 3))
        for finger_index, (tip_pos, tip_vel) in enumerate(
            zip(tip_positions, tip_velocities)
        ):
            local_jacobian = self.tip_jacobians[finger_index][:3, :]

            # pos_target = rot_matrix_finger @ torch.tensor([0.0 , 0.15, 0.09])
            pos_target = grasp_points[finger_index, :]

            # PD controller in xyz space
            pos_error = tip_pos - pos_target
            xyz_force = -5.0 * pos_error - 1.0 * tip_vel

            joint_torques = local_jacobian.T @ xyz_force
            applied_torque += joint_torques
        return applied_torque

    def object_pos_control(self, observation, grasp_points, in_normal, target_pos):
        vel, angular_vel = self.compute_object_velocity()
        # TODO: smooth using kalman filter/position average
        cg_pos = observation["object_observation"]["position"]

        quat = observation["object_observation"]["orientation"]
        quat = Quaternion.fromWLast(quat)
        target_quat = Quaternion.Identity()
        pos_error = cg_pos - target_pos

        if not hasattr(self, "zpos_error_integral"):
            self.zpos_error_integral = 0
        ki = 0.02
        self.zpos_error_integral += pos_error[2] * ki

        object_weight_comp = utils.CUBE_MASS * 9.8 * np.array([0, 0, 1.0])
        # object_weight_comp = - self.zpos_error_integral * torch.tensor([0, 0, 1])

        # Box tunning - tunned without moving CG and compensated normals
        target_force = object_weight_comp - 0.2 * pos_error - 0.10 * vel
        target_torque = (
            -0.4 * (quat @ target_quat.T).to_tangent_space() - 0.01 * angular_vel
        )
        tip_positions = np.asarray(
            self.kinematics.forward_kinematics(
                observation["robot_observation"]["position"]
            )
        ).reshape((3, 3))

        # not necessary for box - changes tunning parameters
        # makes the grasp points and normals follow the tip positions and object rotation
        # TODO grasp points should in object frame
        grasp_points = tip_positions - cg_pos

        in_normal = np.stack([quat.rotate(x) for x in in_normal], axis=0)
        try:
            global_forces = utils.calculate_grip_forces(
                grasp_points, in_normal, target_force, target_torque
            )
        except AssertionError:
            logging.warning("solve failed, maintaining previous forces")
            global_forces = (
                self.previous_global_forces
            )  # will fail if we failed solve on first iteration
            assert global_forces is not None
        else:
            self.previous_global_forces = global_forces

        return self.compute_fingertip_forces(-global_forces)

    def pi_controller(self, des_wrench, step=0):
        if step == 0:
            # reset integral everytime set_point changes
            self._integral = 0
            _, self._des_tip_force = tip_forces_of, tip_forces_wf = np.asarray(
                self.compute_tip_forces(des_wrench)
            )
        if self._current_tip_force is None:
            # TODO: Try using des_wrench instead of zeros here, initializing
            # error term to 0
            obs_tip_force = self._des_tip_force
        else:
            obs_tip_force = self._current_tip_force
        error = self._des_tip_force - np.asarray(obs_tip_force)
        integral_weight = np.array([1] * 3 + [1] * 3 + [0.8, 0.8, 1]).reshape(
            3, 3
        )  # np.where(np.abs(error) > 0.10, step / (step + 1), 0.1)
        self._integral = error * self.ki + self._integral * integral_weight
        tip_forces_of, tip_forces_wf = self.compute_tip_forces(des_wrench)
        tip_forces_wf = np.clip(self._integral + self._des_tip_force, -1.5, 1.5)
        return tip_forces_wf

    def ik_move(
        self,
        observation,
        ft_goal,
        interp_n,
        tol=0.006,
        max_steps=500,
    ):
        current_position = observation["robot_observation"]["position"]
        # current_ft_pos = self.kinematics.forward_kinematics(current_position)
        assert self.action_space.contains("position")
        # TODO: tol currently hardcoded to 0.001, make it smaller and a variable
        q = self.kinematics.inverse_kinematics(
            ft_goal, current_position, tolerance=tol, max_iterations=max_steps
        )
        return q

    def control(self, mode, observation):
        if mode != "up":
            self.grasp_points, self.grasp_normals = self.compute_grasp_points_normals(
                observation
            )
        # safe grasp point moves slightly off of contact surface
        grasp_points, grasp_normals = self.grasp_points, self.grasp_normals
        safe_pos = grasp_points - grasp_normals * 0.1

        if mode == "off":
            pass
        if mode == "safe":
            return self.tip_position_control(observation, safe_pos)
        if mode == "pos":
            return self.tip_position_control(observation, grasp_points)
        if mode == "vel":
            # move radially in along xy plane
            return self.vel_control_force_limit(
                observation, grasp_points, grasp_normals
            )
        if mode == "ik":
            desired_position = self.ik_move(observation, grasp_points)
            return self.compute_pd_control_torques(observation, desired_position)
        if mode == "up":
            # TODO: Add nerf
            # if self.use_grad_est:
            # in_normals = self.get_grad_ests(grasp_points, grasp_normals)
            # else:
            in_normals = grasp_normals
            target_pos = observation["desired_goal"]
            return self.object_pos_control(
                observation, grasp_points, in_normals, target_pos
            )

    def gravity_comp(self, observation):
        q_current = observation["robot_observation"]["position"]
        # dq0 = observation["robot_observation"]["velocity"]
        # g = np.asarray(
        #     p.calculateInverseDynamics(
        #         1, list(q_current), list(dq0), [0.0 for _ in range(9)]
        #     )
        # )
        # return g
        g_torques = np.zeros(9)
        for finger_id in range(3):
            _, g = self.kinematics.compute_lambda_and_g_matrix(
                finger_id, q_current, self.tip_jacobians[finger_id][:3, :]
            )
            g_torques += g
        return g_torques

    @staticmethod
    def set_mode(t):
        if t < 0:  # make 500 to test gravity comp
            # fingers need to compensate for gravity and hold tip positions:
            mode = "off"  # Box needs this to get ot graps position
        elif t < 1000:
            mode = "safe"
        elif t < 1500:
            mode = "pos"
        elif t < 2000:
            mode = "pos"
        else:
            mode = "up"
        return mode

    def predict(self, observation, t):
        if t == 0:
            self._prev_obs.clear()
        # store prev object states to compute velocity
        self._prev_obs.append(observation)
        # in the first few steps keep the target position fixed to move to the
        # initial position (to avoid collisions between the fingers)

        mode = self.set_mode(t)
        if t % 100 == 0:
            print(t, mode)

        if mode != "off":
            # get joint torques for finger 0 to move its tip to the goal position
            self.joint_torques = self.control(mode, observation)
            # self.joint_torques += self.gravity_comp(observation)
            self.joint_torques = self.clip_to_space(self.joint_torques)
        else:
            self.joint_torques = self.gravity_comp(observation)

        # make sure to return a copy, not a reference to self.joint_positions
        self._tip_jacobians = []
        return np.array(self.joint_torques)

    @property
    def tip_jacobians(self):
        if len(self._tip_jacobians) == 0:
            q0 = self._prev_obs[-1]["robot_observation"]["position"]
            self._tip_jacobians = self.compute_tip_jacobians(q0)
        return self._tip_jacobians

    @property
    def prev_obj_position(self):
        return np.array([x["object_observation"]["position"] for x in self._prev_obs])

    @property
    def prev_obj_orientation(self):
        return np.array(
            [
                Rotation.from_quat(x["object_observation"]["orientation"]).as_euler(
                    "xyz"
                )
                for x in self._prev_obs
            ]
        )
