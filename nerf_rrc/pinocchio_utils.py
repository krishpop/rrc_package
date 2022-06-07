import typing

import numpy as np
import pinocchio


class Kinematics:
    """Forward and inverse kinematics for arbitrary Finger robots.

    Provides forward and inverse kinematics functions for a Finger robot with
    arbitrarily many independent fingers.
    """

    # values from trifinger_simulation/robot_properties_fingers/urdf/pro/trifingerpro.urdf
    m1 = 0.26689  # finger_upper_link_i mass
    m2 = 0.27163  # finger_middle_link_i mass
    m3 = 0.05569  # finger_lower_link_i mass
    ms = [m1, m2, m3]
    I1 = np.array(
        [
            [0.00102362, 0.00000889, -0.00000019],
            [0.00000889, 0.00006450, 0.00000106],
            [-0.00000019, 0.00000106, 0.00102225],
        ]
    )  # finger_upper_link_i inertia matrix
    I2 = np.array(
        [
            [0.00094060, -0.00000046, -0.00003479],
            [-0.00000046, 0.00094824, 0.00000164],
            [-0.00003479, 0.00000164, 0.00007573],
        ]
    )  # finger_middle_link_i inertia matrix
    I3 = np.array(
        [
            [0.00013626, 0.00000000, -0.00000662],
            [0.00000000, 0.00013372, 0.00000004],
            [-0.00000662, 0.00000004, 0.00000667],
        ]
    )  # finger_lower_link_i inertia matrix
    Is = [I1, I2, I3]

    def __init__(self, finger_urdf_path: str, tip_link_names: typing.Iterable[str]):
        """Initializes the robot model.

        Args:
            finger_urdf_path:  Path to the URDF file describing the robot.
            tip_link_names:  Names of the finger tip frames, one per finger.
        """
        self.robot_model = pinocchio.buildModelFromUrdf(finger_urdf_path)
        self.data = self.robot_model.createData()
        self.tip_link_ids = [
            self.robot_model.getFrameId(link_name) for link_name in tip_link_names
        ]
        finger_names = ["0", "120", "240"]
        finger_links = ["upper", "middle", "lower"]
        self.finger_link_names = []
        for finger_name in finger_names:
            for finger_link in finger_links:
                self.finger_link_names.append(
                    f"finger_{finger_link}_link_{finger_name}"
                )
        self.finger_link_ids = {
            link_name: self.robot_model.getFrameId(link_name)
            for link_name in self.finger_link_names
        }

    def forward_kinematics(self, joint_positions) -> typing.List[np.ndarray]:
        """Compute end-effector positions for the given joint configuration.

        Args:
            joint_positions:  Flat list of angular joint positions.

        Returns:
            List of end-effector positions. Each position is given as an
            np.array with x,y,z positions.
        """
        pinocchio.framesForwardKinematics(
            self.robot_model,
            self.data,
            joint_positions,
        )

        return [
            np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist()
            for link_id in self.tip_link_ids
        ]

    def compute_local_jacobian(self, frame_id: int, q0: np.ndarray) -> np.ndarray:
        pinocchio.computeJointJacobians(
            self.robot_model,
            self.data,
            q0,
        )
        pinocchio.framesForwardKinematics(
            self.robot_model,
            self.data,
            q0,
        )
        Ji = pinocchio.getFrameJacobian(
            self.robot_model,
            self.data,
            frame_id,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return Ji

    def compute_gravity_comp_one_finger(self, finger_id, q, Jvi, grav=-9.81):
        links = ["lower", "middle", "upper"]
        finger_links = [f"finger_{link}_link_{finger_id*120}" for link in links]
        g = 0
        for j, finger_link in enumerate(finger_links):
            fid = self.finger_link_ids[finger_link] - 2
            Jj = self.compute_local_jacobian(fid, q)
            Jjv = Jj[:3, :]
            # want to resist grav_force so acceleration at tip is 0
            grav_force = np.array([0, 0, grav * self.ms[j]])
            g -= Jjv.T @ grav_force
        return g

    def compute_lambda_and_g_matrix(self, finger_id, q, Jvi, grav=-9.81):
        Ai = np.zeros((9, 9))
        g = np.zeros(9)
        grav = np.array([0, 0, grav])
        links = ["upper", "middle", "lower"]
        finger_links = [f"finger_{link}_link_{finger_id * 120}" for link in links]
        for j, finger_link in enumerate(finger_links):
            fid = self.finger_link_ids[finger_link] - 2
            Jj = self.compute_local_jacobian(fid, q)
            Jjv = Jj[:3, :]
            Jjw = Jj[3:, :]
            g -= self.ms[j] * Jjv.T @ grav  # * 0.33
            Ai += self.ms[j] * Jjv.T @ Jjv + Jjw.T @ self.Is[j] @ Jjw  # * 0.33
        # Ai is kinetic energy matrix in configuration space
        Jvi_inv = np.linalg.pinv(Jvi)
        Li = Jvi_inv.T @ Ai @ Jvi_inv
        # Li = Ai;
        # Li is Lambda matrix (kinetic energy matrix in operation space)
        return Li, g

    def _inverse_kinematics_step(
        self, frame_id: int, xdes: np.ndarray, q0: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Compute one IK iteration for a single finger."""
        dt = 1.0e-1
        Ji = self.compute_local_jacobian(frame_id, q0)[:3, :]
        xcurrent = self.data.oMf[frame_id].translation
        try:
            Jinv = np.linalg.inv(Ji)
        except Exception:
            Jinv = np.linalg.pinv(Ji)
        err = xdes - xcurrent
        dq = Jinv.dot(xdes - xcurrent)
        qnext = pinocchio.integrate(self.robot_model, q0, dt * dq)
        return qnext, err

    def inverse_kinematics_one_finger(
        self,
        finger_idx: int,
        tip_target_position: np.ndarray,
        joint_angles_guess: np.ndarray,
        tolerance: float = 0.005,
        max_iterations: int = 1000,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Inverse kinematics for a single finger.

        Args:
            finger_idx: Index of the finger.
            tip_target_positions: Target position for the finger tip in world
                frame.
            joint_angles_guess: Initial guess for the joint angles.
            tolerance: Position error tolerance.  Stop if the error is less
                then that.
            max_iterations: Max. number of iterations.

        Returns:
            tuple: First element is the joint configuration (for joints that
                are not part of the specified finger, the values from the
                initial guess are kept).
                Second element is (x,y,z)-error of the tip position.
        """
        q = joint_angles_guess
        for i in range(max_iterations):
            q, err = self._inverse_kinematics_step(
                self.tip_link_ids[finger_idx], tip_target_position, q
            )

            if np.linalg.norm(err) < tolerance:
                break
        return q, err

    def inverse_kinematics(
        self,
        tip_target_positions: typing.Iterable[np.ndarray],
        joint_angles_guess: np.ndarray,
        tolerance: float = 0.005,
        max_iterations: int = 1000,
    ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        """Inverse kinematics for the whole manipulator.

        Args:
            tip_target_positions: List of finger tip target positions, one for
                each finger.
            joint_angles_guess: See :meth:`inverse_kinematics_one_finger`.
            tolerance: See :meth:`inverse_kinematics_one_finger`.
            max_iterations: See :meth:`inverse_kinematics_one_finger`.

        Returns:
            tuple: First element is the joint configuration, second element is
            a list of (x,y,z)-errors of the tip positions.
        """
        q = joint_angles_guess
        errors = []
        for i, pos in enumerate(tip_target_positions):
            q, err = self.inverse_kinematics_one_finger(
                i, pos, q, tolerance, max_iterations
            )
            errors.append(err)

        return q, errors
