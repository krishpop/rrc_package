import cvxpy as cp
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks import move_cube

CUBE_MASS = 0.08
CUBE_SIZE = move_cube._CUBE_WIDTH
CUBE_HALF_SIZE = move_cube._CUBE_WIDTH / 2 + 0.001


r = 0.15
theta_0 = 90
theta_1 = 310
theta_2 = 200

FINGER_BASE_POSITIONS = [
    np.array(
        [[np.cos(theta_0 * (np.pi / 180)) * r, np.sin(theta_0 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_1 * (np.pi / 180)) * r, np.sin(theta_1 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_2 * (np.pi / 180)) * r, np.sin(theta_2 * (np.pi / 180)) * r, 0]]
    ),
]
BASE_ANGLE_DEGREES = [0, -120, -240]

# Information about object faces given face_id
OBJ_FACES_INFO = {
    1: {
        "center_param": np.array([0.0, -1.0, 0.0]),
        "face_down_default_quat": np.array([0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 2,
        "up_axis": np.array([0.0, 1.0, 0.0]),  # UP axis when this face is ground face
    },
    2: {
        "center_param": np.array([0.0, 1.0, 0.0]),
        "face_down_default_quat": np.array([-0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 1,
        "up_axis": np.array([0.0, -1.0, 0.0]),
    },
    3: {
        "center_param": np.array([1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, 0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 5,
        "up_axis": np.array([-1.0, 0.0, 0.0]),
    },
    4: {
        "center_param": np.array([0.0, 0.0, 1.0]),
        "face_down_default_quat": np.array([0, 1, 0, 0]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 6,
        "up_axis": np.array([0.0, 0.0, -1.0]),
    },
    5: {
        "center_param": np.array([-1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, -0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 3,
        "up_axis": np.array([1.0, 0.0, 0.0]),
    },
    6: {
        "center_param": np.array([0.0, 0.0, -1.0]),
        "face_down_default_quat": np.array([0, 0, 0, 1]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 4,
        "up_axis": np.array([0.0, 0.0, 1.0]),
    },
}


def get_wf_from_of(p, obj_pose):
    if isinstance(obj_pose, move_cube.Pose):
        obj_pose = obj_pose.as_dict()
    cube_pos_wf = obj_pose["position"]
    cube_quat_wf = obj_pose["orientation"]

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(p) + translation


def get_of_from_wf(p, obj_pose):
    if isinstance(obj_pose, move_cube.Pose):
        obj_pose = obj_pose.as_dict()
    cube_pos_wf = obj_pose["position"]
    cube_quat_wf = obj_pose["orientation"]

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    rotation_inv = rotation.inv()
    translation_inv = -rotation_inv.apply(translation)

    return rotation_inv.apply(p) + translation_inv


def get_closest_ground_face(obj_pose):
    min_z = np.inf
    min_face = None
    for i in range(1, 7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face


def __get_parallel_ground_plane_xy(ground_face):
    if ground_face in [1, 2]:
        x_ind = 0
        y_ind = 2
    if ground_face in [3, 5]:
        x_ind = 2
        y_ind = 1
    if ground_face in [4, 6]:
        x_ind = 0
        y_ind = 1
    return x_ind, y_ind


def __get_distance_from_pt_2_line(a, b, p):
    a = np.squeeze(a)
    b = np.squeeze(b)
    p = np.squeeze(p)

    ba = b - a
    ap = a - p
    c = ba * (np.dot(ap, ba) / np.dot(ba, ba))
    d = ap - c

    return np.sqrt(np.dot(d, d))


def get_cp_pos_wf_from_cp_param(
    cp_param, cube_pos_wf, cube_quat_wf, obj_half_size=CUBE_HALF_SIZE
):
    pos_of, _ = get_cp_of_from_cp_param(cp_param, obj_half_size)

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(pos_of) + translation


def get_cp_of_from_cp_param(cp_param, obj_half_size=CUBE_HALF_SIZE):
    if isinstance(obj_half_size, float):
        obj_shape = (obj_half_size, obj_half_size, obj_half_size)
    else:
        obj_shape = obj_half_size
    cp_of = []
    # Get cp position in OF
    for i in range(3):
        cp_of.append(-obj_shape[i] + (cp_param[i] + 1) * obj_shape[i])

    cp_of = np.asarray(cp_of)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        quat = (0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2)
    elif y_param == 1:
        quat = (0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2)
    elif x_param == 1:
        quat = (0, 0, 1, 0)
    elif z_param == 1:
        quat = (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)
    elif x_param == -1:
        quat = (0, 0, 0, 1)
    elif z_param == -1:
        quat = (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)

    cp = (cp_of, quat)
    return cp


def get_cp_pos_wf_from_cp_params(
    cp_params, cube_pos, cube_quat, obj_half_size=CUBE_HALF_SIZE, **kwargs
):
    """Get contact point positions in world frame from cp_params"""
    fingertip_goal_list = []
    for i in range(len(cp_params)):
        # for i in range(cp_params.shape[0]):
        fingertip_goal_list.append(
            get_cp_pos_wf_from_cp_param(
                cp_params[i], cube_pos, cube_quat, obj_half_size
            )
        )
    return fingertip_goal_list


def closest_face_centers(obj_pose):
    """
    Get initial contact points on cube
    Assign closest cube face to each finger
    Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
    """
    # face that is touching the ground
    ground_face = get_closest_ground_face(obj_pose)

    # Transform finger base positions to object frame
    base_pos_list_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        base_pos_list_of.append(f_of)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)

    xy_distances = np.zeros(
        (3, 2)
    )  # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(base_pos_list_of):
        point_in_plane = np.array(
            [f_of[0, x_ind], f_of[0, y_ind]]
        )  # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0, 0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0, 0]), point_in_plane)

        xy_distances[f_i, 0] = x_dist
        xy_distances[f_i, 1] = y_dist

    # Do the face assignment - greedy approach (assigned closest fingers first)
    free_faces = OBJ_FACES_INFO[ground_face][
        "adjacent_faces"
    ].copy()  # List of face ids that haven't been assigned yet
    assigned_faces = np.zeros(3)
    for i in range(3):
        # Find indices max element in array
        max_ind = np.unravel_index(np.argmax(xy_distances), xy_distances.shape)
        curr_finger_id = max_ind[0]
        furthest_axis = max_ind[1]

        # print("current finger {}".format(curr_finger_id))
        # Do the assignment
        x_dist = xy_distances[curr_finger_id, 0]
        y_dist = xy_distances[curr_finger_id, 1]
        if furthest_axis == 0:  # distance to x axis is greater than to y axis
            if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
        else:
            if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
        # print("first choice face: {}".format(face))

        # Handle faces that may already be assigned
        if face not in free_faces:
            alternate_axis = abs(furthest_axis - 1)
            if alternate_axis == 0:
                if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
            else:
                if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
            # print("second choice face: {}".format(face))

        # If backup face isn't free, assign random face from free_faces
        if face not in free_faces:
            # print("random face:")
            # print(xy_distances[curr_finger_id, :])
            face = free_faces[0]
        assigned_faces[curr_finger_id] = face

        # Replace row with -np.inf so we can assign other fingers
        xy_distances[curr_finger_id, :] = -np.inf
        # Remove face from free_faces
        free_faces.remove(face)
    # print(assigned_faces)
    # Set contact point params
    cp_params = []
    for i in range(3):
        face = assigned_faces[i]
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        # print(i)
        # print(param)
        cp_params.append(param)
    # print("assigning cp params for lifting")
    # print(cp_params)
    return get_cp_pos_wf_from_cp_params(
        cp_params, obj_pose["position"], obj_pose["orientation"]
    )


def skew_matrix(vectors):
    skew = np.zeros(vectors.shape[:-1] + (3, 3))

    skew[..., 0, 1] = -vectors[..., 2]
    skew[..., 1, 2] = -vectors[..., 0]
    skew[..., 2, 0] = -vectors[..., 1]
    skew[..., 1, 0] = vectors[..., 2]
    skew[..., 2, 1] = vectors[..., 0]
    skew[..., 0, 2] = vectors[..., 1]

    return skew


def example_rotation_transform(normals):
    # hopefully no one will try grabing directly under or above
    global_z_axis = np.array([0, 0, 1])

    #  n,3, 1      3, 3                       n, 3, 1
    local_x = skew_matrix(global_z_axis) @ normals[..., None]

    #  n,3,1         n,3,3              n,3,1
    local_y = skew_matrix(normals) @ local_x

    local_x /= np.linalg.norm(local_x, keepdims=True, axis=-2)
    local_y /= np.linalg.norm(local_y, keepdims=True, axis=-2)

    rotations = np.stack([local_x, local_y, normals[..., None]], axis=-1)[..., 0, :]
    return rotations


def calculate_grip_forces(positions, normals, target_force, target_torque):
    """positions are relative to object CG if we want unbalanced torques"""
    mu = 0.5

    torch_input = type(positions) == torch.Tensor
    if torch_input:
        assert type(normals) == torch.Tensor, "numpy vs torch needs to be consistant"
        assert (
            type(target_force) == torch.Tensor
        ), "numpy vs torch needs to be consistant"
        assert (
            type(target_torque) == torch.Tensor
        ), "numpy vs torch needs to be consistant"
        positions = positions.numpy()
        normals = normals.numpy()
        target_force = target_force.numpy()
        target_torque = target_torque.numpy()

    n, _ = positions.shape
    assert normals.shape == (n, 3)
    assert target_force.shape == (3,)

    F = cp.Variable((n, 3))
    constraints = []

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    total_force = np.zeros((3))
    total_torque = np.zeros((3))

    Q = []
    for pos, norm, f in zip(positions, normals, F):
        q = example_rotation_transform(norm)
        Q.append(q)

        total_force += q @ f
        total_torque += skew_matrix(pos) @ q @ f

    constraints.append(total_force == target_force)
    constraints.append(total_torque == target_torque)

    friction_cone = cp.norm(F[:, :2], axis=1) <= mu * F[:, 2]
    constraints.append(friction_cone)

    force_magnitudes = cp.norm(F, axis=1)
    # friction_magnitudes = cp.norm(F[:,2], axis=1)
    prob = cp.Problem(cp.Minimize(cp.max(force_magnitudes)), constraints)
    prob.solve()

    if F.value is None:
        print("Failed to solve!")
        print("F.value: ", F.value)
        print("positions: ", positions)
        print("normals: ", normals)
        print("target_force: ", target_force)
        print("target_torque: ", target_torque)
        assert False

    global_forces = np.zeros_like(F.value)
    for i in range(n):
        global_forces[i, :] = Q[i] @ F.value[i, :]

    if torch_input:
        global_forces = torch.tensor(global_forces).float()

    return global_forces
