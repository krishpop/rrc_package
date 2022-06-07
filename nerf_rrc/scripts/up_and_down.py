import json
import os.path as osp
import numpy as np
import pybullet as p
import nerf_rrc
from nerf_rrc.pd_control import PDControlPolicy
from nerf_rrc.cube_trajectory_env import ActionType, SimCubeTrajectoryEnv
from nerf_grasping.control import pos_control

# initialize visualization vars
save_vid = False
save_freq = 10
visualization = True


# load goal
goal_json = osp.join(osp.dirname(nerf_rrc.__path__[0]), "goal.json")
goal = json.loads(open(goal_json, "r").read())["_goal"]

# create env
env = SimCubeTrajectoryEnv(
    goal_trajectory=goal,  # passing None to sample a random trajectory
    action_type=ActionType.TORQUE,
    visualization=visualization,
)

# initialize loop vars and visualizer (if visualization==True)
is_done = False
observation = env.reset()
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

t = 0

# create policy
policy = PDControlPolicy(env.action_space, env.info["trajectory"])

control_params = pos_control.load_config()
kin = policy.kinematics
orig_tp = np.asarray(
    policy.kinematics.forward_kinematics(observation["robot_observation"]["position"])
).flatten()

while t < 5000:
    min_height, max_height = 0.05, 0.1
    h = 0.5 * (1 + np.sin(np.pi / 500 * t)) * (max_height - min_height) + min_height
    tip_pos = orig_tp.copy()
    tip_pos[2::3] = h
    q, dq = (
        observation["robot_observation"]["position"],
        observation["robot_observation"]["velocity"],
    )
    action = pos_control.get_joint_torques(
        tip_pos, kin.robot_model, kin.data, q, dq, control_params
    )
    # action = policy.position_pd_control(observation, tip_pos)
    # action += policy.gravity_comp(observation)
    action = policy.clip_to_space(action)
    observation, reward, is_done, info = env.step(action)
    t = info["time_index"]
