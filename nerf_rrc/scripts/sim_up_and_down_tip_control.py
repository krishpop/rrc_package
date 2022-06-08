#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import robot_interfaces
import numpy as np
import robot_fingers
from trifinger_simulation.tasks import move_cube_on_trajectory
from nerf_rrc import cube_trajectory_env
from nerf_rrc.control import pos_control
from nerf_rrc.pd_control import PDControlPolicy


def clip_to_space(action_space, action):
    return np.clip(action, action_space.low, action_space.high)


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    if len(sys.argv) == 4:
        kp = float(sys.argv[2])
        kd = float(sys.argv[3])
        control_params = pos_control.PosControlConfig(
            Kp=kp,
            Kd=kd,
            damping=1e-12,
        )
    elif len(sys.argv) == 3:
        control_params = pos_control.load_config(sys.argv[2])
    else:
        control_params = None
    goal = json.loads(open(goal_json, "r").read())["_goal"]

    env = cube_trajectory_env.SimCubeTrajectoryEnv(
        goal,
        cube_trajectory_env.ActionType.POSITION,
        step_size=1,
        visualization=True,
    )

    policy = PDControlPolicy(env.action_space, goal, control_params)
    kin = policy.kinematics

    observation = env.reset()
    t = 0
    is_done = False
    orig_tp = np.asarray(
        kin.forward_kinematics(observation["robot_observation"]["position"])
    ).flatten()
    min_height, max_height = 0.05, 0.1
    tip_dists = []
    while t < 2000:
        h = 0.5 * (1 + np.sin(np.pi / 500 * t)) * (max_height - min_height) + min_height
        des_tip_pos = orig_tp.copy()
        des_tip_pos[2::3] = h
        policy.position_pd_control(observation, des_tip_pos.reshape(3, 3))
        action = policy.joint_positions.copy()
        action = clip_to_space(env.action_space, action)
        # obs = observation["robot_observation"]
        # q, dq = obs["position"], obs["velocity"]
        # action = pos_control.get_joint_torques(
        #     des_tip_pos, kin.robot_model, kin.data, q, dq, control_params
        # )
        # action = clip_to_space(env.action_space, action)

        observation, reward, is_done, info = env.step(action)
        obs = observation["robot_observation"]
        tip_pos = np.asarray(kin.forward_kinematics(obs["position"])).flatten()
        tip_dist = np.linalg.norm(des_tip_pos - tip_pos)
        tip_dists.append(tip_dist)
        if t % 100 == 0:
            print(f"iteration {t}")
            print("reward:", reward)
            print("tip dist:", tip_dist)
        t = info["time_index"]

    print("Mean tip dist:", np.mean(tip_dists))
    print("Min tip dist:", np.min(tip_dists))
