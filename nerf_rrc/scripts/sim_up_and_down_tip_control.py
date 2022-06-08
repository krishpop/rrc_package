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
    goal = json.loads(open(goal_json, "r").read())["_goal"]

    env = cube_trajectory_env.SimCubeTrajectoryEnv(
        goal,
        cube_trajectory_env.ActionType.TORQUE,
        step_size=1,
    )
    control_params = pos_control.PosControlConfig(
        Kp=20.0, Kd=[0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5], damping=1e-12
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
    while not is_done:
        if t % 100:
            print(f"iteration {t}")
        h = 0.5 * (1 + np.sin(np.pi / 500 * t)) * (max_height - min_height) + min_height
        tip_pos = orig_tp.copy()
        tip_pos[2::3] = h
        obs = observation["robot_observation"]
        q, dq = obs["position"], obs["velocity"]
        action = pos_control.get_joint_torques(
            tip_pos, kin.robot_model, kin.data, q, dq, control_params
        )
        action = clip_to_space(env.action_space, action)

        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        print("reward:", reward)
