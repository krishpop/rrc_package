#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a SimCubeTrajectoryEnv environment and runs one episode using
a dummy policy.
"""
import json
import sys

from rrc_example_package import cube_trajectory_env
from rrc_example_package.pd_control import PDControlPolicy


def main():
    goal_json = sys.argv[1]
    goal = json.loads(open(goal_json, "r").read())["_goal"]
    env = cube_trajectory_env.SimCubeTrajectoryEnv(
        goal_trajectory=goal,  # passing None to sample a random trajectory
        action_type=cube_trajectory_env.ActionType.TORQUE,
        visualization=True,
    )

    is_done = False
    observation = env.reset()
    t = 0

    # policy = PointAtTrajectoryPolicy(env.action_space, env.info["trajectory"])
    policy = PDControlPolicy(env.action_space, env.info["trajectory"])

    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()
