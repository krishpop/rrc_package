#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a SimCubeTrajectoryEnv environment and runs one episode using
a dummy policy.
"""
import json
import sys
import numpy as np

from nerf_rrc import cube_trajectory_env
from nerf_rrc.pd_control import PDControlPolicy
import trifinger_simulation.tasks.move_cube_on_trajectory as task


def main():
    goal_json = sys.argv[1]
    goal = json.loads(open(goal_json, "r").read())["_goal"]
    initial_pose = task.move_cube.Pose(
        position=np.array([-0.0023, 0.0013, 0.034]),
        orientation=np.array(
            [3.04629235e-03, 1.34290364e-02, 7.28152931e-01, 6.85276330e-01]
        ),
    )
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
