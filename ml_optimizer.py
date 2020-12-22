import argparse
import json
import multiprocessing as mp
import pickle
import sys
from math import tau
from os import path
from queue import Empty
from time import sleep
from typing import List, Optional, Tuple

import numpy as np
from cma import CMAEvolutionStrategy
from cma.interfaces import OOOptimizer
from gym import logger
from gym import make as make_env
from gym import wrappers

from ml_policies import get_policy_class
from policy_vis_model import VisDataModel, VisPolicy, model_filepath


class ConstantEvolutionaryStrategy(OOOptimizer):
    """
    This class just runs the given parameters again and again to give a distribution of outcomes.
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.rewards = []

    def ask(self, number=1000, **kwargs):
        return [self.parameters] * number

    def tell(self, _, rewards):
        self.rewards += rewards

    def disp(self):
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)
        print(f"{mean:3.2f} +-{std:3.2f}")

    def stop(self):
        return False


class CrossEntopyMethodStrategy(OOOptimizer):
    """
    Generic implementation of the cross-entropy method for minimizing a black-box function

    Args:
        parameters_mean (np.array): initial mean of distribution of parameters. Has a shape acceptible by f.
        parameters_std (np.array): initial standard deviation of distribution of parameters. Same shape as parameters_mean.
        elite_frac (float): each batch, select this fraction of the top-performing solutions
    """

    def __init__(self, parameters_mean, parameters_std, elite_frac=0.1):
        self.parameters_mean = np.array(parameters_mean)
        self.parameters_std = np.array(parameters_std)
        self.elite_frac = elite_frac
        assert len(self.parameters_mean.shape) == 1
        assert self.parameters_mean.shape == self.parameters_std.shape

    def ask(self, number=1000):
        return np.array(
            [
                self.parameters_mean + dth
                for dth in self.parameters_std[None, :]
                * np.random.randn(number, self.parameters_mean.size)
            ]
        )

    def tell(self, solutions, rewards):
        assert len(solutions) == len(rewards)
        n_elite = max(1, int(np.round(len(solutions) * self.elite_frac)))

        # Keep the best performing parameters.
        elite_indecies = np.argsort(rewards)[:n_elite]
        elite_solutions = solutions[elite_indecies]
        # Fit a gaussian distribution distribution to the elites.
        self.parameters_mean = elite_solutions.mean(axis=0)
        self.parameters_std = elite_solutions.std(axis=0)

    def stop(self):
        return self.parameters_std.mean() < 0.0001


def do_rollout(agent, env, num_steps, render=False):
    total_reward = 0
    internal_repetitions = 3
    for _ in range(internal_repetitions):
        ob = env.reset()
        for t in range(num_steps):
            a = agent.act(ob)
            (ob, reward, done, _info) = env.step(a)

            total_reward += reward
            if render and t % 1 == 0:
                env.render()
            if done:
                break
    return total_reward / internal_repetitions


def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys

    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api
        import win32con
        import win32process

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)


def evaluator_process(
    env_id: str,
    policy_id: str,
    seed: int,
    num_steps: int,
    render: bool,
    parameters_q: mp.Queue,
    reward_q: mp.Queue,
):
    """
    A class to hold data to run evaluations of the environment against a policy.
    It is designed to be run.
    parameters_q produces items like (ID: int, parameters: np.array)
    reward_q receives items like (ID: int, reward: float)
    """
    env = make_env(env_id)
    env.seed(seed)
    np.random.seed(seed)

    policy_class = get_policy_class(policy_id)
    lowpriority()
    try:
        while True:
            (i, parameters) = parameters_q.get(block=True)
            agent = policy_class(parameters)
            reward = do_rollout(agent, env, num_steps, render)
            reward_q.put((i, reward), block=True)
    except KeyboardInterrupt:
        pass


def write_policy_vis_data(
    policy_id: str, solutions: List[np.array], rewards: List[float]
):
    policy_class = get_policy_class(policy_id)
    vis_resolution = 30
    policy = policy_class([])

    def get_pixel(x, y, parameters) -> float:
        policy.parameters = parameters
        max_speed = 8
        thetadot = y * max_speed * 2 - max_speed
        theta = x * tau - tau / 2
        x = np.sin(theta)
        y = np.cos(theta)
        observation = np.array([y, x, thetadot])
        return policy.act(observation)

    data_model = VisDataModel(
        [
            VisPolicy(
                pixels=np.array(
                    [
                        [
                            get_pixel(x, y, parameters)
                            for x in np.linspace(0, 1, vis_resolution)
                        ]
                        for y in np.linspace(0, 1, vis_resolution)
                    ]
                ),
                reward=reward,
            )
            for parameters, reward in zip(solutions, rewards)
        ]
    )
    with open(model_filepath, "wb") as f:
        pickle.dump(data_model, f)


if __name__ == "__main__":
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true")
    parser.add_argument("env_id", nargs="?", default="Pendulum-v0")
    parser.add_argument("policy_id", nargs="?", default="EnergyPendulumPolicy")
    args = parser.parse_args()

    # Generate our parameter list by running once.
    render_env = make_env(args.env_id)
    bootstrap = get_policy_class(args.policy_id)(parameters=None)
    bootstrap.act(render_env.reset())

    # Set up paralellism for running evaluations (multiprocessing gets around Python's GIL)
    num_steps = 200
    num_processes = 10
    eval_timeout = 20.0  # maximum seconds for a single result to come back
    ctx = mp.get_context("spawn")
    parameters_q = ctx.Queue()
    reward_q = ctx.Queue()
    evaluators = [
        ctx.Process(
            target=evaluator_process,
            args=(
                args.env_id,
                args.policy_id,
                i,
                num_steps,
                False,
                parameters_q,
                reward_q,
            ),
            daemon=True,
        )
        for i in range(num_processes)
    ]
    for evaluator in evaluators:
        evaluator.start()

    # Allow rendering of the current batch while we train the next.
    render_timeout = 20.0
    render_parameters_q = ctx.Queue()
    render_reward_q = ctx.Queue()
    render_evaluator = ctx.Process(
        target=evaluator_process,
        args=(
            args.env_id,
            args.policy_id,
            0,
            num_steps,
            True,
            render_parameters_q,
            render_reward_q,
        ),
        daemon=True,
    )
    render_evaluator.start()

    def evaluate_batch(solutions: List[np.array]) -> List[float]:
        for i, parameters in enumerate(solutions):
            parameters_q.put((i, list(parameters)))

        results = [None] * len(solutions)
        for _ in range(len(solutions)):
            try:
                (i, result) = reward_q.get(block=True, timeout=eval_timeout)
                results[i] = result
            except Empty:
                raise RuntimeError("Not all evaluations completed. :(")
        assert all(result is not None for result in results)
        write_policy_vis_data(args.policy_id, solutions, results)
        return results

    # Define our optimizer.
    batch_size = 20
    es = CMAEvolutionStrategy(
        bootstrap.parameters, np.array(bootstrap.parameters_std).mean(), {}
    )
    # es = CrossEntopyMethodStrategy(
    #     bootstrap.parameters,
    #     bootstrap.parameters_std,
    #     elite_frac=0.1,
    # )
    fixed_parameters = None
    if fixed_parameters is not None:
        es = ConstantEvolutionaryStrategy(fixed_parameters)

    # Train the agent, and snapshot each stage.
    iterdata = None
    batch_best = None
    try:

        i = 0
        while not es.stop():
            solutions = es.ask(number=batch_size)
            rewards = evaluate_batch(solutions)
            batch_best = solutions[np.argmax(rewards)]
            es.tell(
                solutions, [-reward for reward in rewards]
            )  # negate because es minimizes.
            # es.disp()
            print(
                f"i={i:4d} {es.__class__.__name__} {args.policy_id} mean reward: {np.mean(rewards) :4.1f}"
            )
            # Do a little async preview of the current best estimate.
            if args.display:
                if render_parameters_q.empty():
                    render_parameters_q.put((i, batch_best))
                if not render_reward_q.empty():
                    _ = render_reward_q.get(block=True, timeout=render_timeout)
            i += 1
        print("evolution stopped.")

    except KeyboardInterrupt:
        print("cancelled")
    finally:
        if batch_best is not None:
            print("current batch_best: ", list(batch_best))

        # try to work around https://bugs.python.org/issue41761
        for p in evaluators:
            p.kill()
            p.join(timeout=0.1)
            p.close()
        parameters_q.close()
        reward_q.close()

        render_evaluator.kill()
        render_evaluator.join(timeout=0.1)
        render_evaluator.close()
        render_parameters_q.close()
        render_reward_q.close()

        sleep(1)
