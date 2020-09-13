from gym import wrappers, logger, make as make_env
from math import atan2
from os import path
from queue import Empty
from typing import Optional, List, Tuple
from cma import CMAEvolutionStrategy
from cma.interfaces import OOOptimizer
import argparse
import json
import sys
import multiprocessing as mp
import numpy as np
import pickle

class ParameterCreatingPolicy():
    '''
    This class can create or read parameters when act() executes.
    This allows fast iteration on the act() code without needing to change
    the declarations of parameters since this declaration information
    is created in a dummy run of the act() function.
    '''
    MODE_CREATE_PARAMS = 1
    MODE_READ_PARAMS = 2

    def __init__(self, parameters: Optional[np.array]=None):
        '''
        When parameters are not specified, the default values are chosen in when act()
        which then become visible.
        When parameters are specified, it is assumed they match the shape of the default parameters.
        '''
        self.mode = self.MODE_CREATE_PARAMS if parameters is None else self.MODE_READ_PARAMS
        self.parameters = [] if parameters is None else parameters  # these are our parameters to be tuned by the optimizer
        self.parameters_std = [] if parameters is None else None   # some hints for the optimizer for their standard deviations.
        self.current_parameter = 0

    def param(self, guess_value: float, guess_std: Optional[float]=None):
        '''
        In create mode:
            Create parameters with an estimation over their initial value.
            guess_value is returned.
        In read mode:
            return a value from parameters as passed to __init__()
            by using a counter which assumes we get called in the same order.
            Arguments to this function are discarded.
        '''
        if self.mode == self.MODE_CREATE_PARAMS:
            self.parameters.append(guess_value)
            if guess_std is None:
                if guess_value == 0:
                    guess_std = 1
                else:
                    guess_std = abs(guess_value)
            self.parameters_std.append(guess_std)
            return guess_value
        elif self.mode == self.MODE_READ_PARAMS:
            assert 0 <= self.current_parameter < len(self.parameters), f'make sure to reset self.current_parameter and to always call self.param() the same amount of times. \n got current_parameter={self.current_parameter}, want 0 <= current_parameter < {len(self.parameters)}'
            parameter = self.parameters[self.current_parameter]
            self.current_parameter += 1
            return parameter

    def act(self, observation):
        self.current_parameter = 0  # any subclass must reset this counter!
        raise NotImplementedError()

class EnergyPendulumPolicy(ParameterCreatingPolicy):
    '''
    A policy to solve Pendulum-v0
    '''
    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        y, x, thetadot = observation
        theta = atan2(x,y)

        torque_add_energy = thetadot*10000
        torque_top_control = p(0) * x + p(0) * thetadot + p(0) * y + p(0) * x**2
        energy = y * p(1) + abs(thetadot)
        torque = torque_add_energy if energy < p(1, 10) else torque_top_control

        return [torque]

class PiecewisePendulumPolicy(ParameterCreatingPolicy):
    '''
    A piecewise linear policy to solve Pendulum-v0
    '''
    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.


        squared = observation * observation # square each variable
        all_variables = np.hstack([observation, squared])


        def p_vec3():
            return np.array([p(0), p(0), p(0)])

        def linear_function():
            return p_vec3().dot(observation)

        num_piecewise = 3
        conditions = [linear_function() > 0 for _ in range(num_piecewise)]
        torques = [linear_function() for _ in range(num_piecewise)]
        for condition, torque in zip(conditions, torques):
            if condition:
                return [torque]

        return [torques[-1]]

class CartpolePolicy(ParameterCreatingPolicy):
    '''
    A policy to solve CartPole-v0
    '''
    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        x, x_dot, theta, theta_dot = observation

        go_right = (
            p(0) * x +
            p(0) * x_dot +
            p(0) * theta +
            p(0) * theta_dot
        ) > 0
        return 1 if go_right else 0

class CartpolePIDPolicy(ParameterCreatingPolicy):
    '''
    A PID controller for CartPole-v0 based on
    https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c
    without knowing the PID values.
    '''

    desired_state = np.array([0, 0, 0, 0])
    desired_mask  = np.array([0, 0, 1, 0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

    def act(self, state):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        error = state - self.desired_state
        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error

        P, I, D = p(0.0), p(0.0), p(0.0)
        # P, I, D = p(.1, .01), p(.01, .01), p(.5, .01)  # uncommenting this would be cheating.
        pid = np.dot(P * error + I * self.integral + D * self.derivative, self.desired_mask)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        return action


def cem(batch_f, parameters_mean, parameters_std, batch_size, elite_frac=.2):
    '''
    Generic implementation of the cross-entropy method for maximizing a black-box function

    Args:
        batch_f: A function mapping from [parameters0, parameters1...]  -> [reward0, reward1, ...]
            CEM attempts to maximize reward. (total reward in RL scenarios)
            In usual implementations, cem() only takes the function mapping from parameters to rewards but this allows more parallelism.
        parameters_mean (np.array): initial mean of distribution of parameters. Has a shape acceptible by f.
        parameters_std (np.array): initial standard deviation of distribution of parameters. Same shape as parameters_mean.
        batch_size (int): number of solutions of theta to evaluate per batch
        elite_frac (float): each batch, select this fraction of the top-performing solutions

    returns:
        An infinite generator of dicts. Subsequent dicts correspond to iterations of CEM algorithm.
        The dicts contain the following values:
        'parameters_mean': the mean of distribution of parameters. Should be a valid to pass to f.
        'parameters_std': the standard deviation of distribution of parameters
        'solutions': used solutions from the distribution of parameters.
        'rewards': numpy array with outputs of f evaluated at corresponding solutions
        'elite_solutions': a subset of solutions chosen for the next iteration.
    '''
    parameters_mean = np.array(parameters_mean)
    parameters_std = np.array(parameters_std)
    assert len(parameters_mean.shape) == 1
    assert parameters_mean.shape == parameters_std.shape
    n_elite = int(np.round(batch_size*elite_frac))

    while True:  # The caller is responsible for deciding when to stop.

        # Draw solutions from a guassian distribution over parameters
        solutions = np.array([parameters_mean + dth for dth in  parameters_std[None,:]*np.random.randn(batch_size, parameters_mean.size)])
        rewards = np.array(batch_f(solutions))

        # Keep the best performing parameters.
        elite_indecies = rewards.argsort()[::-1][:n_elite]
        elite_solutions = solutions[elite_indecies]
        # Fit a gaussian distribution distribution to the elites.
        parameters_mean = elite_solutions.mean(axis=0)
        parameters_std = elite_solutions.std(axis=0)

        # Give a progress update.
        yield {
            'parameters_mean' : parameters_mean,  # our current best guess at the best parameters.
            'parameters_std': parameters_std,
            'solutions': solutions,
            'rewards' : rewards,  # same order as solutions.
            'elite_solutions': elite_solutions,
        }


def do_rollout(agent, env, num_steps, render=False):
    total_reward = 0
    internal_repetitions = 10
    for _ in range(internal_repetitions):
        ob = env.reset()
        for t in range(num_steps):
            a = agent.act(ob)
            (ob, reward, done, _info) = env.step(a)

            total_reward += reward
            if render and t%1==0: env.render()
            if done: break
    return total_reward / internal_repetitions

def get_policy_class(policy_id):
    policy_id_to_policy_class = {
        'EnergyPendulumPolicy': EnergyPendulumPolicy,
        'PiecewisePendulumPolicy': PiecewisePendulumPolicy,
        'CartpolePolicy': CartpolePolicy,
        'CartpolePIDPolicy': CartpolePIDPolicy,
    }
    if policy_id not in policy_id_to_policy_class:
        raise NotImplementedError(f'No policy_id not found: {repr(policy_id)}')
    return policy_id_to_policy_class[policy_id]

def evaluator_process(env_id: str, policy_id: str, seed: int, num_steps: int, render: bool,
        parameters_q: mp.Queue, reward_q: mp.Queue):
    '''
    A class to hold data to run evaluations of the environment against a policy.
    It is designed to be run.
    parameters_q produces items like (ID: int, parameters: np.array)
    reward_q receives items like (ID: int, reward: float)
    '''
    env = make_env(env_id)
    env.seed(seed)
    np.random.seed(seed)

    policy_class = get_policy_class(policy_id)
    try:
        while True:
            (i, parameters) = parameters_q.get(block=True)
            agent = policy_class(parameters)
            reward = do_rollout(agent, env, num_steps, render)
            reward_q.put((i, reward), block=True)
    except KeyboardInterrupt:
        pass

class ConstantEvolutionaryStrategy(OOOptimizer):
    '''
    This class just runs the given parameters again and again to give a distribution of outcomes.
    '''
    def __init__(self, parameters, num_evals_per_batch=1000):
        self.parameters = parameters
        self.num_evals_per_batch = num_evals_per_batch
        self.rewards = []

    def ask(self):
        return [self.parameters] * self.num_evals_per_batch

    def tell(self, _, rewards):
        self.rewards += rewards

    def disp(self):
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)
        print(f'{mean:3.2f} +-{std:3.2f}')

    def stop(self):
        return False


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('env_id', nargs='?', default='Pendulum-v0')
    parser.add_argument('policy_id', nargs='?', default='EnergyPendulumPolicy')
    args = parser.parse_args()

    # Generate our parameter list by running once.
    render_env = make_env(args.env_id)
    bootstrap = get_policy_class(args.policy_id)(parameters=None)
    bootstrap.act(render_env.reset())
    cem_args = {
        'parameters_mean': bootstrap.parameters,
        'parameters_std': bootstrap.parameters_std,
        'batch_size': 250,
        'elite_frac': 0.1,
    }

    # ----------------------------------------
    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/cem-agent-results'
    # Prepare snapshotting
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
    # Write out the env so we store the parameters of this environment.
    writefile('info.json', json.dumps({
        'argv': sys.argv,
        'env_id': args.env_id,
        'policy_id': args.policy_id,
        'cem_args': cem_args,
    }))
    # ------------------------------------------


    # Set up paralellism for running evaluations (multiprocessing gets around Python's GIL)
    num_steps = num_steps = 200
    num_processes = 12
    eval_timeout = 1.0  # maximum seconds for a single result to come back
    ctx = mp.get_context('spawn')
    parameters_q = ctx.Queue()
    reward_q = ctx.Queue()
    evaluators = [
        ctx.Process(
            target=evaluator_process,
            args=(args.env_id, args.policy_id, i, num_steps, False, parameters_q, reward_q),
            daemon=True
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
        args=(args.env_id, args.policy_id, 0, num_steps, True, render_parameters_q, render_reward_q),
        daemon=True
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
                raise RuntimeError('Not all evaluations completed. :(')
        assert all(result is not None for result in results)
        return results

    es = CMAEvolutionStrategy(
        bootstrap.parameters,
        np.array(bootstrap.parameters_std).mean(),
        {}
    )
    fixed_parameters = None
    # fixed_parameters =  [-35.58775647064161, 7.139141109188349, 0.15075132253285728, 17.941036682053063, -13.076765083017541, 12.493062191075545, -8.244998281690211, -4.064504191474658, -12.403395249634258, -4.78482465245739, 11.249974937254391, 5.247903860892631, 5.090399438379839, -36.18993376643062, -6.5979123984128645, 8.232448235480183, 12.767369625204546, -5.977121808173037]
    if fixed_parameters is not None:
        es = ConstantEvolutionaryStrategy(fixed_parameters)


    # Train the agent, and snapshot each stage.
    iterdata = None
    batch_best = None
    try:
        # for (i, iterdata) in enumerate(cem(evaluate_batch, **cem_args)):
        #     print('Iteration %2i. Episode mean reward: %5.0f  std: %2.3f'%(i, iterdata['rewards'].mean(), iterdata['parameters_std'].mean()))

        #     # Do a little preview of the current best estimate.
        #     if args.display:
        #         if i > 0:
        #             _ = render_reward_q.get(block=True, timeout=render_timeout)
        #         render_parameters_q.put((i, iterdata['parameters_mean']))

        #     if iterdata['parameters_std'].mean() < 0.0001:
        #         print('done: parameters have converged: ', iterdata['parameters_mean'])
        #         break


        i = 0
        while not es.stop():
            solutions = es.ask()
            rewards = evaluate_batch(solutions)
            batch_best = solutions[np.argmax(rewards)]
            es.tell(solutions, [-reward for reward in rewards])  # negate because es minimizes.
            es.disp()
            # Do a little async preview of the current best estimate.
            if args.display:
                if render_parameters_q.empty():
                    render_parameters_q.put((i, batch_best))
                if not render_reward_q.empty():
                    _ = render_reward_q.get(block=True, timeout=render_timeout)
            i += 1
        es.result_pretty()
    except KeyboardInterrupt:
        print('cancelled')
    finally:
        if batch_best is not None:
            print('current batch_best: ', list(batch_best))

        # try to work around https://bugs.python.org/issue41761
        for p in evaluators:
            p.kill()
            p.join(timeout=.1)
            p.close()
        parameters_q.close()
        reward_q.close()

        render_evaluator.kill()
        render_evaluator.join(timeout=.1)
        render_evaluator.close()
        render_parameters_q.close()
        render_reward_q.close()
