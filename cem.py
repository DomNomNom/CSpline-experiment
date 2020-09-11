import gym
from gym import wrappers, logger
import numpy as np
import pickle
import json, sys, os
from os import path
import argparse
from math import atan2
from typing import Optional


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

class PendulumPolicy(ParameterCreatingPolicy):
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


def cem(f, parameters_mean, parameters_std, batch_size, elite_frac=.2):
    '''
    Generic implementation of the cross-entropy method for maximizing a black-box function

    Args:
        f: a function mapping from parameters -> reward. CEM attempts to maximize reward. (total reward in RL scenarios)
        parameters_mean (np.array): initial mean of distribution of parameters. Has a shape acceptible by f.
        parameters_std (np.array): initial standard deviation of distribution of parameters. Same shape as parameters_mean.
        batch_size (int): number of samples of theta to evaluate per batch
        elite_frac (float): each batch, select this fraction of the top-performing samples

    returns:
        An infinite generator of dicts. Subsequent dicts correspond to iterations of CEM algorithm.
        The dicts contain the following values:
        'parameters_mean': the mean of distribution of parameters. Should be a valid to pass to f.
        'parameters_std': the standard deviation of distribution of parameters
        'samples': used samples from the distribution of parameters.
        'rewards': numpy array with outputs of f evaluated at corresponding samples
        'elite_samples': a subset of samples chosen for the next iteration.
    '''
    parameters_mean = np.array(parameters_mean)
    parameters_std = np.array(parameters_std)
    assert len(parameters_mean.shape) == 1
    assert parameters_mean.shape == parameters_std.shape
    n_elite = int(np.round(batch_size*elite_frac))

    while True:  # The caller is responsible for deciding when to stop.

        # Draw samples from a guassian distribution over parameters
        samples = np.array([parameters_mean + dth for dth in  parameters_std[None,:]*np.random.randn(batch_size, parameters_mean.size)])
        rewards = np.array([f(parameters) for parameters in samples])
        # Keep the best performing parameters.
        elite_indecies = rewards.argsort()[::-1][:n_elite]
        elite_samples = samples[elite_indecies]
        # Fit a gaussian distribution distribution to the elites.
        parameters_mean = elite_samples.mean(axis=0)
        parameters_std = elite_samples.std(axis=0)

        # Give a progress update.
        yield {
            'parameters_mean' : parameters_mean,  # our current best guess at the best parameters.
            'parameters_std': parameters_std,
            'samples': samples,
            'rewards' : rewards,  # same order as samples.
            'elite_samples': elite_samples,
        }

def do_rollout(agent, env, num_steps, render=False):
    total_reward = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)

        total_reward += reward
        if render and t%1==0: env.render()
        if done: break
    return total_reward


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('env_id', nargs='?', default='Pendulum-v0')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.seed(0)
    np.random.seed(0)
    num_steps = 200
    if args.env_id == 'Pendulum-v0':
        policy_class = PendulumPolicy
    elif args.env_id == 'CartPole-v0':
        policy_class = CartpolePolicy
        # policy_class = CartpolePIDPolicy
    else:
        raise NotImplementedError(f'No policy for environment "{args.env_id}"')

    # Generate our parameter list by running once.
    bootstrap = policy_class(parameters=None)
    bootstrap.act(env.reset())
    cem_args = {
        'parameters_mean': bootstrap.parameters,
        'parameters_std': bootstrap.parameters_std,
        'batch_size': 250,
        'elite_frac': 0.2,
    }

    # ----------------------------------------
    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/cem-agent-results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # Prepare snapshotting
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
    # Write out the env so we store the parameters of this environment.
    writefile('info.json', json.dumps({
        'argv': sys.argv,
        'env_id': env.spec.id,
        'cem_args': cem_args,
    }))
    # ------------------------------------------

    def noisy_evaluation(parameters):
        agent = policy_class(parameters)
        return do_rollout(agent, env, num_steps)

    # Train the agent, and snapshot each stage.
    for (i, iterdata) in enumerate(cem(noisy_evaluation, **cem_args)):
        print('Iteration %2i. Episode mean reward: %5.0f  std: %2.3f'%(i, iterdata['rewards'].mean(), iterdata['parameters_std'].mean()))

        # Do a little preview of the current best estimate.
        agent = policy_class(iterdata['parameters_mean'])
        if args.display: do_rollout(agent, env, num_steps, render=True)
        writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))

        if iterdata['parameters_std'].mean() < 0.0001:
            print('done: parameters have converged: ', iterdata['parameters_mean'])
            break

    env.close()
