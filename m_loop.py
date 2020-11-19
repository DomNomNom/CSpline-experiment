#Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function
__metaclass__ = type

#Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

#Other imports
import numpy as np
import time

from queue import Queue, Empty

import multiprocessing as mp
from ml_optimizer import evaluator_process
from gym import wrappers, logger, make as make_env
from ml_policies import get_policy_class


#Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):

    #Initialization of the interface, including this method is optional
    def __init__(self, ctx, env_id, policy_id):
        #You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()
        self.env_id = env_id
        self.policy_id = policy_id

        num_steps = 200
        num_processes = 10

        self.parameters_q = ctx.Queue()
        self.reward_q = ctx.Queue()
        self.evaluators = [
            ctx.Process(
                target=evaluator_process,
                args=(self.env_id, self.policy_id, i, num_steps, False, self.parameters_q, self.reward_q),
                daemon=True
            )
            for i in range(num_processes)
        ]
        for evaluator in self.evaluators:
            evaluator.start()


        # self.render_timeout = 20.0
        # self.render_parameters_q = ctx.Queue()
        # self.render_reward_q = ctx.Queue()
        # self.render_evaluator = ctx.Process(
        #     target=evaluator_process,
        #     args=(self.env_id, self.policy_id, 0, num_steps, True, self.render_parameters_q, self.render_reward_q),
        #     daemon=True
        # )
        # self.render_evaluator.start()


    #You must include the get_next_cost_dict method in your class
    #this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):
        params = params_dict['params']
        # self.render_parameters_q.put((self.iteration, params))
        # i, reward = self.render_reward_q.get(block=True, timeout=20.0)

        internal_repetitions = 20
        eval_timeout = 10.0
        for i in range(internal_repetitions):
            self.parameters_q.put((i, params))

        rewards = [None] * internal_repetitions
        for _ in range(internal_repetitions):
            try:
                (i, reward) = self.reward_q.get(block=True, timeout=eval_timeout)
                rewards[i] = reward
            except Empty:
                raise RuntimeError('Not all evaluations completed. :(')
        assert all(reward is not None for reward in rewards)

        return {
            'cost': -np.array(rewards).mean(),
            'uncer': np.array(rewards).std(),
            'bad': False
        }

def main():
    #M-LOOP can be run with three commands

    env_id = "Pendulum-v0"
    policy_id = "PiecewisePendulumPolicy"
    ctx = mp.get_context('spawn')
    interface = CustomInterface(ctx, env_id, policy_id)

    bootstrap_env = make_env(env_id)
    bootstrap = get_policy_class(policy_id)(parameters=None)
    bootstrap.act(bootstrap_env.reset())

    controller = mlc.create_controller(interface,
        max_num_runs = 10000,
        target_cost = -2.99,
        # num_params = 3,
        # min_boundary = [-2,-2,-2],
        # max_boundary = [2,2,2]
        num_params = len(bootstrap.parameters),
        min_boundary = np.array(bootstrap.parameters) - 3*np.array(bootstrap.parameters_std),
        max_boundary = np.array(bootstrap.parameters) + 3*np.array(bootstrap.parameters_std),
    )
    #To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()
    print(controller.learner)

    #The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)

    #You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)


#Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()
