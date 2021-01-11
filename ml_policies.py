import itertools
from math import atan2, tau
from typing import List, Optional, Tuple

import numpy as np


class ParameterCreatingPolicy:
    """
    This class can create or read parameters when act() executes.
    This allows fast iteration on the act() code without needing to change
    the declarations of parameters since this declaration information
    is created in a dummy run of the act() function.
    """

    MODE_CREATE_PARAMS = 1
    MODE_READ_PARAMS = 2

    def __init__(self, parameters: Optional[np.array] = None):
        """
        When parameters are not specified, the default values are chosen in when act()
        which then become visible.
        When parameters are specified, it is assumed they match the shape of the default parameters.
        """
        self.mode = (
            self.MODE_CREATE_PARAMS if parameters is None else self.MODE_READ_PARAMS
        )
        self.parameters = (
            [] if parameters is None else parameters
        )  # these are our parameters to be tuned by the optimizer
        self.parameters_std = (
            [] if parameters is None else None
        )  # some hints for the optimizer for their standard deviations.
        self.current_parameter = 0

    def param(self, guess_value: float, guess_std: Optional[float] = None):
        """
        In create mode:
            Create parameters with an estimation over their initial value.
            guess_value is returned.
        In read mode:
            return a value from parameters as passed to __init__()
        by using a counter which assumes we get called in the same order.
            Arguments to this function are discarded.
        """
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
            assert (
                0 <= self.current_parameter < len(self.parameters)
            ), f"make sure to reset self.current_parameter and to always call self.param() the same amount of times. \n got current_parameter={self.current_parameter}, want 0 <= current_parameter < {len(self.parameters)}"
            parameter = self.parameters[self.current_parameter]
            self.current_parameter += 1
            return parameter

    def act(self, observation):
        self.current_parameter = 0  # any subclass must reset this counter!
        raise NotImplementedError()


class EnergyPendulumPolicy(ParameterCreatingPolicy):
    """
    A policy to solve Pendulum-v0
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        y, x, thetadot = observation

        torque_add_energy = thetadot * 10000
        torque_top_control = p(0) * x + p(0) * thetadot + p(0) * y + p(0) * x ** 2
        energy = y * p(1) + abs(thetadot)
        torque = torque_add_energy if energy < p(1, 10) else torque_top_control

        return [torque]


class PiecewisePendulumPolicy(ParameterCreatingPolicy):
    """
    A piecewise linear policy to solve Pendulum-v0
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        squared = observation * observation  # square each variable
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

        return [np.clip(torques[-1], -2, 2)]


class CodeGolfPendulumPolicy(ParameterCreatingPolicy):
    """
    A policy to solve Pendulum-v0
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.
        p(0)
        p(0)

        y, x, thetadot = observation
        return [thetadot if y < 0 else -5 * thetadot + -11 * x]


class DecisionTreePendulumPolicy(ParameterCreatingPolicy):
    """
    A piecewise linear policy to solve Pendulum-v0
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        squared = observation * observation  # square each variable
        all_variables = np.hstack([observation, squared])

        def p_vec3():
            return np.array([p(0), p(0), p(0)])

        def linear_function():
            return p_vec3().dot(observation)

        depth = 1
        # Have 2^depth - 1 binary decision nodes
        # Then another layer of action functions returning torque.
        funcs = [linear_function() for _ in range(2 ** (depth + 1) - 1)]
        i = 0
        for _ in range(depth):
            if funcs[i] > 0:
                i = 2 * i + 1
            else:
                i = 2 * i + 2
        return [funcs[i]]


class CompositePendulumPolicy(DecisionTreePendulumPolicy):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.fixed = DecisionTreePendulumPolicy(
            [
                -16.672725288303283,
                -12.134884959878233,
                -1.4285792155686454,
                -20.39331887632949,
                5.652793888691333,
                4.996374674468103,
                0.3263373036218867,
                -16.441060042818854,
                -4.130268121623661,
            ]
        )

    def act(self, observation):
        out_super = super().act(observation)
        out_fixed = self.fixed.act(observation)
        if np.dot(observation, [self.param(0), self.param(0), self.param(0)]) > 0:
            return out_fixed

        return out_fixed


class NeuralNetPendulumPolicy(ParameterCreatingPolicy):
    """
    A policy that uses a neural net to make decisions
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        layers = [
            np.array(observation),
            np.zeros(shape=(3,)),
            np.zeros(shape=(3,)),
            np.zeros(shape=(1,)),
        ]
        connections = []
        for i in range(len(layers) - 1):
            matrix = np.array(
                [
                    [p(0) for _ in range(len(layers[i]))]
                    for _ in range(len(layers[i + 1]))
                ]
            )
            bias = [p(0) for _ in range(len(layers[i + 1]))]
            layers[i + 1] = np.tanh(np.dot(matrix, layers[i]) + bias)

        return 2 * layers[-1]
        # should_negate = -1 if layers[-1][1] > 0 else 1
        # return [layers[-1][0] * should_negate * 2.1]  # usually output something in the range -2..2


class CosineTransformPendulumPolicy(ParameterCreatingPolicy):
    """
    A policy that uses a neural net to make decisions
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        y, x, thetadot = observation

        max_speed = 8
        x = (atan2(x, y) + tau / 2) / tau
        y = (thetadot + max_speed) / (max_speed * 2)

        frequencies = np.array([0, tau / 2, tau, tau * 2])
        cos_f_x = np.cos(frequencies * x)
        cos_f_y = np.cos(frequencies * y)

        out = 0
        for cx, cy in itertools.product(cos_f_x, cos_f_y):
            out += p(0) * cx * cy
        # for f_x, f_y in itertools.product(frequencies, frequencies):
        #     out += p() * cos(f_x * x) * cos(f_y * y)
        return [np.clip(out, -1, 1) * 2]


class RecurrantNeuralNetPendulumPolicy(ParameterCreatingPolicy):
    """
    A policy that uses a neural net to make decisions
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.prev_out = np.zeros(shape=(2,))

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        layers = [
            np.hstack([observation, self.prev_out]),
            np.zeros(shape=(6,)),
            np.zeros(shape=(len(self.prev_out),)),
        ]
        connections = []
        for i in range(len(layers) - 1):
            matrix = np.array(
                [
                    [p(0) for _ in range(len(layers[i]))]
                    for _ in range(len(layers[i + 1]))
                ]
            )
            bias = [p(0) for _ in range(len(layers[i + 1]))]
            layers[i + 1] = np.tanh(np.dot(matrix, layers[i]) + bias)

        self.prev_out = layers[-1]
        return 2 * layers[-1][:1]
        # should_negate = -1 if layers[-1][1] > 0 else 1
        # return [layers[-1][0] * should_negate * 2.1]  # usually output something in the range -2..2


class CartpolePolicy(ParameterCreatingPolicy):
    """
    A policy to solve CartPole-v0
    """

    def act(self, observation):
        self.current_parameter = 0
        p = self.param  # shorthand for creating or reading parameters.

        x, x_dot, theta, theta_dot = observation

        go_right = (p(0) * x + p(0) * x_dot + p(0) * theta + p(0) * theta_dot) > 0
        return 1 if go_right else 0


class CartpolePIDPolicy(ParameterCreatingPolicy):
    """
    A PID controller for CartPole-v0 based on
    https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c
    without knowing the PID values.
    """

    desired_state = np.array([0, 0, 0, 0])
    desired_mask = np.array([0, 0, 1, 0])

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
        pid = np.dot(
            P * error + I * self.integral + D * self.derivative, self.desired_mask
        )

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        return action


def get_policy_class(policy_id):
    policy_id_to_policy_class = {
        "EnergyPendulumPolicy": EnergyPendulumPolicy,
        "PiecewisePendulumPolicy": PiecewisePendulumPolicy,
        "DecisionTreePendulumPolicy": DecisionTreePendulumPolicy,
        "CompositePendulumPolicy": CompositePendulumPolicy,
        "NeuralNetPendulumPolicy": NeuralNetPendulumPolicy,
        "CartpolePolicy": CartpolePolicy,
        "CartpolePIDPolicy": CartpolePIDPolicy,
        "RecurrantNeuralNetPendulumPolicy": RecurrantNeuralNetPendulumPolicy,
        "CodeGolfPendulumPolicy": CodeGolfPendulumPolicy,
        "CosineTransformPendulumPolicy": CosineTransformPendulumPolicy,
    }
    if policy_id not in policy_id_to_policy_class:
        raise NotImplementedError(f"No policy_id not found: {repr(policy_id)}")
    return policy_id_to_policy_class[policy_id]
