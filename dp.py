from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class Pi(Policy):
    def __init__(self, nS, nA):
        super().__init__()
        self.p = np.zeros((nS, nA))

    def action_prob(self, state:int, action:int) -> float:
        return self.p[state][action]
    
    def action(self, state:int) -> int:
        return np.argmax(self.p[state])


def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################

    Q = np.zeros((env.spec.nS, env.spec.nA))

    while True: 
        delta = 0.0
        for state in range(env.spec.nS):
            v = initV[state]
            sum = 0.0
            for action in range(env.spec.nA):
                probAS = pi.action_prob(state, action)
                temp_sum = 0.0 
                for sp in range(env.spec.nS):
                    r = env.R[state, action, sp]
                    value = r + env.spec.gamma * initV[sp]
                    temp_sum += env.TD[state, action, sp] * value
                sum += probAS * temp_sum
            initV[state] = sum
            delta = max(delta, abs(v - initV[state]))
        if delta < theta:
            break

    while True:
        delta = 0.0
        for state in range(env.spec.nS):
            for action in range(env.spec.nA):
                q = Q[state][action]
                temp_sum = 0.0
                for sp in range(env.spec.nS):
                    r = env.R[state, action, sp]
                    value = r + env.spec.gamma * initV[sp]
                    temp_sum += env.TD[state, action, sp] * value
                Q[state][action] = temp_sum
                delta = max(delta, abs(q - Q[state][action]))
        if delta < theta:
            break
    V = initV
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    pi = Pi(env.spec.nS, env.spec.nA)
    while True:
        delta = 0.0
        for state in range(env.spec.nS):
            maxAV = -float('inf')
            for action in range(env.spec.nA):
                v = initV[state]
                actionV = 0
                for sp in range(env.spec.nS):
                    r = env.R[state, action, sp]
                    actionV += env.TD[state, action, sp] * (r + env.spec.gamma * initV[sp])
                maxAV = max(maxAV, actionV)
            initV[state] = maxAV
            delta = max(delta, abs(v - initV[state]))
        if delta < theta:
            break
    
    values = []
    for state in range(env.spec.nS):
        for action in range(env.spec.nA):
            actionV = 0
            for sp in range(env.spec.nS):
                r = env.R[state, action, sp]
                actionV += env.TD[state, action, sp] * (r + env.spec.gamma * initV[sp])
            values.append(actionV)
        pi.p[state][np.argmax(values)] = 1.0
        values.clear()
    V = initV
    return V, pi
