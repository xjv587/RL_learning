from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class GreedyPolicy(object):
    def __init__(self, Q):
        self._Q = Q

    def action_prob(self, state:int, action:int) -> float:
        if self.action(state) == action:
            return 1
        else:
            return 0
        
    def action(self, state:int) -> int:
        return np.argmax(self._Q[state, :])


def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = initV
    for episode in trajs:
        T = len(episode)
        for t in range(T):
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau, min(tau+n, T)):
                    G += np.power(env_spec.gamma, i-tau-1) * episode[i][2]
                if tau + n < T:
                    G += np.power(env_spec.gamma, n) * V[episode[tau+n][0]]
                V[episode[tau][0]] += alpha * (G - V[episode[tau][0]])
            if tau == T-1:
                break
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    
    Q = initQ
    pi = GreedyPolicy(Q)
    for episode in trajs:
        T = len(episode)
        states, actions, rewards = zip(*[(step[0], step[1], step[2]) for step in episode])
        for t in range(T):
            G = 0
            w = 1
            for k in range(t, min(t + n, T)):
                G += (env_spec.gamma ** (k - t)) * rewards[k]
                if k + 1 < T:
                    G += (env_spec.gamma ** (n)) * Q[states[k + 1], actions[k + 1]]
                tau = t + 1 - n
                if tau >= 0:
                    Q[states[tau], actions[tau]] += alpha * w * (G - Q[states[tau], actions[tau]])
                    if bpi.action_prob(states[tau], actions[tau]) > 0:
                        w *= pi.action_prob(states[tau], actions[tau]) / bpi.action_prob(states[tau], actions[tau])
                    if w == 0:
                        break
    
    pi = GreedyPolicy(Q)
    return Q, pi
