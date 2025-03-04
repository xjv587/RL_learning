from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################

    c = np.zeros((env_spec.nS, env_spec.nA))
    for traj in trajs:
        g = 0
        w = 1
        for step in range(len(traj)-1, -1, -1):
            state = traj[step][0]
            action = traj[step][1]
            reward = traj[step][2]
            g = env_spec.gamma * g + reward
            c[state][action] += 1
            initQ[state][action] = initQ[state][action] + (w / c[state][action]) * (g - initQ[state][action])
            if bpi.action_prob(state, action) > 0:
                w *= (pi.action_prob(state, action) / bpi.action_prob(state, action))
            if w == 0:
                break
            
    Q = initQ
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################

    c = np.zeros((env_spec.nS, env_spec.nA))
    for traj in trajs:
        g = 0
        w = 1
        for step in range(len(traj)-1, -1, -1):
            state = traj[step][0]
            action = traj[step][1]
            reward = traj[step][2]
            g = env_spec.gamma * g + reward
            c[state][action] += w
            initQ[state][action] = initQ[state][action] + (w / c[state][action]) * (g - initQ[state][action])
            if bpi.action_prob(state, action) > 0:
                w *= (pi.action_prob(state, action) / bpi.action_prob(state, action))
            if w == 0:
                break
            
    Q = initQ
    return Q
