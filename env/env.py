import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
TASK_MAX_VALUES = {'psi': 340.0 * 2.0, 'b': 10.0}

def generate_task(rng):
    # psi ~ N(270, 50), b ~ N(5,1), tau ~ N(15,3)
    psi = max(1.0, float(rng.normal(loc=270.0, scale=50.0)))
    b = max(0.1, float(rng.normal(loc=5.0, scale=1.0)))
    tau = max(1.0, float(rng.normal(loc=15.0, scale=3.0)))
    return {'psi': psi, 'b': b, 'tau': tau, 'origin': 0}

def plot_rewards(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

import gym
from gym import spaces
#from utils import generate_task, TASK_MAX_VALUES

class FederatedEdgeEnv(gym.Env):
    """Gym-like evironment for the federated edge dispatching problem.

    State vector (dimension = 3*h + 16) per the paper:
      Z (h)      - cluster average utilization
      Gamma (h)  - cluster average allocated resources
      Status (h) - cluster status (1/0)
      Qhat (3)   - task encoding: TLi, TSi, TPI(3) -> in total 5? (we encode as 5)
      W (7)      - day-of-week one-hot
      D (4)      - part-of-day one-hot

    Action: choose a cluster index (0..h-1)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, edge_nodes, h_clusters, seed=None):
        super().__init__()
        self.edge_nodes = edge_nodes  # list of dicts with node properties
        self.n_nodes = len(edge_nodes)
        self.h = h_clusters
        self.rng = np.random.RandomState(seed)

        # action: choose cluster
        self.action_space = spaces.Discrete(self.h)

        # compute state dim = 3*h+16 per paper; we'll use 3*h + 16
        self.state_dim = 3 *self.h + 16
        self.observation_space =spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.cluster_map = None  # list of lists: cluster -> node indices

        # runtime
        self.current_task = None
        self.time_step = 0

    def set_clusters(self, cluster_map):
        self.cluster_map = cluster_map

    def reset(self):
        for n in self.edge_nodes:
            n['utilization'] = self.rng.uniform(0.0, 0.3)  # start lightly loaded
            n['allocated'] = self.rng.uniform(0.0, 0.2)
            n['status'] = 1
        self.time_step = 0
        self.current_task = generate_task(self.rng)
        return self._get_state()

    def step(self, action, omega=0.5):
        assert self.cluster_map is not None, "cluster_map must be set"
        chosen_cluster = int(action)

        cluster_nodes = self.cluster_map[chosen_cluster]
        node_idx = min(cluster_nodes, key=lambda i: self.edge_nodes[i]['utilization'])
        node = self.edge_nodes[node_idx]

        # compute reward (profit + delay) using the formulas in the paper
        psi, b, tau = self.current_task['psi'], self.current_task['b'], self.current_task['tau']

        local_node = self.edge_nodes[self.current_task['origin']]
        if node_idx == self.current_task['origin']:
            # local execution
            beta = local_node['allocated'] if local_node['allocated'] > 0 else 0.1
            cpi = local_node['cp']
            chi = local_node['chi']
            pi = local_node['price']
            V = pi * psi - (beta * cpi * chi)
            U = np.exp(-psi / (beta * cpi + 1e-8))
        else:
            cmir = min(local_node['cm'].get(node_idx, 1.0), 1.0)
            nu = local_node['nu']
            pr = node['price']
            V = local_node['price'] * psi - (cmir * nu + pr * psi)
            beta_r = node['allocated'] if node['allocated'] > 0 else 0.1
            cpr = node['cp']
            U = np.exp(-(psi / (beta_r * cpr + 1e-8) + b / (cmir + 1e-8)))

        reward = omega * V + (1.0 - omega) * U
        
        node['utilization'] += (psi / (node['cp'] + 1e-8)) * 0.01
        node['allocated'] = min(1.0, node['allocated'] + 0.01)

        self.time_step += 1
        done = False
        self.current_task = generate_task(self.rng)
        info = {
              'node_idx': node_idx,
              'reward_components': {
                  'V': V,
                  'U': U,
                  'psi': psi,
                  'b': b,
                  'tau': tau
              }
        }

        return self._get_state(), float(reward), done, info

    def _get_state(self):
        # build Z, Gamma, Status for each cluster
        Z = np.zeros(self.h)
        G = np.zeros(self.h)
        S = np.zeros(self.h)
        for c in range(self.h):
            nodes = self.cluster_map[c]
            if len(nodes) == 0:
                Z[c] = 0.0
                G[c] = 0.0
                S[c] = 0.0
            else:
                utilizations = [self.edge_nodes[i]['utilization'] for i in nodes]
                allocs =[self.edge_nodes[i]['allocated'] for i in nodes]
                statuses = [self.edge_nodes[i]['status'] for i in nodes]
                Z[c] = np.mean(utilizations)
                G[c] = np.mean(allocs)
                S[c] = 1.0 if np.any(np.array(statuses) > 0) else 0.0

        psi, b, tau = self.current_task['psi'], self.current_task['b'], self.current_task['tau']
        TLi = psi /TASK_MAX_VALUES['psi']
        TSi = b /TASK_MAX_VALUES['b']
        if tau <= 10:
            TPI = [1, 0, 0]
        elif tau < 20:
            TPI =[0, 1, 0]
        else:
            TPI = [0, 0, 1]

        Qhat = np.array([TLi, TSi] + TPI)
        W = np.zeros(7)
        W[self.time_step % 7] = 1
        D = np.zeros(4)
        D[(self.time_step // 6) % 4] = 1

        state = np.concatenate([Z, G, S, Qhat, W, D]).astype(np.float32)
        if state.shape[0] != self.state_dim:
            arr = np.zeros(self.state_dim, dtype=np.float32)
            arr[:state.shape[0]] = state[:min(state.shape[0], self.state_dim)]
            state = arr
        return state

    def render(self, mode='human'):
        print(f"Time {self.time_step}: sample task {self.current_task}")
