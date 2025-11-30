#%%writefile env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
MARKUP_FACTOR = 1.5 

TASK_MAX_VALUES = {'psi': 340.0 * 2.0, 'b': 10.0}

def generate_task(rng):
    psi = max(1.0, float(rng.normal(loc=270.0, scale=50.0)))
    b = max(0.1, float(rng.normal(loc=5.0, scale=1.0)))
    tau = max(1.0, float(rng.normal(loc=15.0, scale=3.0)))
    return {'psi': psi, 'b': b, 'tau': tau, 'origin': 0}

class FederatedEdgeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, edge_nodes, h_clusters, seed=None):
        super().__init__()
        self.edge_nodes = edge_nodes
        self.n_nodes = len(edge_nodes)
        self.h = h_clusters
        self.rng = np.random.RandomState(seed)
        self.action_space = spaces.Discrete(self.h)
        self.state_dim = 3 * self.h + 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.cluster_map = None
        self.current_task = None
        self.time_step = 0

    def set_clusters(self, cluster_map):
        self.cluster_map = cluster_map

    def reset(self):
        for n in self.edge_nodes:
            n['utilization'] = self.rng.uniform(0.0, 0.3)
            n['allocated'] = self.rng.uniform(0.0, 0.2)
            n['status'] = 1
        self.time_step = 0
        self.current_task = generate_task(self.rng)
        return self._get_state()

    def step(self, action, omega=0.5):
        assert self.cluster_map is not None, "cluster_map must be set"
        chosen_cluster = int(action)
        cluster_nodes = self.cluster_map[chosen_cluster]
        
        if not cluster_nodes:
            node_idx = self.rng.choice(range(self.n_nodes))
        else:
            node_idx = min(cluster_nodes, key=lambda i: self.edge_nodes[i]['utilization'])
            
        node = self.edge_nodes[node_idx]
        psi, b, tau = self.current_task['psi'], self.current_task['b'], self.current_task['tau']
        local_node = self.edge_nodes[self.current_task['origin']]
        revenue = (local_node['price'] * MARKUP_FACTOR) * psi

        if node_idx == self.current_task['origin']:
            # Local Execution
            beta = max(local_node['allocated'], 0.1)
            cpi = local_node['cp']
            chi = local_node['chi']
            
            # FIX: Cost = Total Cycles (psi) * Cost per Cycle (chi)
            # This ensures cost scales with task length.
            cost = psi * chi
            
            V = revenue - cost
            raw_delay = psi / (beta * cpi + 1e-8)
        else:
            cmir = min(local_node['cm'].get(node_idx, 1.0), 1.0)
            nu = local_node['nu']
            pr = node['price']
        
            offloading_cost = (cmir * nu) + (pr * psi)
            
            V = revenue - offloading_cost
            
            beta_r = max(node['allocated'], 0.1)
            cpr = node['cp']
            raw_delay = (psi / (beta_r * cpr + 1e-8)) + (b / (cmir + 1e-8))
        U_reward = np.exp(-raw_delay) 

        reward = omega * V + (1.0 - omega) * U_reward
        
        node['utilization'] += (psi / (node['cp'] + 1e-8)) * 0.01
        node['allocated'] = min(1.0, node['allocated'] + 0.01)
        self.time_step += 1
        
        info = {
            'V': V,          
            'U': raw_delay,  
            'psi': psi,
            'b': b
        }
        
        self.current_task = generate_task(self.rng)
        return self._get_state(), float(reward), False, info

    def _get_state(self):
        Z, G, S = np.zeros(self.h), np.zeros(self.h), np.zeros(self.h)
        for c in range(self.h):
            nodes = self.cluster_map[c]
            if nodes:
                Z[c] = np.mean([self.edge_nodes[i]['utilization'] for i in nodes])
                G[c] = np.mean([self.edge_nodes[i]['allocated'] for i in nodes])
                S[c] = 1.0
        
        psi, b, tau = self.current_task['psi'], self.current_task['b'], self.current_task['tau']
        TLi = psi / TASK_MAX_VALUES['psi']
        TSi = b / TASK_MAX_VALUES['b']
        TPI = [1,0,0] if tau <= 10 else ([0,1,0] if tau < 20 else [0,0,1])
        W = np.zeros(7); W[self.time_step % 7] = 1
        D = np.zeros(4); D[(self.time_step // 6) % 4] = 1
        
        state = np.concatenate([Z, G, S, [TLi, TSi], TPI, W, D]).astype(np.float32)
        return state
