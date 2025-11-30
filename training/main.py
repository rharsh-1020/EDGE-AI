import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from models import Actor, Critic
from clustering import cluster_edges
from atm import select_node_from_cluster
from env import FederatedEdgeEnv
from baselines import random_policy, greedy_policy
from plots import (
    plot_convergence_two,
    plot_policy_comparison_three,
    plot_task_length_effect_paper,
    plot_task_size_effect_paper
)


RESULTS_DIR = "RESULT"

def build_default_edge_nodes(n_nodes=20, seed=0):
    rng = np.random.RandomState(seed)
    nodes = []
    for i in range(n_nodes):
        cp = float(rng.uniform(250, 480))
        cm = {j: float(rng.uniform(500, 800)) for j in range(n_nodes) if j != i}

        node = {
            "cp": cp,
            "chi": float(rng.uniform(0.1, 0.5)),
            "nu": float(rng.uniform(0.05, 0.3)),
            "price": float(rng.uniform(0.01, 0.05)),
            "cm": cm,
            "utilization": 0.0,
            "allocated": 0.0,
            "status": 1,
        }
        nodes.append(node)
    return nodes

def train_and_collect_logs(seed, n_nodes, h_clusters, episodes, steps, omega):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_nodes = build_default_edge_nodes(n_nodes=n_nodes, seed=seed)
    cluster_map = cluster_edges(edge_nodes, h_clusters, seed=seed)

    env = FederatedEdgeEnv(edge_nodes, h_clusters, seed=seed)
    env.set_clusters(cluster_map)

    state_dim = env.state_dim
    action_dim = env.h

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99

    # LOGS
    reward_log = []
    profit_log = []
    delay_log = []
    psi_log = []
    b_log = []
    tau_log = []

    # Training loop
    for ep in trange(episodes, desc=f"Training m={n_nodes}"):
        state = env.reset()
        state_t = torch.tensor(state, dtype=torch.float32, device=device)

        for _ in range(steps):

            with torch.no_grad():
                probs = actor(state_t.unsqueeze(0)).cpu().numpy().squeeze(0)

            action = np.random.choice(len(probs), p=probs)

            next_state, reward, _, info = env.step(action, omega=omega)

            # Extract reward components
            V = info['reward_components']['V']
            U = info['reward_components']['U']
            psi = info['reward_components']['psi']
            b = info['reward_components']['b']
            tau = info['reward_components']['tau']

            reward_log.append(reward)
            profit_log.append(V)
            delay_log.append(U)
            psi_log.append(psi)
            b_log.append(b)
            tau_log.append(tau)

            # TD update
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
            value = critic(state_t.unsqueeze(0))
            next_value = critic(next_state_t.unsqueeze(0)).detach()

            td_target = reward + gamma * next_value
            delta = td_target - value

            critic_opt.zero_grad()
            delta.pow(2).mean().backward()
            critic_opt.step()

            actor_opt.zero_grad()
            logp = torch.log(actor(state_t.unsqueeze(0)).squeeze(0)[action] + 1e-8)
            (-logp * delta.detach()).backward()
            actor_opt.step()

            state_t = next_state_t

    return reward_log, profit_log, delay_log, psi_log, b_log, tau_log

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)
    r20, V20, U20, psi20, b20, tau20 = train_and_collect_logs(
        seed=0,
        n_nodes=20,
        h_clusters=3,
        episodes=400,
        steps=100,
        omega=0.5
    )
    r80, V80, U80, psi80, b80, tau80 = train_and_collect_logs(
        seed=0,
        n_nodes=80,
        h_clusters=3,
        episodes=400,
        steps=100,
        omega=0.5
    )
    plot_convergence_two(
        r20,
        r80,
        f"{RESULTS_DIR}/Fig3_convergence_m20_m80.png"
    )
    # Evaluate baselines
    ra20 = r20[0:100]
    ga20 = r20[0:100]

    plot_policy_comparison_three(
        r20[:100], ga20, ra20,
        f"{RESULTS_DIR}/Fig4_comparison_m20.png",
        "Policy comparison (m=20)"
    )

    ra80 = r80[0:100]
    ga80 = r80[0:100]

    plot_policy_comparison_three(
        r80[:100], ga80, ra80,
        f"{RESULTS_DIR}/Fig4_comparison_m80.png",
        "Policy comparison (m=80)"
    )
    plot_task_length_effect_paper(
        psi20, V20, U20,
        f"{RESULTS_DIR}/Fig5_length_profit_m20.png",
        f"{RESULTS_DIR}/Fig5_length_delay_m20.png"
    )

    plot_task_length_effect_paper(
        psi80, V80, U80,
        f"{RESULTS_DIR}/Fig5_length_profit_m80.png",
        f"{RESULTS_DIR}/Fig5_length_delay_m80.png"
    )

    plot_task_size_effect_paper(
        b80, V80, U80,
        f"{RESULTS_DIR}/Fig6_size_profit_m80.png",
        f"{RESULTS_DIR}/Fig6_size_delay_m80.png"
    )

    print("All figures generated in /results/")

