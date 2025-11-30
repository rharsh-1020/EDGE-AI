import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from models import Actor, Critic
from clustering import cluster_edges
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
            "chi": float(rng.uniform(0.005, 0.01)),
            "nu": float(rng.uniform(0.001, 0.005)),
            "price": float(rng.uniform(0.20, 0.25)),
            "cm": cm,
            "utilization": 0.0,
            "allocated": 0.0,
            "status": 1,
        }
        nodes.append(node)
    return nodes

def run_baseline(env, policy_func, episodes=100, steps=100, omega=0.5):
    rewards = []
    env.rng = np.random.RandomState(42)
    for _ in range(episodes):
        env.reset()
        ep_reward = 0
        for _ in range(steps):
            action = policy_func(env)
            _, r, _, _ = env.step(action, omega=omega)
            ep_reward += r
        rewards.append(ep_reward)
    return rewards

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

    actor_opt = optim.Adam(actor.parameters(), lr=5e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.95

    reward_log = []
    profit_log = []
    delay_log = []
    psi_log = []
    b_log = []

    for ep in trange(episodes, desc=f"Training m={n_nodes}"):
        state = env.reset()
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        ep_total_reward = 0

        for _ in range(steps):
            with torch.no_grad():
                probs = actor(state_t.unsqueeze(0)).cpu().numpy().squeeze(0)

            action = np.random.choice(len(probs), p=probs)
            next_state, reward, _, info = env.step(action, omega=omega)

            ep_total_reward += reward
            profit_log.append(info['V'])
            delay_log.append(info['U'])
            psi_log.append(info['psi'])
            b_log.append(info['b'])

            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
            value = critic(state_t.unsqueeze(0))
            next_value = critic(next_state_t.unsqueeze(0)).detach()

            td_target = reward + gamma * next_value
            delta = td_target - value

            critic_opt.zero_grad()
            delta.pow(2).mean().backward()
            critic_opt.step()

            actor_opt.zero_grad()
            curr_probs = actor(state_t.unsqueeze(0)).squeeze(0)
            logp = torch.log(curr_probs[action] + 1e-8)
            actor_loss = -logp * delta.detach()
            actor_loss.backward()
            actor_opt.step()

            state_t = next_state_t

        reward_log.append(ep_total_reward)

    ra_rewards = run_baseline(env, random_policy, episodes=100, steps=steps, omega=omega)
    ga_rewards = run_baseline(env, greedy_policy, episodes=100, steps=steps, omega=omega)

    return reward_log, ra_rewards, ga_rewards, profit_log, delay_log, psi_log, b_log

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r20, ra20, ga20, V20, U20, psi20, b20 = train_and_collect_logs(
        seed=42, n_nodes=20, h_clusters=3, episodes=300, steps=50, omega=0.5
    )

    r80, ra80, ga80, V80, U80, psi80, b80 = train_and_collect_logs(
        seed=42, n_nodes=80, h_clusters=7, episodes=300, steps=50, omega=0.5
    )

    plot_convergence_two(r20, r80, f"{RESULTS_DIR}/Fig3_convergence.png")

    plot_policy_comparison_three(
        r20[-100:], ga20, ra20,
        f"{RESULTS_DIR}/Fig4_comparison_m20.png",
        "Policy Comparison (m=20)"
    )

    plot_policy_comparison_three(
        r80[-100:], ga80, ra80,
        f"{RESULTS_DIR}/Fig4_comparison_m80.png",
        "Policy Comparison (m=80)"
    )

    plot_task_length_effect_paper(
        psi80, V80, U80,
        f"{RESULTS_DIR}/Fig5_profit.png",
        f"{RESULTS_DIR}/Fig5_delay.png"
    )

    plot_task_size_effect_paper(
        b80, V80, U80,
        f"{RESULTS_DIR}/Fig6_profit.png",
        f"{RESULTS_DIR}/Fig6_delay.png"
    )

    print("Done. Check /RESULT/ folder.")
