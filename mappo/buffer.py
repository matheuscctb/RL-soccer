from typing import Generator, Dict, List
import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """
    Buffer de Rollout para MAPPO com suporte a CTDE (Treinamento Centralizado, Execução Descentralizada).
    Armazena trajetórias para múltiplos agentes e múltiplos ambientes em paralelo.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        agents: List[str],
        obs_dim: int,
        state_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.agents = agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Buffers descentralizados para cada agente
        self.obs = {
            a: torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
            for a in agents
        }
        self.actions = {
            a: torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float32, device=device)
            for a in agents
        }
        self.action_log_probs = {
            a: torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)
            for a in agents
        }
        self.rewards = {
            a: torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)
            for a in agents
        }
        self.dones = {
            a: torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)
            for a in agents
        }

        # Buffers centralizados (Crítico Centralizado)
        self.states = torch.zeros((num_steps, num_envs, state_dim), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=device)

        self.step = 0

    def insert(
        self,
        obs_dict: Dict[str, np.ndarray],
        state: np.ndarray,
        actions_dict: Dict[str, torch.Tensor],
        log_probs_dict: Dict[str, torch.Tensor],
        values: torch.Tensor,
        rewards_dict: Dict[str, float],
        dones_dict: Dict[str, bool],
    ):
        """Insere uma transição completa do rollout."""
        for a in self.agents:
            self.obs[a][self.step] = torch.as_tensor(obs_dict[a], device=self.device, dtype=torch.float32)
            self.actions[a][self.step] = actions_dict[a]
            self.action_log_probs[a][self.step] = log_probs_dict[a]
            
            # Tratar rewards e dones
            rew = rewards_dict[a] if isinstance(rewards_dict[a], (list, np.ndarray)) else [rewards_dict[a]]
            self.rewards[a][self.step] = torch.as_tensor(rew, device=self.device, dtype=torch.float32).view(-1, 1)
            
            done = dones_dict[a] if isinstance(dones_dict[a], (list, np.ndarray)) else [dones_dict[a]]
            self.dones[a][self.step] = torch.as_tensor(done, device=self.device, dtype=torch.float32).view(-1, 1)

        self.states[self.step] = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        self.values[self.step] = values

        self.step = (self.step + 1) % self.num_steps

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor):
        """
        Calcula o Generalized Advantage Estimation (GAE) centralizado.
        """
        gae = 0
        # Média da recompensa cooperativa compartilhada
        mean_rewards = torch.stack([self.rewards[a] for a in self.agents]).mean(dim=0)
        
        # Obter terminação geral (se qualquer agente deu done)
        any_dones = torch.stack([self.dones[a] for a in self.agents]).max(dim=0)[0]

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_v = next_value
            else:
                next_non_terminal = 1.0 - any_dones[step + 1]
                next_v = self.values[step + 1]

            delta = mean_rewards[step] + self.gamma * next_v * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
            self.returns[step] = self.advantages[step] + self.values[step]

    def get_generator(self, batch_size: int) -> Generator:
        """Gera mini-batches aleatórios para otimização PPO."""
        total_samples = self.num_steps * self.num_envs
        indices = np.random.permutation(total_samples)

        # Flatten buffers
        flat_states = self.states.view(-1, self.state_dim)
        flat_values = self.values.view(-1, 1)
        flat_returns = self.returns.view(-1, 1)
        flat_advs = self.advantages.view(-1, 1)

        # Normalizar vantagens
        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)

        flat_obs = {a: self.obs[a].view(-1, self.obs_dim) for a in self.agents}
        flat_actions = {a: self.actions[a].view(-1, self.act_dim) for a in self.agents}
        flat_log_probs = {a: self.action_log_probs[a].view(-1, 1) for a in self.agents}

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            batch_obs = {a: flat_obs[a][batch_indices] for a in self.agents}
            batch_actions = {a: flat_actions[a][batch_indices] for a in self.agents}
            batch_log_probs = {a: flat_log_probs[a][batch_indices] for a in self.agents}

            yield (
                batch_obs,
                flat_states[batch_indices],
                batch_actions,
                batch_log_probs,
                flat_values[batch_indices],
                flat_returns[batch_indices],
                flat_advs[batch_indices],
            )
