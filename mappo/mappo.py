import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mappo.models import ContinuousActor, CentralizedCritic, ActorONNXWrapper
from mappo.buffer import MultiAgentRolloutBuffer


class MAPPOTrainer:
    """
    Treinador MAPPO (Multi-Agent PPO) com CTDE (Centralized Training, Decentralized Execution):
    - Ator compartilhado (com identificadores One-Hot) ou atores individuais.
    - Crítico Centralizado observando o estado global s.
    - PPO Clipped Objective com Normalização de Vantagem.
    - Value Clipping e Huber Loss para estabilidade.
    - Suporte a salvamento em .pt e exportação simultânea em .onnx, com suporte a continuação de treino (resume).
    """

    def __init__(
        self,
        agents: List[str],
        obs_dim: int = 35,
        state_dim: int = 53,
        act_dim: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 5,
        batch_size: int = 128,
        use_shared_actor: bool = True,
        device: str = "cpu",
    ):
        self.agents = agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.use_shared_actor = use_shared_actor
        self.device = device

        # Redes Neurais
        if use_shared_actor:
            self.actor = ContinuousActor(obs_dim, act_dim).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        else:
            self.actors = {a: ContinuousActor(obs_dim, act_dim).to(device) for a in agents}
            self.actor_optimizers = {
                a: optim.Adam(self.actors[a].parameters(), lr=lr_actor, eps=1e-5) for a in agents
            }

        self.critic = CentralizedCritic(state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def get_actions(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False):
        """Coleta ações descentralizadas para cada agente."""
        actions_dict = {}
        log_probs_dict = {}

        with torch.no_grad():
            for a in self.agents:
                obs_t = torch.as_tensor(obs_dict[a], device=self.device, dtype=torch.float32)
                if self.use_shared_actor:
                    act, log_p, _ = self.actor.get_action(obs_t, deterministic=deterministic)
                else:
                    act, log_p, _ = self.actors[a].get_action(obs_t, deterministic=deterministic)
                actions_dict[a] = act
                log_probs_dict[a] = log_p

        return actions_dict, log_probs_dict

    def get_value(self, state: np.ndarray):
        """Avalia o valor do estado global com o Crítico Centralizado."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            value = self.critic(state_t)
        return value

    def train_step(self, buffer: MultiAgentRolloutBuffer) -> Dict[str, float]:
        """Executa a atualização de parâmetros da política e do crítico centralizado."""
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.ppo_epochs):
            data_generator = buffer.get_generator(self.batch_size)

            for batch in data_generator:
                (
                    b_obs,
                    b_states,
                    b_actions,
                    b_log_probs,
                    b_values,
                    b_returns,
                    b_advs,
                ) = batch

                # 1. Otimização do Crítico Centralizado
                current_values = self.critic(b_states)
                v_clipped = b_values + (current_values - b_values).clamp(-self.clip_param, self.clip_param)
                vf_loss_1 = (current_values - b_returns).pow(2)
                vf_loss_2 = (v_clipped - b_returns).pow(2)
                critic_loss = 0.5 * torch.max(vf_loss_1, vf_loss_2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 2. Otimização do Ator (ou Atores)
                if self.use_shared_actor:
                    actor_loss_sum = 0.0
                    ent_sum = 0.0
                    for a in self.agents:
                        new_log_prob, ent = self.actor.evaluate_actions(b_obs[a], b_actions[a])
                        ratio = torch.exp(new_log_prob - b_log_probs[a])
                        surr1 = ratio * b_advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advs
                        actor_loss = -torch.min(surr1, surr2).mean()
                        actor_loss_sum += actor_loss - self.entropy_coef * ent.mean()
                        ent_sum += ent.mean().item()

                    self.actor_optimizer.zero_grad()
                    actor_loss_sum.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    total_actor_loss += actor_loss_sum.item()
                    total_entropy += ent_sum / len(self.agents)
                else:
                    for a in self.agents:
                        new_log_prob, ent = self.actors[a].evaluate_actions(b_obs[a], b_actions[a])
                        ratio = torch.exp(new_log_prob - b_log_probs[a])
                        surr1 = ratio * b_advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advs
                        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()

                        self.actor_optimizers[a].zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(self.actors[a].parameters(), self.max_grad_norm)
                        self.actor_optimizers[a].step()

                        total_actor_loss += actor_loss.item()
                        total_entropy += ent.mean().item()

                total_critic_loss += critic_loss.item()
                update_count += 1

        return {
            "actor_loss": total_actor_loss / max(update_count, 1),
            "critic_loss": total_critic_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
        }

    def export_onnx(self, onnx_filepath: str):
        """Exporta o ator treinado para formato ONNX para implantação rápida."""
        os.makedirs(os.path.dirname(onnx_filepath) or ".", exist_ok=True)
        dummy_input = torch.randn(1, self.obs_dim, dtype=torch.float32, device="cpu")

        if self.use_shared_actor:
            onnx_wrapper = ActorONNXWrapper(self.actor).to("cpu")
            onnx_wrapper.eval()
            with torch.no_grad():
                torch.onnx.export(
                    onnx_wrapper,
                    dummy_input,
                    onnx_filepath,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["observation"],
                    output_names=["action"],
                    dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
                )
            self.actor.to(self.device)
        else:
            base, ext = os.path.splitext(onnx_filepath)
            for a in self.agents:
                agent_onnx = f"{base}_{a}{ext}"
                onnx_wrapper = ActorONNXWrapper(self.actors[a]).to("cpu")
                onnx_wrapper.eval()
                with torch.no_grad():
                    torch.onnx.export(
                        onnx_wrapper,
                        dummy_input,
                        agent_onnx,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=["observation"],
                        output_names=["action"],
                        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
                    )
                self.actors[a].to(self.device)


    def save(self, filepath: str, extra_info: dict = None, export_onnx: bool = True):
        """
        Salva o checkpoint em formato PyTorch (.pt) com pesos e estado dos otimizadores
        e exporta simultaneamente o modelo em formato ONNX (.onnx).
        """
        if not filepath.endswith(".pt"):
            pt_path = filepath + ".pt"
        else:
            pt_path = filepath

        os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)

        checkpoint = {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "use_shared_actor": self.use_shared_actor,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
        }

        if self.use_shared_actor:
            checkpoint["actor"] = self.actor.state_dict()
            checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        else:
            checkpoint["actors"] = {a: self.actors[a].state_dict() for a in self.agents}
            checkpoint["actor_optimizers"] = {a: self.actor_optimizers[a].state_dict() for a in self.agents}

        if extra_info:
            checkpoint.update(extra_info)

        torch.save(checkpoint, pt_path)

        if export_onnx:
            onnx_path = pt_path[:-3] + ".onnx"
            try:
                self.export_onnx(onnx_path)
            except Exception as e:
                print(f"[!] Aviso: Falha ao exportar ONNX ({e})")

    def load(self, filepath: str, load_optimizers: bool = True) -> dict:
        """
        Carrega os pesos do modelo e otimizadores para execução ou continuação de treino (resume).
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restaurar Crítico
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers and "critic_optimizer" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            except Exception:
                pass

        # Restaurar Ator
        if checkpoint.get("use_shared_actor", True):
            self.actor.load_state_dict(checkpoint["actor"])
            if load_optimizers and "actor_optimizer" in checkpoint:
                try:
                    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
                except Exception:
                    pass
        else:
            for a in self.agents:
                self.actors[a].load_state_dict(checkpoint["actors"][a])
                if load_optimizers and "actor_optimizers" in checkpoint:
                    try:
                        self.actor_optimizers[a].load_state_dict(checkpoint["actor_optimizers"][a])
                    except Exception:
                        pass

        return checkpoint
