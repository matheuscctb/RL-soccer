import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Inicialização ortogonal dos pesos das camadas neurais."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ContinuousActor(nn.Module):
    """
    Ator Descentralizado para Controle Contínuo:
    - Entrada: Observação local do agente o_i (dim: 35)
    - Saída: Média e desvio padrão calibrado para ações [v_x, v_y, v_theta, kick_x]
    - Execução 100% descentralizada (não precisa do crítico durante o jogo).
    """

    def __init__(self, obs_dim: int = 35, act_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Cabeça para média da ação
        self.mu_head = init_layer(nn.Linear(hidden_dim, act_dim), std=0.01)

        # Parâmetro inicializado em -0.5 (std ~ 0.60) para exploração suave sem explodir
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = torch.tanh(self.mu_head(x))
        # std restrito a [0.135, 1.0] para evitar saturação e perda de gradiente
        std = torch.exp(self.log_std.clamp(-2.0, 0.0))
        return mu, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Amostra uma ação e retorna ação, log_prob e valor de entropia."""
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = dist.sample()

        action_clamped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action_clamped).sum(dim=-1, keepdim=True)
        return action_clamped, log_prob, dist.entropy().sum(dim=-1, keepdim=True)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Avalia ações armazenadas para cálculo de perda PPO."""
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class ActorONNXWrapper(nn.Module):
    """
    Wrapper limpo para exportação determinística da política em formato ONNX.
    Recebe observações locais (Batch, 35) e retorna ações diretas (Batch, 4) em [-1, 1].
    """

    def __init__(self, actor: ContinuousActor):
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.actor.net(observation)
        return torch.tanh(self.actor.mu_head(x))


class CentralizedCritic(nn.Module):
    """
    Crítico Centralizado (CTDE) para MAPPO:
    - Entrada: Estado Global s (dim: 53) contendo o campo completo e todos os robôs
    - Saída: Valor escalar do estado V(s)
    - Usado APENAS durante o treino para guiar a cooperação e eliminar não-estacionariedade.
    """

    def __init__(self, state_dim: int = 53, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            init_layer(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim // 2, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
