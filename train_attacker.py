import os
import glob
import time
import shutil
import re
import argparse
import multiprocessing as mp
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

import rsoccer_gym
from rsoccer_gym.ssl.ssl_el_cooperation_attacker import SSLELCooperationAttackerEnv
from mappo import MAPPOTrainer, MultiAgentRolloutBuffer


def get_next_treino_id(pt_dir: str) -> int:
    """Descobre o próximo número de treino sequencial analisando arquivos salvos em pt/."""
    existing = glob.glob(os.path.join(pt_dir, "attacker_treino*.pt"))
    max_id = 0
    for f in existing:
        match = re.search(r"attacker_treino(\d+)_", os.path.basename(f), re.IGNORECASE)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento MAPPO para Cooperação de Atacantes SSL-EL com TensorBoard, WandB e ONNX (cooperation_attacker)"
    )
    parser.add_argument("--treino-id", type=int, default=0, help="ID do treino (ex: 1 para treino1, 2 para treino2. 0 = automático)")
    parser.add_argument("--total-timesteps", type=int, default=40_000_000, help="Total de passos de treino (padrão: 40.000.000)")
    parser.add_argument("--num-envs", type=int, default=128, help="Número de ambientes paralelos (padrão otimizado: 128)")
    parser.add_argument("--num-workers", type=int, default=8, help="Número de processos CPU workers (padrão: 8 núcleos)")
    parser.add_argument("--num-steps", type=int, default=100, help="Passos de rollout por atualização (padrão: 100)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Épocas PPO por atualização (padrão: 4)")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamanho do mini-batch PPO (padrão: 256)")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Taxa de aprendizado do Ator")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Taxa de aprendizado do Crítico Central")
    parser.add_argument("--gamma", type=float, default=0.99, help="Fator de desconto")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Fator GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="Parâmetro de clip do PPO")
    parser.add_argument("--entropy-coef", type=float, default=0.001, help="Coeficiente de entropia (padrão: 0.001)")
    parser.add_argument("--save-interval-steps", type=int, default=100_000, help="Intervalo de passos para salvar checkpoint (padrão: 100.000)")
    parser.add_argument("--log-interval", type=int, default=5, help="Intervalo de iterações para log no console/tensorboard/wandb")
    parser.add_argument("--exp-name", type=str, default="cooperation_attacker", help="Nome do experimento")
    parser.add_argument("--save-dir", type=str, default="modelos", help="Diretório onde os modelos e checkpoints serão salvos")
    parser.add_argument("--tensorboard-dir", type=str, default="runs", help="Diretório de logs do TensorBoard (padrão: runs)")
    parser.add_argument("--wandb", action="store_true", default=True, help="Ativar acompanhamento online no Weights & Biases (WandB)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Desativar WandB")
    parser.add_argument("--wandb-project", type=str, default="RL-soccer", help="Nome do projeto no WandB (padrão: RL-soccer)")
    parser.add_argument("--wandb-entity", type=str, default="", help="Entidade/Time do WandB (opcional)")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de execução (padrão: cuda)")
    parser.add_argument("--resume", action="store_true", help="Retomar o treinamento automaticamente a partir do último checkpoint")
    parser.add_argument("--resume-path", type=str, default="", help="Caminho específico de um checkpoint (.pt) para retomar o treino")
    return parser.parse_args()


def _worker_process(remote, parent_remote, count: int):
    """Processo worker isolado de alta performance com buffers pré-alocados."""
    parent_remote.close()
    envs = [SSLELCooperationAttackerEnv() for _ in range(count)]
    agents = SSLELCooperationAttackerEnv.agents
    obs_dim = envs[0].num_local_obs
    state_dim = envs[0].num_state_features

    # Buffers contíguos pré-alocados para evitar alocações dinâmicas a cada passo
    obs_buf = {a: np.zeros((count, obs_dim), dtype=np.float32) for a in agents}
    state_buf = np.zeros((count, state_dim), dtype=np.float32)
    rew_buf = {a: np.zeros(count, dtype=np.float32) for a in agents}
    done_buf = {a: np.zeros(count, dtype=np.float32) for a in agents}

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                results_info = []
                for i, env in enumerate(envs):
                    actions = {a: data[a][i] for a in agents}
                    obs, rew, term, trunc, info = env.step(actions)
                    done = term[agents[0]] or trunc[agents[0]]
                    if done:
                        obs, _ = env.reset()
                    state = env.get_global_state()
                    for a in agents:
                        obs_buf[a][i] = obs[a]
                        rew_buf[a][i] = rew[a]
                        done_buf[a][i] = done
                    state_buf[i] = state
                    results_info.append(info)

                remote.send((
                    {a: obs_buf[a].copy() for a in agents},
                    state_buf.copy(),
                    {a: rew_buf[a].copy() for a in agents},
                    {a: done_buf[a].copy() for a in agents},
                    results_info
                ))
            elif cmd == "reset":
                for i, env in enumerate(envs):
                    obs, _ = env.reset()
                    state = env.get_global_state()
                    for a in agents:
                        obs_buf[a][i] = obs[a]
                    state_buf[i] = state
                remote.send((
                    {a: obs_buf[a].copy() for a in agents},
                    state_buf.copy()
                ))
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


class MultiEnvWrapper:
    """Wrapper síncrono para poucos ambientes locais (baixo overhead em single-core)."""
    def __init__(self, num_envs: int):
        self.envs = [SSLELCooperationAttackerEnv() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.agents = SSLELCooperationAttackerEnv.agents

    def reset(self):
        obs_list = []
        state_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            state_list.append(env.get_global_state())

        batched_obs = {
            a: np.array([obs_list[i][a] for i in range(self.num_envs)], dtype=np.float32)
            for a in self.agents
        }
        batched_states = np.array(state_list, dtype=np.float32)
        return batched_obs, batched_states

    def step(self, actions_dict: dict):
        next_obs_list = []
        next_state_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for i, env in enumerate(self.envs):
            env_actions = {a: actions_dict[a][i].cpu().numpy() for a in self.agents}
            obs, rew, term, trunc, info = env.step(env_actions)

            done = term[self.agents[0]] or trunc[self.agents[0]]
            if done:
                obs, _ = env.reset()

            next_obs_list.append(obs)
            next_state_list.append(env.get_global_state())
            rewards_list.append(rew)
            dones_list.append(done)
            infos_list.append(info)

        batched_obs = {
            a: np.array([next_obs_list[i][a] for i in range(self.num_envs)], dtype=np.float32)
            for a in self.agents
        }
        batched_states = np.array(next_state_list, dtype=np.float32)
        batched_rewards = {
            a: np.array([rewards_list[i][a] for i in range(self.num_envs)], dtype=np.float32)
            for a in self.agents
        }
        batched_dones = {
            a: np.array([dones_list[i] for i in range(self.num_envs)], dtype=np.float32)
            for a in self.agents
        }

        return batched_obs, batched_states, batched_rewards, batched_dones, infos_list

    def close(self):
        for env in self.envs:
            env.close()


class SubprocMultiEnvWrapper:
    """Wrapper multiprocesso paralelo escalável para dezenas, centenas ou milhares de ambientes."""
    def __init__(self, num_envs: int, num_workers: int = 0):
        self.num_envs = num_envs
        self.agents = SSLELCooperationAttackerEnv.agents
        if num_workers <= 0:
            num_workers = min(num_envs, os.cpu_count() or 4)
        self.num_workers = max(1, min(num_workers, num_envs))

        base = num_envs // self.num_workers
        rem = num_envs % self.num_workers
        self.counts = [base + (1 if i < rem else 0) for i in range(self.num_workers)]

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_workers)])
        self.processes = []

        for work_remote, remote, count in zip(self.work_remotes, self.remotes, self.counts):
            p = ctx.Process(target=_worker_process, args=(work_remote, remote, count), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]

        batched_obs = {
            a: np.concatenate([r[0][a] for r in results], axis=0)
            for a in self.agents
        }
        batched_states = np.concatenate([r[1] for r in results], axis=0)
        return batched_obs, batched_states

    def step(self, actions_dict: dict):
        numpy_actions = {a: actions_dict[a].cpu().numpy() for a in self.agents}
        idx = 0
        for remote, count in zip(self.remotes, self.counts):
            worker_actions = {a: numpy_actions[a][idx:idx + count] for a in self.agents}
            remote.send(("step", worker_actions))
            idx += count

        results = [remote.recv() for remote in self.remotes]

        batched_obs = {
            a: np.concatenate([r[0][a] for r in results], axis=0)
            for a in self.agents
        }
        batched_states = np.concatenate([r[1] for r in results], axis=0)
        batched_rewards = {
            a: np.concatenate([r[2][a] for r in results], axis=0)
            for a in self.agents
        }
        batched_dones = {
            a: np.concatenate([r[3][a] for r in results], axis=0)
            for a in self.agents
        }
        infos_list = []
        for r in results:
            infos_list.extend(r[4])

        return batched_obs, batched_states, batched_rewards, batched_dones, infos_list

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=1.0)


def make_vec_env(num_envs: int, num_workers: int = 0):
    """Fábrica para instanciar o melhor wrapper de ambientes paralelos."""
    if num_envs <= 4 or num_workers == 1:
        return MultiEnvWrapper(num_envs)
    return SubprocMultiEnvWrapper(num_envs, num_workers)


def train():
    args = parse_args()
    save_dir = args.save_dir
    pt_dir = os.path.join(save_dir, "pt")
    onnx_dir = os.path.join(save_dir, "onnx")
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Identificação sequencial do treino (ex: treino1, treino2, ...)
    if args.treino_id > 0:
        treino_tag = f"treino{args.treino_id}"
    else:
        next_id = get_next_treino_id(pt_dir)
        treino_tag = f"treino{next_id}"

    # Configuração do TensorBoard
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.tensorboard_dir, f"{treino_tag}_{args.exp_name}_{timestamp_str}")
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Configuração do WandB (Weights & Biases Online Tracking)
    wandb_run = None
    if args.wandb:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                name=f"attacker_{treino_tag}",
                config=vars(args),
                reinit=True,
            )
            wandb_url = getattr(wandb_run, "url", f"https://wandb.ai/{args.wandb_entity or 'kenthope40'}/{args.wandb_project}")
            wandb_status = f"ATIVADO -> {wandb_url}"
        except Exception as e:
            print(f"[!] Aviso: Falha ao inicializar WandB ({e}). Continuando com TensorBoard local.")
            args.wandb = False
            wandb_status = "DESATIVADO (Erro na conexão)"
    else:
        wandb_status = "DESATIVADO (--no-wandb)"

    # Otimizações de GPU CUDA (Ampere Tensor Cores / RTX 3060)
    device_name_str = args.device
    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_name_str = f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM) [CUDA]"
        else:
            print("[!] AVISO: 'cuda' foi selecionado mas não há GPU NVIDIA disponível. Recorrendo a 'cpu'.")
            args.device = "cpu"
            device_name_str = "CPU (Fallback)"

    vec_env = make_vec_env(args.num_envs, args.num_workers)
    workers_info = getattr(vec_env, "num_workers", 1)
    print("=" * 75)
    print(f" 🚀 INICIANDO TREINAMENTO MAPPO: {treino_tag.upper()} ({args.exp_name})")
    print(f" 📂 Modelos (.pt): {pt_dir}/ | Modelos (.onnx): {onnx_dir}/")
    print(f" 💾 Checkpoints Ativos: {ckpt_dir}/ (apagados automaticamente ao final)")
    print(f" 📊 TensorBoard Logs: {tb_log_dir}/")
    print(f" 🌐 WandB Online: {wandb_status}")
    print(f" ⚡ Dispositivo: {device_name_str} | Ambientes: {args.num_envs} ({workers_info} CPU Workers)")
    print(f" 🎯 Total Timesteps: {args.total_timesteps:,} | Passos por Rollout: {args.num_steps}")
    print(f" 💾 Checkpoints a cada: {args.save_interval_steps:,} passos")
    print(f" 📈 Para abrir o TensorBoard: tensorboard --logdir {args.tensorboard_dir}")
    print("=" * 75)
    agents = vec_env.agents
    obs_dim = SSLELCooperationAttackerEnv().num_local_obs
    state_dim = SSLELCooperationAttackerEnv().num_state_features
    act_dim = 4

    trainer = MAPPOTrainer(
        agents=agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        clip_param=args.clip_param,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        use_shared_actor=True,
        device=args.device,
    )

    buffer = MultiAgentRolloutBuffer(
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        agents=agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=args.device,
    )

    cur_obs, cur_states = vec_env.reset()
    total_steps = 0
    last_saved_step = 0
    start_update = 1
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)

    # Rastreamento por episódio para métricas reais e robustas
    env_ep_returns = {a: np.zeros(args.num_envs, dtype=np.float32) for a in agents}
    env_ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    recent_ep_returns_all = deque(maxlen=100)
    recent_ep_returns = {a: deque(maxlen=100) for a in agents}
    recent_ep_lengths = deque(maxlen=100)
    recent_goals = deque(maxlen=100)
    recent_passes = deque(maxlen=100)

    shaping_keys = [
        "goal", "pass_success", "shot_on_goal", "shot_attempt", "ball_grad",
        "move_to_ball", "alignment", "kick_action", "push_to_goal", "infrared",
        "receiver_positioning", "collision_teammates", "out_of_bounds",
        "area_violation", "ball_out", "energy"
    ]
    recent_shaping = {k: deque(maxlen=100) for k in shaping_keys}

    goals_scored = 0
    passes_completed = 0
    episodes_completed = 0
    best_mean_reward = -float("inf")

    # Verificar se devemos retomar o treinamento (Resume)
    resume_target = None
    if args.resume_path:
        resume_target = args.resume_path
    elif args.resume:
        candidates = [
            os.path.join(ckpt_dir, "mappo_latest.pt"),
            os.path.join(pt_dir, "mappo_best.pt"),
            os.path.join(pt_dir, "mappo_final.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                resume_target = c
                break
        if not resume_target:
            all_pts = glob.glob(os.path.join(ckpt_dir, "mappo_step_*.pt"))
            if all_pts:
                resume_target = max(all_pts, key=os.path.getmtime)

    if resume_target and os.path.exists(resume_target):
        print("\n" + "🔄" * 38)
        print(f" [RESUME] RETOMANDO TREINAMENTO A PARTIR DO CHECKPOINT:")
        print(f" 📂 Arquivo: {resume_target}")
        ckpt_meta = trainer.load(resume_target, load_optimizers=True)
        total_steps = ckpt_meta.get("total_steps", 0)
        last_saved_step = total_steps
        goals_scored = ckpt_meta.get("goals_scored", 0)
        passes_completed = ckpt_meta.get("passes_completed", 0)
        start_update = (total_steps // (args.num_steps * args.num_envs)) + 1
        print(f" 🎯 Passos Anteriores: {total_steps:,} | Gols: {goals_scored} | Passes: {passes_completed}")
        print(f" ⚡ Continuando da iteração {start_update} até {num_updates}!")
        print("🔄" * 38 + "\n")

    start_time = time.time()

    # Barra de Progresso Visual (Inicia a partir de total_steps)
    pbar = tqdm(
        total=args.total_timesteps,
        initial=total_steps,
        desc="⚽ Treinando MAPPO (cooperation_attacker)",
        unit="steps",
        dynamic_ncols=True,
    )

    for iteration in range(start_update, num_updates + 1):
        # 1. Coleta do Rollout
        for step in range(args.num_steps):
            actions_dict, log_probs_dict = trainer.get_actions(cur_obs)
            values = trainer.get_value(cur_states)

            next_obs, next_states, rewards, dones, infos = vec_env.step(actions_dict)

            # Acumular retorno e passos dos episódios em andamento
            for a in agents:
                env_ep_returns[a] += rewards[a]
            env_ep_lengths += 1

            # Rastrear episódios finalizados (done)
            for i, done in enumerate(dones[agents[0]]):
                if done:
                    episodes_completed += 1
                    ret_0 = float(env_ep_returns["blue_0"][i])
                    ret_1 = float(env_ep_returns["blue_1"][i])
                    ret_team = (ret_0 + ret_1) / 2.0

                    recent_ep_returns["blue_0"].append(ret_0)
                    recent_ep_returns["blue_1"].append(ret_1)
                    recent_ep_returns_all.append(ret_team)
                    recent_ep_lengths.append(int(env_ep_lengths[i]))

                    env_ep_returns["blue_0"][i] = 0.0
                    env_ep_returns["blue_1"][i] = 0.0
                    env_ep_lengths[i] = 0

                    shaping = infos[i][agents[0]].get("reward_shaping")
                    if shaping is not None:
                        g_hit = 1 if shaping.get("goal", 0.0) > 0 else 0
                        p_hit = 1 if shaping.get("pass_success", 0.0) > 0 else 0
                        goals_scored += g_hit
                        passes_completed += p_hit
                        recent_goals.append(g_hit)
                        recent_passes.append(p_hit)
                        for k, v in shaping.items():
                            if k in recent_shaping:
                                recent_shaping[k].append(float(v))

            buffer.insert(
                obs_dict=cur_obs,
                state=cur_states,
                actions_dict=actions_dict,
                log_probs_dict=log_probs_dict,
                values=values,
                rewards_dict=rewards,
                dones_dict=dones,
            )

            cur_obs = next_obs
            cur_states = next_states
            total_steps += args.num_envs

        # 2. Computar Vantagem com Crítico Centralizado
        next_values = trainer.get_value(cur_states)
        next_dones = torch.as_tensor(dones[agents[0]], device=args.device, dtype=torch.float32).view(-1, 1)
        buffer.compute_gae(next_values, next_dones)

        # 3. Treinar MAPPO
        metrics = trainer.train_step(buffer)

        # 4. Atualizar Barra de Progresso e Métricas
        steps_this_update = args.num_steps * args.num_envs
        pbar.update(steps_this_update)

        elapsed = time.time() - start_time
        steps_trained_this_session = total_steps - (start_update - 1) * (args.num_steps * args.num_envs)
        fps = int(steps_trained_this_session / max(elapsed, 1e-6))

        mean_team_rew = float(np.mean(recent_ep_returns_all)) if len(recent_ep_returns_all) > 0 else 0.0
        mean_b0_rew = float(np.mean(recent_ep_returns["blue_0"])) if len(recent_ep_returns["blue_0"]) > 0 else 0.0
        mean_b1_rew = float(np.mean(recent_ep_returns["blue_1"])) if len(recent_ep_returns["blue_1"]) > 0 else 0.0
        mean_ep_len = float(np.mean(recent_ep_lengths)) if len(recent_ep_lengths) > 0 else 0.0
        goal_rate = (sum(recent_goals) / len(recent_goals) * 100) if len(recent_goals) > 0 else 0.0
        pass_rate = (sum(recent_passes) / len(recent_passes) * 100) if len(recent_passes) > 0 else 0.0

        pbar.set_postfix({
            "FPS": f"{fps:d}",
            "Rew": f"{mean_team_rew:+.2f}",
            "Gols": goals_scored,
            "Passes": passes_completed,
            "A_Loss": f"{metrics['actor_loss']:.2f}",
            "C_Loss": f"{metrics['critic_loss']:.3f}",
        })

        # 5. Registro Completo e Estruturado no TensorBoard e WandB
        if iteration % args.log_interval == 0:
            # A) Desempenho e Velocidade
            writer.add_scalar("Desempenho/FPS", fps, total_steps)
            writer.add_scalar("Desempenho/Duracao_Media_Episodio_Passos", mean_ep_len, total_steps)
            writer.add_scalar("Desempenho/Episodios_Concluidos_Total", episodes_completed, total_steps)

            # B) Perdas das Redes Neurais (MAPPO)
            writer.add_scalar("Perdas_MAPPO/Actor_Loss", metrics["actor_loss"], total_steps)
            writer.add_scalar("Perdas_MAPPO/Critic_Loss", metrics["critic_loss"], total_steps)
            writer.add_scalar("Perdas_MAPPO/Entropia", metrics["entropy"], total_steps)

            # C) Recompensa Média por Episódio (Retorno Total por Agente e Equipe)
            writer.add_scalar("Recompensa_Media/Retorno_Medio_Equipe", mean_team_rew, total_steps)
            writer.add_scalar("Recompensa_Media/Retorno_Blue0_Condutor", mean_b0_rew, total_steps)
            writer.add_scalar("Recompensa_Media/Retorno_Blue1_Receptor", mean_b1_rew, total_steps)

            # D) Métricas de Eficácia do Jogo
            writer.add_scalar("Metricas_Jogo/Taxa_de_Gols_Pct", goal_rate, total_steps)
            writer.add_scalar("Metricas_Jogo/Taxa_de_Passes_Pct", pass_rate, total_steps)
            writer.add_scalar("Metricas_Jogo/Total_Gols_Marcados", goals_scored, total_steps)
            writer.add_scalar("Metricas_Jogo/Total_Passes_Completados", passes_completed, total_steps)

            # Preparar payload unificado para o WandB
            wandb_payload = {
                "Desempenho/FPS": fps,
                "Desempenho/Duracao_Media_Episodio_Passos": mean_ep_len,
                "Desempenho/Episodios_Concluidos_Total": episodes_completed,
                "Perdas_MAPPO/Actor_Loss": metrics["actor_loss"],
                "Perdas_MAPPO/Critic_Loss": metrics["critic_loss"],
                "Perdas_MAPPO/Entropia": metrics["entropy"],
                "Recompensa_Media/Retorno_Medio_Equipe": mean_team_rew,
                "Recompensa_Media/Retorno_Blue0_Condutor": mean_b0_rew,
                "Recompensa_Media/Retorno_Blue1_Receptor": mean_b1_rew,
                "Metricas_Jogo/Taxa_de_Gols_Pct": goal_rate,
                "Metricas_Jogo/Taxa_de_Passes_Pct": pass_rate,
                "Metricas_Jogo/Total_Gols_Marcados": goals_scored,
                "Metricas_Jogo/Total_Passes_Completados": passes_completed,
            }

            # E) Recompensa Detalhada por Ação ("Por Coisa")
            if len(recent_shaping["goal"]) > 0:
                writer.add_scalar("Recompensa_Por_Item/1_Gol_Marcado", float(np.mean(recent_shaping["goal"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/2_Passe_Conectado", float(np.mean(recent_shaping["pass_success"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/3_Chute_ao_Gol_Veloz", float(np.mean(recent_shaping["shot_on_goal"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/4_Aproximacao_e_Contorno_Bola", float(np.mean(recent_shaping["move_to_ball"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/5_Avanco_da_Bola_ao_Gol", float(np.mean(recent_shaping["ball_grad"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/6_Alinhamento_Corporal_Gol", float(np.mean(recent_shaping["alignment"])), total_steps)
                kick_total_rew = float(np.mean(recent_shaping["kick_action"])) + float(np.mean(recent_shaping["push_to_goal"]))
                writer.add_scalar("Recompensa_Por_Item/7_Disparo_e_Momento_do_Chute", kick_total_rew, total_steps)
                writer.add_scalar("Recompensa_Por_Item/8_Controle_Infravermelho", float(np.mean(recent_shaping["infrared"])), total_steps)
                writer.add_scalar("Recompensa_Por_Item/9_Desmarcacao_2o_Atacante", float(np.mean(recent_shaping["receiver_positioning"])), total_steps)

                # F) Penalidades
                writer.add_scalar("Penalidades/1_Colisao_entre_Companheiros", float(np.mean(recent_shaping["collision_teammates"])), total_steps)
                writer.add_scalar("Penalidades/2_Invasao_de_Area", float(np.mean(recent_shaping["area_violation"])), total_steps)
                writer.add_scalar("Penalidades/3_Fora_de_Campo", float(np.mean(recent_shaping["out_of_bounds"])), total_steps)
                writer.add_scalar("Penalidades/4_Bola_Fora", float(np.mean(recent_shaping["ball_out"])), total_steps)
                writer.add_scalar("Penalidades/5_Gasto_Tempo_e_Energia", float(np.mean(recent_shaping["energy"])), total_steps)

                wandb_payload.update({
                    "Recompensa_Por_Item/1_Gol_Marcado": float(np.mean(recent_shaping["goal"])),
                    "Recompensa_Por_Item/2_Passe_Conectado": float(np.mean(recent_shaping["pass_success"])),
                    "Recompensa_Por_Item/3_Chute_ao_Gol_Veloz": float(np.mean(recent_shaping["shot_on_goal"])),
                    "Recompensa_Por_Item/4_Aproximacao_e_Contorno_Bola": float(np.mean(recent_shaping["move_to_ball"])),
                    "Recompensa_Por_Item/5_Avanco_da_Bola_ao_Gol": float(np.mean(recent_shaping["ball_grad"])),
                    "Recompensa_Por_Item/6_Alinhamento_Corporal_Gol": float(np.mean(recent_shaping["alignment"])),
                    "Recompensa_Por_Item/7_Disparo_e_Momento_do_Chute": kick_total_rew,
                    "Recompensa_Por_Item/8_Controle_Infravermelho": float(np.mean(recent_shaping["infrared"])),
                    "Recompensa_Por_Item/9_Desmarcacao_2o_Atacante": float(np.mean(recent_shaping["receiver_positioning"])),
                    "Penalidades/1_Colisao_entre_Companheiros": float(np.mean(recent_shaping["collision_teammates"])),
                    "Penalidades/2_Invasao_de_Area": float(np.mean(recent_shaping["area_violation"])),
                    "Penalidades/3_Fora_de_Campo": float(np.mean(recent_shaping["out_of_bounds"])),
                    "Penalidades/4_Bola_Fora": float(np.mean(recent_shaping["ball_out"])),
                    "Penalidades/5_Gasto_Tempo_e_Energia": float(np.mean(recent_shaping["energy"])),
                })

            if args.wandb:
                try:
                    wandb.log(wandb_payload, step=total_steps)
                except Exception:
                    pass

        # 6. Salvar Checkpoint a cada save_interval_steps (para acompanhamento ao vivo no play_attacker)
        if (total_steps - last_saved_step >= args.save_interval_steps) or (iteration == num_updates):
            last_saved_step = total_steps
            step_save_path = os.path.join(ckpt_dir, f"mappo_{treino_tag}_step_{total_steps}.pt")
            metadata = {
                "total_steps": total_steps,
                "iteration": iteration,
                "goals_scored": goals_scored,
                "passes_completed": passes_completed,
                "mean_reward": float(mean_team_rew),
                "goal_rate": float(goal_rate),
                "treino_tag": treino_tag,
            }
            trainer.save(step_save_path, extra_info=metadata, export_onnx=True)
            trainer.save(os.path.join(ckpt_dir, "mappo_latest.pt"), extra_info=metadata, export_onnx=True)

            # Salvar automaticamente o melhor modelo histórico deste treino
            if mean_team_rew > best_mean_reward and total_steps >= 100_000:
                best_mean_reward = mean_team_rew

                best_pt = os.path.join(pt_dir, f"attacker_{treino_tag}_best.pt")
                best_onnx = os.path.join(onnx_dir, f"attacker_{treino_tag}_best.onnx")

                trainer.save(best_pt, extra_info=metadata, export_onnx=False)
                trainer.export_onnx(best_onnx)

                tqdm.write(f"\n🌟 [NOVO RECORDE - {treino_tag.upper()} | Passo {total_steps:,}] Salvo: attacker_{treino_tag}_best (.pt e .onnx) (Rew: {mean_team_rew:+.2f})")

            tqdm.write(
                f"\n💾 [CHECKPOINT {treino_tag.upper()} | {total_steps:,} passos] "
                f"Gols: {goals_scored} | Passes: {passes_completed} | Rew: {mean_team_rew:+.2f} "
                f"-> {step_save_path}\n"
            )

    pbar.close()
    writer.close()
    if args.wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    # 7. Salvar Modelo Final estrito em modelos/pt/ e modelos/onnx/
    final_metadata = {
        "total_steps": total_steps,
        "iteration": iteration,
        "goals_scored": goals_scored,
        "passes_completed": passes_completed,
        "mean_reward": float(mean_team_rew),
        "goal_rate": float(goal_rate),
        "treino_tag": treino_tag,
    }

    final_pt = os.path.join(pt_dir, f"attacker_{treino_tag}_final.pt")
    final_onnx = os.path.join(onnx_dir, f"attacker_{treino_tag}_final.onnx")
    trainer.save(final_pt, extra_info=final_metadata, export_onnx=False)
    trainer.export_onnx(final_onnx)

    # Garantir que exista o arquivo best deste treino
    best_pt = os.path.join(pt_dir, f"attacker_{treino_tag}_best.pt")
    best_onnx = os.path.join(onnx_dir, f"attacker_{treino_tag}_best.onnx")
    if not os.path.exists(best_pt):
        trainer.save(best_pt, extra_info=final_metadata, export_onnx=False)
        trainer.export_onnx(best_onnx)

    # 8. Apagar todos os checkpoints intermediários da pasta checkpoints/
    for ckpt_file in glob.glob(os.path.join(ckpt_dir, "*")):
        try:
            if os.path.isfile(ckpt_file) or os.path.islink(ckpt_file):
                os.remove(ckpt_file)
            elif os.path.isdir(ckpt_file):
                shutil.rmtree(ckpt_file)
        except Exception:
            pass

    # 9. Limpar qualquer arquivo/pasta fora de pt/, onnx/ e checkpoints/
    for item in glob.glob(os.path.join(save_dir, "*")):
        basename = os.path.basename(item)
        if basename not in ("pt", "onnx", "checkpoints"):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
            except Exception:
                pass

    print("\n" + "=" * 75)
    print(f"[*] TREINAMENTO {treino_tag.upper()} CONCLUÍDO COM SUCESSO!")
    print(f"    📂 Modelos PyTorch (.pt) em {pt_dir}/:")
    print(f"       -> attacker_{treino_tag}_best.pt")
    print(f"       -> attacker_{treino_tag}_final.pt")
    print(f"    📂 Modelos ONNX (.onnx) em {onnx_dir}/:")
    print(f"       -> attacker_{treino_tag}_best.onnx")
    print(f"       -> attacker_{treino_tag}_final.onnx")
    print(f"    🧹 Checkpoints em {ckpt_dir}/ foram apagados com sucesso!")
    print(f"    📊 TensorBoard logs prontos em: {tb_log_dir}/")
    print("=" * 75 + "\n")

    vec_env.close()


if __name__ == "__main__":
    train()
