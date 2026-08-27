import os
import glob
import time
import argparse
import multiprocessing as mp
from collections import deque
from typing import List

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import rsoccer_gym
from rsoccer_gym.ssl.ssl_el_cooperation_attacker import SSLELCooperationAttackerEnv
from mappo import MAPPOTrainer, MultiAgentRolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento MAPPO para Cooperação de Atacantes SSL-EL com TensorBoard e ONNX (cooperation_attacker)"
    )
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total de passos de treino (padrão: 5.000.000)")
    parser.add_argument("--num-envs", type=int, default=16, help="Número de ambientes paralelos (ex: 16, 64, 256, 2048)")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de processos CPU workers (0 = auto / todos os núcleos)")
    parser.add_argument("--num-steps", type=int, default=200, help="Passos de rollout por atualização")
    parser.add_argument("--ppo-epochs", type=int, default=5, help="Épocas PPO por atualização")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamanho do mini-batch PPO")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Taxa de aprendizado do Ator")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Taxa de aprendizado do Crítico Central")
    parser.add_argument("--gamma", type=float, default=0.99, help="Fator de desconto")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Fator GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="Parâmetro de clip do PPO")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Coeficiente de entropia")
    parser.add_argument("--save-interval-steps", type=int, default=100_000, help="Intervalo de passos para salvar checkpoint (padrão: 100.000)")
    parser.add_argument("--log-interval", type=int, default=5, help="Intervalo de iterações para log no console/tensorboard")
    parser.add_argument("--exp-name", type=str, default="cooperation_attacker", help="Nome do experimento")
    parser.add_argument("--save-dir", type=str, default="modelos", help="Diretório onde os modelos e checkpoints serão salvos")
    parser.add_argument("--tensorboard-dir", type=str, default="runs", help="Diretório de logs do TensorBoard (padrão: runs)")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de execução (padrão: cuda)")
    parser.add_argument("--resume", action="store_true", help="Retomar o treinamento automaticamente a partir do último checkpoint")
    parser.add_argument("--resume-path", type=str, default="", help="Caminho específico de um checkpoint (.pt) para retomar o treino")
    return parser.parse_args()


def _worker_process(remote, parent_remote, count: int):
    """Processo worker isolado que gerencia um subconjunto de ambientes em paralelo."""
    parent_remote.close()
    envs = [SSLELCooperationAttackerEnv() for _ in range(count)]
    agents = SSLELCooperationAttackerEnv.agents
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                results_obs = {a: [] for a in agents}
                results_state = []
                results_rew = {a: [] for a in agents}
                results_done = {a: [] for a in agents}
                results_info = []

                for i, env in enumerate(envs):
                    actions = {a: data[a][i] for a in agents}
                    obs, rew, term, trunc, info = env.step(actions)
                    done = term[agents[0]] or trunc[agents[0]]
                    if done:
                        obs, _ = env.reset()
                    state = env.get_global_state()
                    for a in agents:
                        results_obs[a].append(obs[a])
                        results_rew[a].append(rew[a])
                        results_done[a].append(done)
                    results_state.append(state)
                    results_info.append(info)

                remote.send((
                    {a: np.array(results_obs[a], dtype=np.float32) for a in agents},
                    np.array(results_state, dtype=np.float32),
                    {a: np.array(results_rew[a], dtype=np.float32) for a in agents},
                    {a: np.array(results_done[a], dtype=np.float32) for a in agents},
                    results_info
                ))
            elif cmd == "reset":
                results_obs = {a: [] for a in agents}
                results_state = []
                for env in envs:
                    obs, _ = env.reset()
                    state = env.get_global_state()
                    for a in agents:
                        results_obs[a].append(obs[a])
                    results_state.append(state)
                remote.send((
                    {a: np.array(results_obs[a], dtype=np.float32) for a in agents},
                    np.array(results_state, dtype=np.float32)
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
    exp_dir = os.path.join(save_dir, args.exp_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Configuração do TensorBoard
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.tensorboard_dir, f"{args.exp_name}_{timestamp_str}")
    writer = SummaryWriter(log_dir=tb_log_dir)

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
    print(f" 🚀 INICIANDO TREINAMENTO MAPPO: {args.exp_name}")
    print(f" 📂 Modelos (.pt e .onnx): {save_dir}/ | Checkpoints: {ckpt_dir}/")
    print(f" 📊 TensorBoard Logs: {tb_log_dir}/")
    print(f" ⚡ Dispositivo: {device_name_str} | Ambientes: {args.num_envs} ({workers_info} CPU Workers)")
    print(f" 🎯 Total Timesteps: {args.total_timesteps:,} | Passos por Rollout: {args.num_steps}")
    print(f" 💾 Checkpoints a cada: {args.save_interval_steps:,} passos (salvos em .pt e .onnx)")
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

    recent_rewards = deque(maxlen=100)
    recent_goals = deque(maxlen=100)
    recent_passes = deque(maxlen=100)
    goals_scored = 0
    passes_completed = 0
    episodes_completed = 0
    best_mean_reward = -float("inf")
    last_shaping_snapshot = None

    # Verificar se devemos retomar o treinamento (Resume)
    resume_target = None
    if args.resume_path:
        resume_target = args.resume_path
    elif args.resume:
        candidates = [
            os.path.join(ckpt_dir, "mappo_latest.pt"),
            os.path.join(save_dir, "mappo_latest.pt"),
            os.path.join(exp_dir, "mappo_latest.pt"),
            os.path.join(save_dir, "mappo_best.pt"),
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

            # Rastrear estatísticas dos episódios
            for i, done in enumerate(dones[agents[0]]):
                if done:
                    episodes_completed += 1
                    shaping = infos[i][agents[0]]["reward_shaping"]
                    if shaping is not None:
                        last_shaping_snapshot = shaping
                        g_hit = 1 if shaping.get("goal", 0.0) > 0 else 0
                        p_hit = 1 if shaping.get("pass_success", 0.0) > 0 else 0
                        goals_scored += g_hit
                        passes_completed += p_hit
                        recent_goals.append(g_hit)
                        recent_passes.append(p_hit)

            for r in rewards[agents[0]]:
                recent_rewards.append(r)

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
        mean_rew = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
        goal_rate = (sum(recent_goals) / len(recent_goals) * 100) if len(recent_goals) > 0 else 0.0
        pass_rate = (sum(recent_passes) / len(recent_passes) * 100) if len(recent_passes) > 0 else 0.0

        pbar.set_postfix({
            "FPS": f"{fps:d}",
            "Rew": f"{mean_rew:+.2f}",
            "Gols": goals_scored,
            "Passes": passes_completed,
            "A_Loss": f"{metrics['actor_loss']:.2f}",
            "C_Loss": f"{metrics['critic_loss']:.3f}",
        })

        # 5. Registro no TensorBoard
        if iteration % args.log_interval == 0:
            writer.add_scalar("Performance/Mean_Reward", mean_rew, total_steps)
            writer.add_scalar("Performance/Goal_Rate_Pct", goal_rate, total_steps)
            writer.add_scalar("Performance/Pass_Rate_Pct", pass_rate, total_steps)
            writer.add_scalar("Performance/Goals_Total", goals_scored, total_steps)
            writer.add_scalar("Performance/Passes_Total", passes_completed, total_steps)
            writer.add_scalar("Performance/Episodes_Total", episodes_completed, total_steps)
            writer.add_scalar("Performance/FPS", fps, total_steps)

            writer.add_scalar("Loss/Actor_Loss", metrics["actor_loss"], total_steps)
            writer.add_scalar("Loss/Critic_Loss", metrics["critic_loss"], total_steps)
            writer.add_scalar("Loss/Entropy", metrics["entropy"], total_steps)

            if last_shaping_snapshot:
                for k, v in last_shaping_snapshot.items():
                    writer.add_scalar(f"Reward_Shaping/{k}", v, total_steps)

        # 6. Salvar Checkpoint a cada 100.000 passos (salva .pt e .onnx)
        if (total_steps - last_saved_step >= args.save_interval_steps) or (iteration == num_updates):
            last_saved_step = total_steps
            step_save_path = os.path.join(ckpt_dir, f"mappo_step_{total_steps}.pt")
            metadata = {
                "total_steps": total_steps,
                "iteration": iteration,
                "goals_scored": goals_scored,
                "passes_completed": passes_completed,
                "mean_reward": float(mean_rew),
                "goal_rate": float(goal_rate),
            }
            trainer.save(step_save_path, extra_info=metadata, export_onnx=True)
            trainer.save(os.path.join(ckpt_dir, "mappo_latest.pt"), extra_info=metadata, export_onnx=True)
            trainer.save(os.path.join(save_dir, "mappo_latest.pt"), extra_info=metadata, export_onnx=True)

            # Salvar automaticamente o melhor modelo histórico (mappo_best.pt e mappo_best.onnx)
            if mean_rew > best_mean_reward and total_steps >= 200_000:
                best_mean_reward = mean_rew
                best_path = os.path.join(save_dir, "mappo_best.pt")
                best_exp_path = os.path.join(exp_dir, "mappo_best.pt")
                trainer.save(best_path, extra_info=metadata, export_onnx=True)
                trainer.save(best_exp_path, extra_info=metadata, export_onnx=True)
                tqdm.write(f"🌟 [NOVO RECORDE] Melhor modelo salvo (.pt e .onnx): {best_path} (Rew: {mean_rew:+.2f})")

            tqdm.write(
                f"\n💾 [CHECKPOINT SALVO | {total_steps:,} passos] "
                f"Gols: {goals_scored} | Passes: {passes_completed} | Rew: {mean_rew:+.2f} "
                f"-> {step_save_path} (+ .onnx)\n"
            )

    pbar.close()
    writer.close()

    # Salvar modelo final ao término do treinamento
    final_exp_path = os.path.join(exp_dir, "mappo_final.pt")
    final_global_path = os.path.join(save_dir, "mappo_final.pt")
    trainer.save(final_exp_path, export_onnx=True)
    trainer.save(final_global_path, export_onnx=True)
    trainer.save(os.path.join(save_dir, "mappo_latest.pt"), export_onnx=True)

    print("\n" + "=" * 75)
    print(f"[*] TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"    -> Modelos Finais (.pt e .onnx) salvos em: {final_global_path}")
    print(f"    -> Todos os checkpoints intermediários estão preservados em: {ckpt_dir}/")
    print(f"    -> TensorBoard logs prontos em: {tb_log_dir}/")
    print("=" * 75 + "\n")

    vec_env.close()


if __name__ == "__main__":
    train()
