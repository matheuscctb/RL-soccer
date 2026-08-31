import os
import re
import glob
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import torch
import pygame

from rsoccer_gym.ssl.ssl_el_cooperation_attacker import SSLELCooperationAttackerEnv
from mappo import MAPPOTrainer


def get_latest_checkpoint_info(search_dir: str = "modelos") -> Tuple[Optional[str], float]:
    """Retorna o caminho e o timestamp de modificação do checkpoint mais recente."""
    latest_ckpt = os.path.join(search_dir, "checkpoints", "mappo_latest.pt")
    if os.path.exists(latest_ckpt):
        return latest_ckpt, os.path.getmtime(latest_ckpt)

    all_pts = glob.glob(os.path.join(search_dir, "**", "*.pt"), recursive=True)
    if all_pts:
        latest_file = max(all_pts, key=os.path.getmtime)
        return latest_file, os.path.getmtime(latest_file)

    return None, 0.0


def extract_step_count(filepath: str, ckpt_dict: dict = None) -> Optional[int]:
    """Extrai a contagem de passos do checkpoint."""
    if ckpt_dict and "total_steps" in ckpt_dict:
        return ckpt_dict["total_steps"]
    match = re.search(r"mappo_(?:step|best_step|final_step)_(\d+)\.pt", filepath)
    if match:
        return int(match.group(1))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Acompanhamento ao Vivo da Evolução dos Atacantes Cooperativos (SSL-EL Live Watcher)"
    )
    parser.add_argument("--save-dir", type=str, default="modelos", help="Diretório onde os modelos estão sendo salvos (padrão: modelos)")
    parser.add_argument("--fps", type=int, default=40, help="Taxa de quadros para renderização em tempo real (padrão: 40 Hz)")
    parser.add_argument("--render", action="store_true", help="Renderizar em Pygame")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Desativar renderização gráfica")
    parser.set_defaults(render=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo para inferência (padrão: cuda se disponível, senão cpu)")
    return parser.parse_args()


def live_watch():
    args = parse_args()
    render_mode = "human" if args.render else None
    env = SSLELCooperationAttackerEnv(render_mode=render_mode)
    agents = env.agents

    trainer = MAPPOTrainer(
        agents=agents,
        obs_dim=env.num_local_obs,
        state_dim=env.num_state_features,
        act_dim=4,
        use_shared_actor=True,
        device=args.device,
    )

    last_loaded_path = None
    last_loaded_mtime = 0.0
    current_ckpt_steps = "Inicial (0)"

    print("=" * 75)
    print(" 📡 PLAY ATTACKER: MONITORAMENTO AO VIVO DO TREINAMENTO (SSL-EL)")
    print(f" Monitorando pasta: '{args.save_dir}/' e '{args.save_dir}/checkpoints/'")
    print(" Cada novo checkpoint salvo a cada 100 mil passos será recarregado ao vivo!")
    print(" Você pode deixar esta janela aberta enquanto o treino roda no outro terminal.")
    print(" Pressione Ctrl+C para encerrar o monitoramento.")
    print("=" * 75)

    episode = 0
    total_goals = 0
    total_passes = 0
    recent_goals_window = []

    try:
        while True:
            episode += 1

            # Verificar se há um checkpoint novo ou atualizado antes de cada episódio
            latest_path, latest_mtime = get_latest_checkpoint_info(args.save_dir)

            if latest_path is not None and (latest_path != last_loaded_path or latest_mtime > last_loaded_mtime + 1e-4):
                try:
                    time.sleep(0.05)
                    ckpt_data = trainer.load(latest_path)
                    last_loaded_path = latest_path
                    last_loaded_mtime = latest_mtime

                    steps_found = extract_step_count(latest_path, ckpt_data)
                    current_ckpt_steps = f"{steps_found:,} passos" if steps_found else "Mais recente"
                    mod_time_str = time.strftime("%H:%M:%S", time.localtime(latest_mtime))

                    print("\n" + "🔥" * 38)
                    print(f" [LIVE UPDATE] NOVO CHECKPOINT DETECTADO E RECARREGADO!")
                    print(f" 🎯 Marco de Evolução: {current_ckpt_steps}")
                    print(f" 📂 Arquivo: {latest_path}")
                    print(f" ⏰ Horário: {mod_time_str}")
                    print("🔥" * 38 + "\n")
                except Exception as e:
                    print(f"[!] Aguardando escrita do checkpoint ({e})...")

            # Atualizar título da janela PyGame se estiver renderizando
            if args.render and pygame.display.get_init():
                pygame.display.set_caption(
                    f"SSL-EL Live | Checkpoint: {current_ckpt_steps} | Ep: {episode} | Gols: {total_goals} | Passes: {total_passes}"
                )

            obs, _ = env.reset()
            ep_reward = {a: 0.0 for a in agents}
            step_count = 0
            done = False

            while not done:
                batched_obs = {a: np.expand_dims(obs[a], axis=0) for a in agents}
                actions_dict, _ = trainer.get_actions(batched_obs, deterministic=True)
                step_actions = {a: actions_dict[a][0].cpu().numpy() for a in agents}

                obs, rewards, terms, truncs, infos = env.step(step_actions)
                for a in agents:
                    ep_reward[a] += rewards[a]

                step_count += 1
                done = terms[agents[0]] or truncs[agents[0]]

                if args.render:
                    time.sleep(1.0 / args.fps)

            # Processar resultado do episódio
            shaping = infos[agents[0]]["reward_shaping"]
            is_goal = False
            is_pass = False
            if shaping is not None:
                if shaping.get("goal", 0.0) > 0:
                    is_goal = True
                    total_goals += 1
                if shaping.get("pass_success", 0.0) > 0:
                    is_pass = True
                    total_passes += 1

            recent_goals_window.append(1 if is_goal else 0)
            if len(recent_goals_window) > 50:
                recent_goals_window.pop(0)

            goal_rate = (sum(recent_goals_window) / len(recent_goals_window)) * 100
            status_icon = "⚽ GOL!" if is_goal else ("🎯 PASSE!" if is_pass else "⏹")
            print(
                f"[Ao Vivo | Ep {episode:04d} | CKPT: {current_ckpt_steps}] {status_icon:8s} | "
                f"Passos: {step_count:03d} | "
                f"Rew: {ep_reward['blue_0']:+06.2f} | "
                f"Taxa de Gol (últimos 50): {goal_rate:04.1f}% | "
                f"Total Gols: {total_goals:03d} | Passes: {total_passes:03d}"
            )

    except KeyboardInterrupt:
        print("\n[!] Monitoramento ao vivo encerrado pelo usuário.")
    finally:
        env.close()
        print(f"Resumo da Sessão: {episode} episódios acompanhados | Gols: {total_goals} | Passes: {total_passes}")


if __name__ == "__main__":
    live_watch()
