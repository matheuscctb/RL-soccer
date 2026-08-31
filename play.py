import os
import glob
import time
import argparse
from typing import Optional
import numpy as np
import torch

from rsoccer_gym.ssl.ssl_el_cooperation_attacker import SSLELCooperationAttackerEnv
from mappo import MAPPOTrainer


def find_latest_checkpoint(search_dir: str = "modelos") -> Optional[str]:
    """Busca o modelo mais recente salvo na pasta de modelos."""
    latest_ckpt = os.path.join(search_dir, "checkpoints", "mappo_latest.pt")
    if os.path.exists(latest_ckpt):
        return latest_ckpt

    all_pts = glob.glob(os.path.join(search_dir, "**", "*.pt"), recursive=True)
    if all_pts:
        latest_file = max(all_pts, key=os.path.getmtime)
        return latest_file

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Execução e Teste dos Atacantes Cooperativos com o Modelo Mais Recente (SSL-EL)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Caminho do modelo específico (padrão: busca automática do modelo mais recente em modelos/)",
    )
    parser.add_argument("--save-dir", type=str, default="modelos", help="Diretório onde os modelos estão salvos")
    parser.add_argument("--episodes", type=int, default=10, help="Número de episódios para simular")
    parser.add_argument("--render", action="store_true", help="Renderizar a interface visual em PyGame")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Desativar renderizador gráfico")
    parser.set_defaults(render=True)
    parser.add_argument("--fps", type=int, default=40, help="Taxa de quadros (FPS) para renderização (padrão: 40 Hz)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo para inferência (padrão: cuda se disponível, senão cpu)")
    return parser.parse_args()


def play():
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

    # Identificar modelo a ser carregado
    model_path = args.model_path
    if model_path is None or not os.path.exists(model_path):
        model_path = find_latest_checkpoint(args.save_dir)

    if model_path and os.path.exists(model_path):
        try:
            trainer.load(model_path)
            mod_time = time.ctime(os.path.getmtime(model_path))
            print(f"[*] Modelo MAIS RECENTE carregado com sucesso:")
            print(f"    -> Arquivo: {model_path}")
            print(f"    -> Salvo em: {mod_time}")
        except Exception as e:
            print(f"[!] Erro ao carregar pesos de '{model_path}': {e}. Usando inicialização padrão.")
    else:
        print(f"[!] Nenhum checkpoint encontrado em '{args.save_dir}'. Executando com pesos iniciais.")

    goals = 0
    passes = 0

    print("=" * 65)
    print(f" Iniciando Partida SSL-EL: cooperation_attacker (2 Azuis vs Defesa)")
    print(f" Total de Episódios: {args.episodes} | Renderização: {args.render}")
    print("=" * 65)

    for ep in range(1, args.episodes + 1):
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

        shaping = infos[agents[0]]["reward_shaping"]
        if shaping is not None:
            if shaping.get("goal", 0.0) > 0:
                goals += 1
                print(f" [Ep {ep:02d}] ⚽ GOL MARCADO! | Passos: {step_count:03d} | Recompensa: {ep_reward['blue_0']:+06.2f}")
            elif shaping.get("pass_success", 0.0) > 0:
                passes += 1
                print(f" [Ep {ep:02d}] 🎯 Passe Concluído! | Passos: {step_count:03d} | Recompensa: {ep_reward['blue_0']:+06.2f}")
            else:
                print(f" [Ep {ep:02d}] ⏹ Fim de jogada | Passos: {step_count:03d} | Recompensa: {ep_reward['blue_0']:+06.2f}")

    env.close()
    print("=" * 65)
    print(f" Resumo Final: {args.episodes} Episódios | Gols: {goals} | Passes: {passes}")
    print("=" * 65)


if __name__ == "__main__":
    play()
