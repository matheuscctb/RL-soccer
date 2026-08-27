import math
import random
from typing import Dict, List, Tuple, Any

import gymnasium as gym
import numpy as np
import pygame

from rsoccer_gym.Entities import Frame, Robot, Ball, Field
from rsoccer_gym.Render.field import VSSRenderField
from rsoccer_gym.Render.utils import COLORS
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLELRenderField(VSSRenderField):
    """Renderizador gráfico para o campo SSL Entry Level (4.5m x 3.0m)."""
    length = 4.5
    width = 3.0
    margin = 0.3
    center_circle_r = 0.5
    penalty_length = 0.50
    penalty_width = 1.350
    goal_width = 0.70
    goal_depth = 0.18
    _scale = 180


class SSLELCooperationAttackerEnv(SSLBaseEnv):
    """
    Ambiente Multi-Agente SSL Entry Level (SSL-EL) para Cooperação de Atacantes (2 Atacantes Azuis vs Defesa):
    
    Características:
    - Campo SSL-EL oficial: 4.5m x 3.0m
    - Área de pênalti: 1.350m (Y) x 0.50m (X)
    - Gol: 0.70m x 0.18m
    - Robôs: 3 Azuis vs 3 Amarelos
    - Agentes Ativos Cooperativos:
        * blue_0: Robô Azul 0 (Passador / Atacante 1)
        * blue_1: Robô Azul 1 (Receptor / Atacante 2)
    - Blue 2: Robô de apoio defensivo estático
    - Amarelos: Goleiro (Amarelo 0) + 2 Defensores (Amarelos 1 e 2)
    - Ações Contínuas por Agente (4): [v_x, v_y, v_theta, kick_x]
    - Suporte nativo a CTDE (Treinamento Centralizado, Execução Descentralizada) para MAPPO.
    """

    agents = ["blue_0", "blue_1"]
    possible_agents = ["blue_0", "blue_1"]

    def __init__(self, render_mode=None):
        super().__init__(
            field_type=2,
            n_robots_blue=3,
            n_robots_yellow=3,
            time_step=0.025,
            render_mode=render_mode,
        )

        # Ajuste dimensional exato para a SSL-EL
        self.field.length = 4.5
        self.field.width = 3.0
        self.field.penalty_width = 1.350  # comprimento no eixo Y
        self.field.penalty_length = 0.50  # largura no eixo X
        self.field.goal_width = 0.70
        self.field.goal_depth = 0.18

        # Normalização de posições
        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )

        # Renderizador gráfico
        self.field_renderer = SSLELRenderField()
        self.window_size = self.field_renderer.window_size

        # Espaço de Ação por Agente: [v_x, v_y, v_theta, kick_x]
        self.action_space_dict = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.agents
        }
        # Para compatibilidade com gymnasium standard
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # =========================================================================
        # Espaço de Observação Local Descentralizado (Por Agente - 35 dimensões):
        # 1. Agente próprio (8): [x, y, sin(th), cos(th), v_x, v_y, v_th, infrared]
        # 2. Bola relativa/global (6): [x, y, v_x, v_y, dist_to_ball, angle_to_ball]
        # 3. Companheiro blue (8): [rel_x, rel_y, sin(th), cos(th), v_x, v_y, v_th, infrared]
        # 4. Companheiro estático blue_2 (2): [rel_x, rel_y]
        # 5. Adversários yellow 0, 1, 2 (2 cada = 6): [rel_x, rel_y]
        # 6. One-hot Agent ID (2): [1, 0] para blue_0, [0, 1] para blue_1
        # 7. Posse estimada / Contexto de passe (3): [has_possession, in_partner_cone, ball_towards_me]
        # Total = 8 + 6 + 8 + 2 + 6 + 2 + 3 = 35 dimensões
        # =========================================================================
        self.num_local_obs = 35
        self.observation_space_dict = {
            agent: gym.spaces.Box(
                low=-self.NORM_BOUNDS,
                high=self.NORM_BOUNDS,
                shape=(self.num_local_obs,),
                dtype=np.float32,
            )
            for agent in self.agents
        }
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(self.num_local_obs,),
            dtype=np.float32,
        )

        # =========================================================================
        # Estado Global para o Crítico Centralizado (CTDE - 53 dimensões):
        # Bola (4) + 3 Azuis (8 cada = 24) + 3 Amarelos (7 cada = 21) + Features Contextuais (4)
        # =========================================================================
        self.num_state_features = 4 + (8 * self.n_robots_blue) + (7 * self.n_robots_yellow) + 4
        self.state_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(self.num_state_features,),
            dtype=np.float32,
        )

        # Limites físicos dos atuadores SSL
        self.max_v = 1.5        # Velocidade linear máxima (m/s)
        self.max_w = 5.0       # Velocidade angular máxima (rad/s)
        self.kick_speed_x = 3.0 # Velocidade máxima do chute frontal (m/s)
        self.max_steps = 600    # Duração máxima do episódio (15 segundos a 40Hz)

        # Variáveis internas de rastreamento de passes e recompensas
        self.pass_in_progress = False
        self.last_touch_agent = None
        self.pass_start_step = 0
        self.shot_opp_active = False
        self.shot_own_active = False
        self.reward_shaping_total = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None
        self.pass_in_progress = False
        self.last_touch_agent = None
        self.pass_start_step = 0
        self.shot_opp_active = False
        self.shot_own_active = False
        self.reward_shaping_total = None

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.rsim.reset(initial_pos_frame)
        self.frame = self.rsim.get_frame()

        obs_dict = self._get_all_agent_observations()
        info_dict = {agent: {} for agent in self.agents}
        return obs_dict, info_dict

    def step(self, action_dict: Dict[str, np.ndarray]):
        self.steps += 1

        # Tratar compatibilidade caso receba array único ou dicionário
        if isinstance(action_dict, (np.ndarray, list)):
            act_blue_0 = action_dict[:4]
            act_blue_1 = action_dict[4:8] if len(action_dict) >= 8 else np.zeros(4, dtype=np.float32)
            actions = {"blue_0": act_blue_0, "blue_1": act_blue_1}
        else:
            actions = action_dict

        # Converter ações dos agentes para comandos do simulador
        commands: List[Robot] = self._get_multi_agent_commands(actions)
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Obter novo frame
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Atualizar estado de toque / passe
        self._update_pass_tracker(actions)

        # Obter observações locais e recompensas
        obs_dict = self._get_all_agent_observations()
        reward_dict, terminated, truncated = self._calculate_multi_agent_reward_and_done()

        if self.steps >= self.max_steps:
            truncated = True

        term_dict = {agent: terminated for agent in self.agents}
        trunc_dict = {agent: truncated for agent in self.agents}
        info_dict = {
            agent: {
                "reward_shaping": self.reward_shaping_total,
                "global_state": self.get_global_state(),
            }
            for agent in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    def get_global_state(self) -> np.ndarray:
        """
        Retorna o Estado Global Centralizado (s) para o Crítico Centralizado do MAPPO.
        Inclui posições e velocidades exatas de todos os robôs e da bola, além de métricas contextuais.
        """
        ball = self.frame.ball
        state = [
            self.norm_pos(ball.x),
            self.norm_pos(ball.y),
            self.norm_v(ball.v_x),
            self.norm_v(ball.v_y),
        ]

        # 3 robôs Azuis
        for i in range(self.n_robots_blue):
            r = self.frame.robots_blue[i]
            state.extend([
                self.norm_pos(r.x),
                self.norm_pos(r.y),
                np.sin(np.deg2rad(r.theta)),
                np.cos(np.deg2rad(r.theta)),
                self.norm_v(r.v_x),
                self.norm_v(r.v_y),
                self.norm_w(r.v_theta),
                1.0 if r.infrared else 0.0,
            ])

        # 3 robôs Amarelos
        for i in range(self.n_robots_yellow):
            r = self.frame.robots_yellow[i]
            state.extend([
                self.norm_pos(r.x),
                self.norm_pos(r.y),
                np.sin(np.deg2rad(r.theta)),
                np.cos(np.deg2rad(r.theta)),
                self.norm_v(r.v_x),
                self.norm_v(r.v_y),
                self.norm_w(r.v_theta),
            ])

        # Métricas globais contextuais
        dist_b0_ball = np.linalg.norm([self.frame.robots_blue[0].x - ball.x, self.frame.robots_blue[0].y - ball.y])
        dist_b1_ball = np.linalg.norm([self.frame.robots_blue[1].x - ball.x, self.frame.robots_blue[1].y - ball.y])
        dist_b0_b1 = np.linalg.norm([self.frame.robots_blue[0].x - self.frame.robots_blue[1].x, self.frame.robots_blue[0].y - self.frame.robots_blue[1].y])
        state.extend([
            self.norm_pos(dist_b0_ball),
            self.norm_pos(dist_b1_ball),
            self.norm_pos(dist_b0_b1),
            1.0 if self.pass_in_progress else 0.0,
        ])

        return np.array(state, dtype=np.float32)

    def _get_initial_positions_frame(self) -> Frame:
        """
        Define o posicionamento inicial estratégico para treinamento de atacantes cooperativos:
        - Bola na intermediária com Robô Azul 0 atrás da bola.
        - Robô Azul 1 posicionado aberto em zona de recepção ofensiva oposta.
        - Robô Azul 2 de suporte defensivo.
        - Goleiro Amarelo no gol e defensores amarelos posicionados.
        """
        frame = Frame()
        half_len = (self.field.length / 2) - 0.25
        half_wid = (self.field.width / 2) - 0.25

        # Bola no lado esquerdo / intermediária
        ball_x = random.uniform(-0.8, 0.2)
        ball_y = random.uniform(-half_wid * 0.6, half_wid * 0.6)
        frame.ball = Ball(x=ball_x, y=ball_y)

        # Robô Azul 0 (Passador / Atacante 1) posicionado atrás da bola
        b0_x = ball_x - random.uniform(0.20, 0.40)
        b0_y = ball_y + random.uniform(-0.15, 0.15)
        angle_b0 = np.rad2deg(math.atan2(ball_y - b0_y, ball_x - b0_x))
        frame.robots_blue[0] = Robot(
            x=np.clip(b0_x, -half_len, half_len),
            y=np.clip(b0_y, -half_wid, half_wid),
            theta=angle_b0,
        )

        # Robô Azul 1 (Receptor / Atacante 2) posicionado no flanco ofensivo oposto
        sign_y = -1.0 if ball_y >= 0 else 1.0
        b1_x = random.uniform(0.0, 1.2)
        b1_y = sign_y * random.uniform(0.4, half_wid * 0.85)
        target_goal_angle = np.rad2deg(math.atan2(-b1_y, (self.field.length / 2) - b1_x))
        frame.robots_blue[1] = Robot(
            x=b1_x,
            y=b1_y,
            theta=target_goal_angle,
        )

        # Robô Azul 2 (Apoio estático na defesa)
        frame.robots_blue[2] = Robot(x=-1.50, y=0.0, theta=0.0)

        # Goleiro Amarelo 0 (na linha do gol)
        gk_x = (self.field.length / 2) - 0.12
        frame.robots_yellow[0] = Robot(x=gk_x, y=random.uniform(-0.25, 0.25), theta=180.0)

        # Defensores Amarelos 1 e 2 (em marcação posicional)
        frame.robots_yellow[1] = Robot(x=0.80, y=0.65, theta=180.0)
        frame.robots_yellow[2] = Robot(x=0.80, y=-0.65, theta=180.0)

        return frame

    def convert_actions(self, action, angle):
        """Desnormaliza velocidades do referencial global para o referencial local do robô."""
        v_x = float(action[0] * self.max_v)
        v_y = float(action[1] * self.max_v)
        v_theta = float(action[2] * self.max_w)

        # Rotação para referencial local
        v_x_local = v_x * np.cos(angle) + v_y * np.sin(angle)
        v_y_local = -v_x * np.sin(angle) + v_y * np.cos(angle)

        # Normalização pela velocidade máxima permitida
        v_norm = np.linalg.norm([v_x_local, v_y_local])
        c = 1.0 if v_norm < self.max_v else (self.max_v / (v_norm + 1e-8))
        v_x_local, v_y_local = v_x_local * c, v_y_local * c

        return v_x_local, v_y_local, v_theta

    def _get_multi_agent_commands(self, actions_dict: Dict[str, np.ndarray]) -> List[Robot]:
        """Gera a lista de comandos para os 6 robôs do simulador."""
        commands = []

        # Robô Azul 0
        act0 = actions_dict.get("blue_0", np.zeros(4, dtype=np.float32))
        angle0 = np.deg2rad(self.frame.robots_blue[0].theta)
        v_x0, v_y0, v_th0 = self.convert_actions(act0, angle0)
        kick0 = self.kick_speed_x if act0[3] > 0 else 0.0
        commands.append(
            Robot(
                yellow=False,
                id=0,
                v_x=v_x0,
                v_y=v_y0,
                v_theta=v_th0,
                kick_v_x=kick0,
                kick_v_z=0.0,
                dribbler=False,
            )
        )

        # Robô Azul 1
        act1 = actions_dict.get("blue_1", np.zeros(4, dtype=np.float32))
        angle1 = np.deg2rad(self.frame.robots_blue[1].theta)
        v_x1, v_y1, v_th1 = self.convert_actions(act1, angle1)
        kick1 = self.kick_speed_x if act1[3] > 0 else 0.0
        commands.append(
            Robot(
                yellow=False,
                id=1,
                v_x=v_x1,
                v_y=v_y1,
                v_theta=v_th1,
                kick_v_x=kick1,
                kick_v_z=0.0,
                dribbler=False,
            )
        )

        # Robô Azul 2 (Apoio estático)
        commands.append(
            Robot(
                yellow=False,
                id=2,
                v_x=0.0,
                v_y=0.0,
                v_theta=0.0,
                kick_v_x=0.0,
                kick_v_z=0.0,
                dribbler=False,
            )
        )

        # Robôs Amarelos (Goleiro e Defensores)
        for i in range(self.n_robots_yellow):
            commands.append(
                Robot(
                    yellow=True,
                    id=i,
                    v_x=0.0,
                    v_y=0.0,
                    v_theta=0.0,
                    kick_v_x=0.0,
                    kick_v_z=0.0,
                    dribbler=False,
                )
            )

        return commands

    def _is_inside_penalty_area(self, x: float, y: float, is_yellow_area: bool = True) -> bool:
        """Verifica se (x, y) está dentro da área de pênalti SSL-EL (1.35m x 0.50m)."""
        half_pen_w = self.field.penalty_width / 2  # 0.675m
        pen_len = self.field.penalty_length       # 0.50m
        half_field_len = self.field.length / 2    # 2.25m

        if is_yellow_area:
            in_x = (x > half_field_len - pen_len)
        else:
            in_x = (x < -half_field_len + pen_len)

        in_y = abs(y) < half_pen_w
        return in_x and in_y

    def _update_pass_tracker(self, actions_dict: Dict[str, np.ndarray]):
        """Rastreia dinâmica de passes entre os dois robôs atacantes azuis."""
        ball = self.frame.ball
        r0 = self.frame.robots_blue[0]
        r1 = self.frame.robots_blue[1]

        d0 = float(np.linalg.norm([r0.x - ball.x, r0.y - ball.y]))
        d1 = float(np.linalg.norm([r1.x - ball.x, r1.y - ball.y]))

        # Identificar toques
        if d0 < 0.15 or r0.infrared:
            self.last_touch_agent = "blue_0"
        elif d1 < 0.15 or r1.infrared:
            self.last_touch_agent = "blue_1"

        # Detectar início de passe (chute de um robô em direção ao outro)
        ball_speed = float(np.linalg.norm([ball.v_x, ball.v_y]))
        if ball_speed > 0.8:
            if self.last_touch_agent == "blue_0":
                vec_b_to_r1 = np.array([r1.x - ball.x, r1.y - ball.y])
                dist_r1 = np.linalg.norm(vec_b_to_r1)
                if dist_r1 > 0.3:
                    dir_r1 = vec_b_to_r1 / dist_r1
                    ball_dir = np.array([ball.v_x, ball.v_y]) / ball_speed
                    if np.dot(ball_dir, dir_r1) > 0.70:
                        self.pass_in_progress = True
                        self.pass_start_step = self.steps
            elif self.last_touch_agent == "blue_1":
                vec_b_to_r0 = np.array([r0.x - ball.x, r0.y - ball.y])
                dist_r0 = np.linalg.norm(vec_b_to_r0)
                if dist_r0 > 0.3:
                    dir_r0 = vec_b_to_r0 / dist_r0
                    ball_dir = np.array([ball.v_x, ball.v_y]) / ball_speed
                    if np.dot(ball_dir, dir_r0) > 0.70:
                        self.pass_in_progress = True
                        self.pass_start_step = self.steps

        if self.pass_in_progress and (self.steps - self.pass_start_step > 50 or ball_speed < 0.2):
            self.pass_in_progress = False

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Gera o vetor de observação local descentralizado para o agente especificado."""
        is_blue_0 = (agent_id == "blue_0")
        me_idx = 0 if is_blue_0 else 1
        partner_idx = 1 if is_blue_0 else 0

        me = self.frame.robots_blue[me_idx]
        partner = self.frame.robots_blue[partner_idx]
        support = self.frame.robots_blue[2]
        ball = self.frame.ball

        # 1. Próprio robô (8)
        obs = [
            self.norm_pos(me.x),
            self.norm_pos(me.y),
            np.sin(np.deg2rad(me.theta)),
            np.cos(np.deg2rad(me.theta)),
            self.norm_v(me.v_x),
            self.norm_v(me.v_y),
            self.norm_w(me.v_theta),
            1.0 if me.infrared else 0.0,
        ]

        # 2. Bola em coordenadas relativas e globais (6)
        rel_bx = ball.x - me.x
        rel_by = ball.y - me.y
        dist_b = float(np.linalg.norm([rel_bx, rel_by]))
        ang_b = float(math.atan2(rel_by, rel_bx) - np.deg2rad(me.theta))
        obs.extend([
            self.norm_pos(rel_bx),
            self.norm_pos(rel_by),
            self.norm_v(ball.v_x),
            self.norm_v(ball.v_y),
            self.norm_pos(dist_b),
            float(math.cos(ang_b)),
        ])

        # 3. Companheiro parceiro ativo (8)
        rel_px = partner.x - me.x
        rel_py = partner.y - me.y
        obs.extend([
            self.norm_pos(rel_px),
            self.norm_pos(rel_py),
            np.sin(np.deg2rad(partner.theta)),
            np.cos(np.deg2rad(partner.theta)),
            self.norm_v(partner.v_x),
            self.norm_v(partner.v_y),
            self.norm_w(partner.v_theta),
            1.0 if partner.infrared else 0.0,
        ])

        # 4. Companheiro estático de suporte (2)
        obs.extend([
            self.norm_pos(support.x - me.x),
            self.norm_pos(support.y - me.y),
        ])

        # 5. Adversários Amarelos relativos (6)
        for i in range(self.n_robots_yellow):
            opp = self.frame.robots_yellow[i]
            obs.extend([
                self.norm_pos(opp.x - me.x),
                self.norm_pos(opp.y - me.y),
            ])

        # 6. One-hot Agent ID (2)
        obs.extend([1.0, 0.0] if is_blue_0 else [0.0, 1.0])

        # 7. Contexto de Passe / Posicionamento (3)
        has_possession = 1.0 if (dist_b < 0.18 or me.infrared) else 0.0
        vec_p_to_me = np.array([me.x - partner.x, me.y - partner.y])
        dist_p_to_me = np.linalg.norm(vec_p_to_me)
        dir_p = np.array([np.cos(np.deg2rad(partner.theta)), np.sin(np.deg2rad(partner.theta))])
        in_partner_cone = 1.0 if (dist_p_to_me > 0.3 and np.dot(vec_p_to_me / max(dist_p_to_me, 1e-6), dir_p) > 0.5) else 0.0
        
        ball_speed = np.linalg.norm([ball.v_x, ball.v_y])
        ball_towards_me = 0.0
        if ball_speed > 0.6:
            vec_b_to_me = np.array([me.x - ball.x, me.y - ball.y])
            d_b2me = np.linalg.norm(vec_b_to_me)
            if d_b2me > 0.1:
                ball_towards_me = 1.0 if np.dot(np.array([ball.v_x, ball.v_y]) / ball_speed, vec_b_to_me / d_b2me) > 0.7 else 0.0

        obs.extend([has_possession, in_partner_cone, ball_towards_me])

        return np.array(obs, dtype=np.float32)

    def _get_all_agent_observations(self) -> Dict[str, np.ndarray]:
        """Retorna observações descentralizadas para todos os agentes."""
        return {agent: self._get_agent_observation(agent) for agent in self.agents}

    def _calculate_multi_agent_reward_and_done(self) -> Tuple[Dict[str, float], bool, bool]:
        """
        FUNÇÃO DE RECOMPENSA COOPERATIVA COMPLETA, PROPORCIONAL E BLINDADA:
        - Gol Marcado: +40.0 a +50.0 compartilhado (bônus para chute veloz).
        - Passe Conectado: +15.0 compartilhado (robô chuta para o parceiro e este domina).
        - Aproximação e Contorno (PBRS): até +-1.0 (alvo a 9.5cm atrás da bola ou contorno lateral).
        - Avanço da Bola ao Gol (PBRS): até +-2.0.
        - Alinhamento Diferencial: até +-0.5 orientado para o gol.
        - Disparo do Chute por Aceleração/Impacto Real: +2.0 a +4.0.
        - Sensor Infravermelho Frontal: +0.05 contínuo por ter a bola no chutador.
        - Desmarcação do 2º Atacante: incentiva infiltração em linha de passe aberta.
        - Barreira Repulsiva da Área Adversária a 20cm da linha.
        - Anti-Colisão entre Companheiros: -0.5 para não disputarem a bola.
        - Penalidades: Invasão de Área (-5.0), Gol Contra (-15.0), Bola Fora (-2.0), Fora de Campo (-5.0).
        - Custo de Tempo e Energia: -0.005 por passo.
        """
        done = False
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal": 0.0,
                "pass_success": 0.0,
                "shot_on_goal": 0.0,
                "shot_attempt": 0.0,
                "ball_grad": 0.0,
                "move_to_ball": 0.0,
                "alignment": 0.0,
                "kick_action": 0.0,
                "push_to_goal": 0.0,
                "infrared": 0.0,
                "receiver_positioning": 0.0,
                "collision_teammates": 0.0,
                "out_of_bounds": 0.0,
                "area_violation": 0.0,
                "ball_out": 0.0,
                "energy": 0.0,
            }

        ball = self.frame.ball
        r0 = self.frame.robots_blue[0]
        r1 = self.frame.robots_blue[1]
        half_len = self.field.length / 2   # 2.25m
        half_wid = self.field.width / 2    # 1.50m
        goal_w = self.field.goal_width / 2 # 0.35m

        shared_reward = 0.0
        r0_reward = 0.0
        r1_reward = 0.0

        # ----------------------------------------------------
        # 1. Eventos Terminais Globais (Gols e Saídas)
        # ----------------------------------------------------
        # Gol Marcado Válido (+40.0 a +50.0 compartilhado)
        if ball.x > half_len and abs(ball.y) < goal_w:
            goal_rw = 40.0
            if self.shot_opp_active or ball.v_x > 0.8:
                goal_rw += 10.0  # Total 50.0 por gol de chute potente
            shared_reward += goal_rw
            done = True
            self.reward_shaping_total["goal"] += goal_rw
            reward_dict = {a: shared_reward for a in self.agents}
            return reward_dict, done, False

        # Gol Sofrido / Gol Contra (-15.0 compartilhado)
        if ball.x < -half_len and abs(ball.y) < goal_w:
            shared_reward -= 15.0
            done = True
            self.reward_shaping_total["goal"] -= 15.0
            reward_dict = {a: shared_reward for a in self.agents}
            return reward_dict, done, False

        # Bola fora pela Linha de Fundo Adversária (+0.5 conclusão de ataque)
        if ball.x > half_len and abs(ball.y) >= goal_w:
            shared_reward += 0.5
            done = True
            reward_dict = {a: shared_reward for a in self.agents}
            return reward_dict, done, False

        # Bola fora das laterais ou defesa (-2.0)
        if abs(ball.x) > (half_len + 0.1) or abs(ball.y) > (half_wid + 0.1):
            shared_reward -= 2.0
            done = True
            self.reward_shaping_total["ball_out"] -= 2.0
            reward_dict = {a: shared_reward for a in self.agents}
            return reward_dict, done, False

        # ----------------------------------------------------
        # 2. Violação de Regras e Limites de Campo
        # ----------------------------------------------------
        for i, robot in enumerate([r0, r1]):
            # Fora do campo
            if abs(robot.x) > (half_len + 0.05) or abs(robot.y) > (half_wid + 0.05):
                shared_reward -= 5.0
                done = True
                self.reward_shaping_total["out_of_bounds"] -= 5.0
                reward_dict = {a: shared_reward for a in self.agents}
                return reward_dict, done, False

            # Invasão de área de pênalti
            if self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=True) or \
               self._is_inside_penalty_area(robot.x, robot.y, is_yellow_area=False):
                shared_reward -= 5.0
                done = True
                self.reward_shaping_total["area_violation"] -= 5.0
                reward_dict = {a: shared_reward for a in self.agents}
                return reward_dict, done, False

        # ----------------------------------------------------
        # 3. Penalidade de Colisão entre Companheiros de Equipe
        # ----------------------------------------------------
        dist_teammates = float(np.linalg.norm([r0.x - r1.x, r0.y - r1.y]))
        if dist_teammates < 0.20:
            col_pen = -0.5
            shared_reward += col_pen
            self.reward_shaping_total["collision_teammates"] += col_pen

        # ----------------------------------------------------
        # 4. Detecção e Recompensa de PASSE BEM-SUCEDIDO (+15.0 Compartilhado!)
        # ----------------------------------------------------
        if self.pass_in_progress:
            if self.last_touch_agent == "blue_0" and (r1.infrared or float(np.linalg.norm([r1.x - ball.x, r1.y - ball.y])) < 0.15):
                pass_rw = 15.0
                shared_reward += pass_rw
                self.pass_in_progress = False
                self.reward_shaping_total["pass_success"] += pass_rw
            elif self.last_touch_agent == "blue_1" and (r0.infrared or float(np.linalg.norm([r0.x - ball.x, r0.y - ball.y])) < 0.15):
                pass_rw = 15.0
                shared_reward += pass_rw
                self.pass_in_progress = False
                self.reward_shaping_total["pass_success"] += pass_rw

        # ----------------------------------------------------
        # 5. Geometria de Posicionamento e Contorno da Bola
        # ----------------------------------------------------
        goal_target = np.array([half_len, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        dist_b2g = float(np.linalg.norm(goal_target - ball_pos))
        dir_b2g = (goal_target - ball_pos) / max(dist_b2g, 1e-6)
        perp_dir = np.array([-dir_b2g[1], dir_b2g[0]])

        dist_r0_b = float(np.linalg.norm([r0.x - ball.x, r0.y - ball.y]))
        dist_r1_b = float(np.linalg.norm([r1.x - ball.x, r1.y - ball.y]))

        # Identificar líder (mais perto da bola) e receptor
        if dist_r0_b <= dist_r1_b:
            lead_robot = r0
            wing_robot = r1
            lead_agent = "blue_0"
            lead_idx = 0
            wing_idx = 1
        else:
            lead_robot = r1
            wing_robot = r0
            lead_agent = "blue_1"
            lead_idx = 1
            wing_idx = 0

        lead_pos = np.array([lead_robot.x, lead_robot.y])
        vec_r2b = ball_pos - lead_pos
        dist_r2b = float(np.linalg.norm(vec_r2b))
        proj_behind = float(np.dot(vec_r2b, dir_b2g))

        # Alvo do condutor: 9.5cm atrás da bola apontando para o gol ou contorno lateral (15cm)
        if proj_behind < -0.05 and dist_r2b < 0.30:
            y_rel = float(np.dot(lead_pos - ball_pos, perp_dir))
            sign_y = 1.0 if y_rel >= 0 else -1.0
            target_pos = ball_pos - 0.12 * dir_b2g + sign_y * 0.15 * perp_dir
        else:
            target_pos = ball_pos - 0.095 * dir_b2g
        cur_dist_target = float(np.linalg.norm(target_pos - lead_pos))

        # Orientação angular do condutor em relação ao gol
        target_angle = math.atan2(dir_b2g[1], dir_b2g[0])
        lead_theta_rad = math.radians(lead_robot.theta)
        align_cos = math.cos(lead_theta_rad - target_angle)

        # Velocidade da bola ao gol
        ball_vel = np.array([ball.v_x, ball.v_y])
        ball_v_to_goal = float(np.dot(ball_vel, dir_b2g))

        # ----------------------------------------------------
        # 6. Tiro em Alta Velocidade no Alvo (+6.0 Evento)
        # ----------------------------------------------------
        if ball.v_x > 0.8 and ball_v_to_goal > 0.5:
            t_opp = (half_len - ball.x) / max(ball.v_x, 1e-6)
            if 0 < t_opp < 2.0:
                y_proj = ball.y + (ball.v_y * t_opp)
                if abs(y_proj) <= (goal_w + 0.12) and not self.shot_opp_active:
                    self.shot_opp_active = True
                    shot_bonus = 6.0
                    shared_reward += shot_bonus
                    self.reward_shaping_total["shot_on_goal"] += shot_bonus
        elif ball.v_x < 0.3:
            self.shot_opp_active = False

        # ----------------------------------------------------
        # 7. Potenciais Diferenciais PBRS (Avanço, Aproximação, Alinhamento, Impacto)
        # ----------------------------------------------------
        if self.last_frame is not None:
            last_ball = self.last_frame.ball
            last_ball_pos = np.array([last_ball.x, last_ball.y])
            last_dist_b2g = float(np.linalg.norm(goal_target - last_ball_pos))
            last_dir_b2g = (goal_target - last_ball_pos) / max(last_dist_b2g, 1e-6)
            last_perp_dir = np.array([-last_dir_b2g[1], last_dir_b2g[0]])

            last_lead_rbt = self.last_frame.robots_blue[lead_idx]
            last_lead_pos = np.array([last_lead_rbt.x, last_lead_rbt.y])
            last_vec_r2b = last_ball_pos - last_lead_pos
            last_dist_r2b = float(np.linalg.norm(last_vec_r2b))
            last_proj_behind = float(np.dot(last_vec_r2b, last_dir_b2g))

            if last_proj_behind < -0.05 and last_dist_r2b < 0.30:
                last_y_rel = float(np.dot(last_lead_pos - last_ball_pos, last_perp_dir))
                last_sign_y = 1.0 if last_y_rel >= 0 else -1.0
                last_target_pos = last_ball_pos - 0.12 * last_dir_b2g + last_sign_y * 0.15 * last_perp_dir
            else:
                last_target_pos = last_ball_pos - 0.095 * last_dir_b2g
            last_dist_target = float(np.linalg.norm(last_target_pos - last_lead_pos))

            # A) Aproximação e Contorno (até +-1.0)
            diff_move = (last_dist_target - cur_dist_target) * 2.5
            if proj_behind > 0.0 and dist_r2b < 0.25 and align_cos > 0.1:
                diff_contact = (last_dist_r2b - dist_r2b) * 3.0
                if diff_contact > 0:
                    diff_move = max(diff_move, diff_contact)
            r_move = float(np.clip(diff_move, -1.0, 1.0))
            if ball_v_to_goal > 0.6:
                r_move = max(0.0, r_move)

            if lead_agent == "blue_0":
                r0_reward += r_move
            else:
                r1_reward += r_move
            self.reward_shaping_total["move_to_ball"] += r_move

            # B) Avanço da Bola ao Gol (até +-2.0 Compartilhado)
            diff_ball_goal = (last_dist_b2g - dist_b2g) * 4.0
            r_ball_grad = float(np.clip(diff_ball_goal, -2.0, 2.0))
            shared_reward += r_ball_grad
            self.reward_shaping_total["ball_grad"] += r_ball_grad

            # C) Alinhamento Diferencial com o Gol (até +-0.5)
            last_target_angle = math.atan2(last_dir_b2g[1], last_dir_b2g[0])
            last_align_cos = math.cos(math.radians(last_lead_rbt.theta) - last_target_angle)
            diff_align = (align_cos - last_align_cos) * 0.5
            r_align = float(np.clip(diff_align, -0.5, 0.5))
            if lead_agent == "blue_0":
                r0_reward += r_align
            else:
                r1_reward += r_align
            self.reward_shaping_total["alignment"] += r_align

            # D) Disparo do Chute com Transferência Real de Momento (+2.0 a +4.0)
            last_ball_vel = np.array([last_ball.v_x, last_ball.v_y])
            last_ball_v_to_goal = float(np.dot(last_ball_vel, last_dir_b2g))
            accel_ball_to_goal = ball_v_to_goal - last_ball_v_to_goal
            if accel_ball_to_goal > 0.3 and (dist_r2b < 0.25 or lead_robot.infrared):
                impulse_rw = float(2.0 * min(accel_ball_to_goal, 2.0))
                if self.sent_commands is not None and self.sent_commands[lead_idx].kick_v_x > 0:
                    impulse_rw += 2.0
                    self.reward_shaping_total["kick_action"] += 2.0
                shared_reward += impulse_rw
                self.reward_shaping_total["push_to_goal"] += impulse_rw

            # E) Desmarcação do 2º Atacante (Infiltração Ofensiva)
            wing_target_x = np.clip(ball.x + 0.8, -0.5, half_len - 0.6)
            wing_target_y = -0.7 if ball.y >= 0 else 0.7
            wing_target_pos = np.array([wing_target_x, wing_target_y])
            cur_wing_dist = float(np.linalg.norm([wing_robot.x - wing_target_pos[0], wing_robot.y - wing_target_pos[1]]))

            last_wing_rbt = self.last_frame.robots_blue[wing_idx]
            last_wing_dist = float(np.linalg.norm([last_wing_rbt.x - wing_target_pos[0], last_wing_rbt.y - wing_target_pos[1]]))
            diff_wing = (last_wing_dist - cur_wing_dist) * 1.5
            r_wing_pos = float(np.clip(diff_wing, -0.5, 0.5))

            if lead_agent == "blue_0":
                r1_reward += r_wing_pos
            else:
                r0_reward += r_wing_pos
            self.reward_shaping_total["receiver_positioning"] += r_wing_pos

        # ----------------------------------------------------
        # 8. Sensor Infravermelho Frontal (+0.05 contínuo)
        # ----------------------------------------------------
        for i, (agent_name, robot) in enumerate([("blue_0", r0), ("blue_1", r1)]):
            if robot.infrared:
                infra_rw = 0.05
                if agent_name == "blue_0":
                    r0_reward += infra_rw
                else:
                    r1_reward += infra_rw
                self.reward_shaping_total["infrared"] += infra_rw

        # ----------------------------------------------------
        # 9. Barreira Repulsiva da Área Adversária (a 20cm da linha)
        # ----------------------------------------------------
        penalty_line_x = half_len - self.field.penalty_length  # 1.75m
        for rbt in [r0, r1]:
            if rbt.x > (penalty_line_x - 0.20) and abs(rbt.y) < (self.field.penalty_width / 2 + 0.10):
                dist_near = rbt.x - (penalty_line_x - 0.20)
                shared_reward -= 0.20 * dist_near

        # ----------------------------------------------------
        # 10. Penalidade Suave de Tempo e Energia (-0.005 por passo)
        # ----------------------------------------------------
        time_pen = -0.005
        energy_0 = float(1e-5 * (abs(r0.v_x) + abs(r0.v_y) + abs(r0.v_theta)))
        energy_1 = float(1e-5 * (abs(r1.v_x) + abs(r1.v_y) + abs(r1.v_theta)))
        r0_reward += time_pen - energy_0
        r1_reward += time_pen - energy_1
        self.reward_shaping_total["energy"] -= (abs(time_pen) + energy_0 + energy_1)

        # Recompensa total
        reward_dict = {
            "blue_0": float(shared_reward + r0_reward),
            "blue_1": float(shared_reward + r1_reward),
        }

        return reward_dict, done, False

    def _frame_to_observations(self):
        """Compatibilidade com SSLBaseEnv."""
        return self._get_agent_observation("blue_0")

    def _get_commands(self, action):
        """Compatibilidade com SSLBaseEnv."""
        return self._get_multi_agent_commands({"blue_0": action, "blue_1": np.zeros(4, dtype=np.float32)})

    def _calculate_reward_and_done(self):
        """Compatibilidade com SSLBaseEnv."""
        r_dict, d, _ = self._calculate_multi_agent_reward_and_done()
        return r_dict["blue_0"], d


# Alias para retrocompatibilidade
SSLELCoopEnv = SSLELCooperationAttackerEnv
