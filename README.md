# ⚽ RL-Soccer: Treinamento Cooperativo de Atacantes em SSL-EL com MAPPO

Ambiente de **Aprendizado por Reforço Multiagente (MARL)** baseado no algoritmo **MAPPO (Multi-Agent PPO)** com arquitetura **CTDE (Centralized Training, Decentralized Execution)** para a liga **SSL Entry Level (SSL-EL)** da RoboCup.

Dois robôs atacantes azuis aprendem a cooperar dinamicamente (trocar passes rápidos, desmarcar, avançar com a bola e finalizar a gol) contra um goleiro e defensores amarelos, respeitando todas as regras oficiais de campo da SSL Entry Level.

---

## 📋 Índice
1. [Visão Geral do Projeto](#-visão-geral-do-projeto)
2. [Arquitetura MAPPO & CTDE](#-arquitetura-mappo--ctde)
3. [Dimensões do Campo (SSL Entry Level)](#-dimensões-do-campo-ssl-entry-level)
4. [Espaços de Observação e Ação](#-espaços-de-observação-e-ação)
5. [Função de Recompensa](#-função-de-recompensa-blindada-e-proporcional)
6. [Formatos dos Modelos (.pt e .onnx)](#-formatos-dos-modelos-pt-e-onnx)
7. [Instalação e Requisitos](#-instalação-e-requisitos)
8. [Como Executar](#-como-executar)
   - [Treinamento do Zero](#1-iniciar-o-treinamento-5-milhões-de-passos)
   - [Continuar Treinamento (--resume)](#2-retomar-o-treinamento-de-onde-parou---resume)
   - [Monitoramento no TensorBoard](#3-monitoramento-em-tempo-real-com-tensorboard)
   - [Acompanhamento Visual ao Vivo (play_attacker.py)](#4-acompanhamento-ao-vivo-play_attackerpy)
   - [Avaliação Pontual (play.py)](#5-avaliar-um-modelo-específico-playpy)
9. [Estrutura do Repositório](#-estrutura-do-repositório)


---

## 🧠 Visão Geral do Projeto

* **Liga**: Small Size League - Entry Level (SSL-EL).
* **Robôs Ativos**:
  * 🔵 **Time Azul**: `blue_0` e `blue_1` (Atacantes cooperativos controlados por RL), `blue_2` (Apoio estático).
  * 🟡 **Time Amarelo**: `yellow_0` (Goleiro ativo com marcação em Y na meta e expulsão da bola da área), `yellow_1` e `yellow_2` (Defensores estáticos).
* **Algoritmo**: **MAPPO** Contínuo Gaussiano.
* **Atuadores**: `dribbler` **desativado** (100% desligado) e chutador frontal (`kick_v_x`) acionado por impacto.

---

## 🏗 Arquitetura MAPPO & CTDE

O projeto utiliza a arquitetura **CTDE (Centralized Training, Decentralized Execution)**:

```
========================================================================================
                               TREINAMENTO (CTDE)
========================================================================================

                 Estado Global s (53 dimensões)
               [Campo Completo, Bola, 6 Robôs, Vels]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Crítico Centralizado │ ──► V(s) (Guia o Passe e Cooperação)
                     │  (CentralizedCritic)  │
                     └───────────────────────┘

                 Observação Local o_0              Observação Local o_1
                    (35 dimensões)                    (35 dimensões)
                          │                                 │
                          ▼                                 ▼
               ┌───────────────────────┐         ┌───────────────────────┐
               │    Ator Descentral.   │         │    Ator Descentral.   │
               │    (ContinuousActor)  │         │    (ContinuousActor)  │
               └───────────────────────┘         └───────────────────────┘
                          │                                 │
                          ▼                                 ▼
                     Ação Atacante 0                   Ação Atacante 1
                   [vx, vy, vth, kick]               [vx, vy, vth, kick]
========================================================================================
                               EXECUÇÃO / JOGO REAL
   (Apenas o Ator é executado localmente nos robôs ou exportado em ONNX para C++/ROS2)
========================================================================================
```

* **Ator Descentralizado**: Cada robô decide suas velocidades e chutes observando apenas seu vetor local de 35 dimensões.
* **Crítico Centralizado**: Observa todas as 53 variáveis de estado global do campo durante o treino, ensinando ao robô passador que a corrida de infiltração do parceiro resultará em recompensa coletiva.

---

## 📐 Dimensões do Campo (SSL Entry Level)

O ambiente implementa com precisão milimétrica as regras e dimensões da categoria **SSL-EL**:

| Elemento | Dimensão Oficial SSL-EL |
| :--- | :--- |
| **Comprimento $\times$ Largura** | $4.50\text{ m} \times 3.00\text{ m}$ (Margem: $0.30\text{ m}$) |
| **Área de Pênalti (Defesa e Ataque)** | $1.35\text{ m} \times 0.50\text{ m}$ (Retangular) |
| **Meta / Traves do Gol** | Largura: $0.70\text{ m}$, Profundidade: $0.18\text{ m}$ |
| **Círculo Central** | Raio: $0.50\text{ m}$ |
| **Dimensões do Robô** | Raio: $0.09\text{ m}$ (Diâmetro: $18\text{ cm}$) |
| **Duração do Episódio** | $800\text{ passos}$ a $40\text{ Hz}$ ($20.0\text{ segundos}$) |

---

## 🎯 Espaços de Observação e Ação

### 1. Observação Local de Cada Agente (35 Dimensões)
* Posição $(x, y)$ e velocidade $(v_x, v_y)$ da bola relativas ao robô.
* Distância e ângulo relativo até o centro do gol adversário.
* Posição $(x, y)$, velocidade e ângulo $\theta$ relativos do parceiro de equipe.
* Posição $(x, y)$ e velocidade relativas do goleiro e dos 2 defensores amarelos.
* Posição $(x, y)$, velocidades $(v_x, v_y, v_\theta)$ e orientação $\cos \theta, \sin \theta$ globais do próprio robô.
* **Sensor Infravermelho Frontal**: `1.0` se a bola estiver alojada na cavidade do chutador, `0.0` caso contrário.
* Identificador One-Hot (`[1, 0]` para `blue_0` e `[0, 1]` para `blue_1`).

### 2. Estado Global Centralizado (53 Dimensões)
Consumido exclusivamente pelo Crítico Central durante o cálculo de vantagens GAE. Contém as coordenadas absolutas e velocidades de todos os 6 robôs, da bola e de distâncias relativas entre todos os elementos.

### 3. Ações Contínuas de Cada Agente (4 Dimensões em $[-1.0, 1.0]$)
1. $a_0$: $v_x$ (Velocidade longitudinal em $\text{m/s}$)
2. $a_1$: $v_y$ (Velocidade lateral em $\text{m/s}$)
3. $a_2$: $v_\theta$ (Velocidade angular de rotação em $\text{rad/s}$)
4. $a_3$: $kick\_x$ (Acionamento do chutador frontal quando $> 0$)

---

## 🏆 Função de Recompensa Blindada e Proporcional

A função de recompensa foi desenhada para guiar a cooperação, punir infrações e **eliminar 100% o *reward hacking***:

| Componente | Tipo | Valor | O que ensina / Finalidade |
| :--- | :--- | :--- | :--- |
| **Gol Marcado Válido** | Terminal | $+40.0 \text{ a } +50.0$ | Super recompensa compartilhada ($+10.0$ de bônus por chute potente). |
| **Passe Conectado** | Evento | $+15.0$ | Bônus compartilhado quando um robô chuta para o parceiro e este domina. |
| **Tiro Veloz no Alvo** | Evento | $+6.0$ | Disparo veloz com trajetória linear no gol (com trava anti-looping). |
| **Disparo com Impacto Real** | Evento | $+2.0 \text{ a } +4.0$ | Recompensa por aceleração mecânica real transmitida à bola ($\Delta v > 0.3\text{ m/s}$). |
| **Sensor Infravermelho** | Contínuo | $+0.05$ | Bônus contínuo por ter a bola perfeitamente alojada de frente no chutador. |
| **Aproximação e Contorno** | PBRS $\Delta\Phi$ | até $\pm 1.0$ | Alvo a $9.5\text{ cm}$ atrás da bola ou contorno lateral ($15\text{ cm}$) se ultrapassar a bola. |
| **Avanço da Bola ao Gol** | PBRS $\Delta\Phi$ | até $\pm 2.0$ | Recompensa proporcional por empurrar a bola em direção à meta. |
| **Alinhamento com o Gol** | PBRS $\Delta\Phi$ | até $\pm 0.5$ | Recompensa diferencial de orientação angular apontando para o gol adversário. |
| **Desmarcação do 2º Atacante** | PBRS $\Delta\Phi$ | até $\pm 0.5$ | Recompensa o atacante sem a bola por se infiltrar em linha de passe aberta. |
| **Barreira Repulsiva da Área** | Contínuo | $< 0$ | Barreira repulsiva a $20\text{ cm}$ da linha da área adversária para chutar de fora. |
| **Anti-Colisão Companheiro** | Contínuo | $-0.5$ | Penaliza aproximação mútua excessiva ($< 20\text{ cm}$) entre os atacantes azuis. |
| **Invasão de Área de Pênalti** | Terminal | $-5.0$ | Penalidade por invadir a área de $1.35\text{ m} \times 0.50\text{ m}$ (falta oficial). |
| **Gol Contra / Sofrido** | Terminal | $-15.0$ | Penalidade caso a bola entre no próprio gol. |
| **Bola Fora** | Terminal | $-2.0$ | Penalidade por chutar para fora dos limites do campo. |
| **Tempo e Energia** | Contínuo | $-0.005 / \text{step}$ | Penalidade suave que força os agentes a concluírem as jogadas com rapidez. |

> [!IMPORTANT]
> **Blindagem Contra Reward Hacking no Chute**:
> O robô **nunca ganha pontos por simplesmente acionar o kicker no ar**. O bônus de disparo só é acionado se a bola sofrer aceleração real positiva ($\Delta v_{\text{bola}} > 0.3\text{ m/s}$) e o robô estiver a menos de $25\text{ cm}$ da bola ou com o sensor infravermelho ativado.

---

## 💾 Formatos dos Modelos (.pt e .onnx)

A cada checkpoint (a cada $100.000$ passos por padrão) e ao término do treinamento, o sistema salva **automaticamente** os modelos em dois formatos:

1. **Arquivo PyTorch (`.pt`)**:
   * Contém os pesos do Ator, do Crítico Centralizado, o estado interno dos otimizadores Adam (momentum/velocidade) e metadados de passos, gols e passes.
   * Permite **retomar o treinamento (`--resume`)** perfeitamente sem perder o progresso.
2. **Arquivo ONNX (`.onnx`)**:
   * Exporta o grafo do Ator (`ActorONNXWrapper`) com entrada `[batch, 35]` e saída `[batch, 4]`.
   * Pronto para implantação em tempo real com **ONNX Runtime** em **C++, Python, ROS2 ou grSim** sem depender do PyTorch.

---

## 📦 Instalação e Requisitos

> [!IMPORTANT]
> **Aceleração por GPU NVIDIA (CUDA 12.1+ / RTX 3060)**:
> Esta branch (`gpu`) é otimizada para placas NVIDIA (como RTX 3060 / Ampere), habilitando automaticamente **Tensor Cores (TF32)** e **cuDNN Benchmark** para velocidade máxima de treinamento.

### Passo a Passo de Instalação na Máquina com GPU:

```bash
# 1. Clonar o repositório e entrar na pasta (selecionando a branch gpu)
git clone -b gpu https://github.com/matheuscctb/RL-soccer.git
cd RL-soccer

# 2. Criar o ambiente virtual com Python 3.10 (Obrigatório para rc-robosim)
# Usando uv (mais rápido e gerencia o Python automaticamente):
uv venv venv --python 3.10
# Ou usando venv padrão:
# python3.10 -m venv venv

# 3. Ativar o ambiente virtual
# No Linux:
source venv/bin/activate
# No Windows:
# venv\Scripts\activate

# 4. Instalar as dependências com suporte a CUDA
uv pip install -r requirements.txt
# Ou via pip padrão:
# pip install --upgrade pip && pip install -r requirements.txt

# 5. Instalar o rsoccer-gym localmente em modo editável
uv pip install -e .
# Ou via pip padrão:
# pip install -e .
```


---

## 🚀 Como Executar com GPU

### 1. Iniciar o Treinamento Acelerado na GPU (Padrão: CUDA)

Inicia o treinamento acelerado em GPU com suporte a paralelização multi-core na CPU para simulação física, salvando checkpoints em `modelos/checkpoints/` e logs em `runs/`:

```bash
# Execução padrão (16 ambientes paralelos distribuídos nos núcleos de CPU)
python train_attacker.py
```

*Para escalar paralelismo (ex: 64, 256 ou 2048 ambientes):*
```bash
# Alta paralelização (ex: 64 ambientes com rollout de 100 passos)
python train_attacker.py --num-envs 64 --num-steps 100 --batch-size 256

# Escala massiva (ex: 2048 ambientes com rollout mais curto e batch maior para saturar a GPU)
python train_attacker.py --num-envs 2048 --num-steps 20 --batch-size 1024
```

---

### 2. Retomar o Treinamento de Onde Parou (`--resume`)

Se o treinamento for interrompido, você pode continuar exatamente de onde parou:

```bash
# Continua automaticamente a partir do checkpoint mais recente
python train_attacker.py --resume
```

*Ou especificar um arquivo de checkpoint específico:*
```bash
python train_attacker.py --resume-path modelos/checkpoints/mappo_step_1500000.pt
```

---

### 3. Monitoramento em Tempo Real com TensorBoard

Abra um terminal e execute:

```bash
tensorboard --logdir runs
```

Acesse **`http://localhost:6006`** no seu navegador para visualizar:
* **Curvas de Recompensa** (`Performance/Mean_Reward`)
* **Taxa de Conversão de Gols e Passes** (`Performance/Goal_Rate_Pct`, `Pass_Rate_Pct`)
* **Total de Gols e Passes** (`Performance/Goals_Total`, `Passes_Total`)
* **Perdas das Redes** (`Loss/Actor_Loss`, `Loss/Critic_Loss`, `Loss/Entropy`)
* **Decomposição das Recompensas** (`Reward_Shaping/*`)

---

### 4. Acompanhamento ao Vivo (`play_attacker.py`)

Abra um terminal paralelo enquanto o treinamento estiver rodando para **assistir à evolução gráfica dos robôs em tempo real**:

```bash
python play_attacker.py
```

* **Hot-Reloading Automático**: A cada $100.000$ passos que um novo checkpoint for salvo em disco, o script atualiza os pesos da rede na janela do Pygame sem precisar reiniciar!
* O título da janela Pygame e o console mostram o número de passos do checkpoint atual, taxa de gols e total de passes.

---

### 5. Avaliar um Modelo Específico (`play.py`)

Para executar e avaliar um modelo salvo em $N$ episódios completos:

```bash
# Executa o modelo mais recente de modelos/
python play.py

# Ou especifica o caminho de um modelo específico
python play.py --model-path modelos/mappo_best.pt --episodes 20 --fps 40
```

---

## 📁 Estrutura do Repositório

```
RL-soccer/
├── mappo/                                      # Arquitetura Multi-Agent PPO
│   ├── __init__.py                             # Exportação dos módulos
│   ├── models.py                               # ContinuousActor, CentralizedCritic, ActorONNXWrapper
│   ├── buffer.py                               # MultiAgentRolloutBuffer com GAE centralizado
│   └── mappo.py                                # Treinador MAPPO, salvamento .pt/.onnx e resume
├── rsoccer_gym/                                # Ambientes de Futebol de Robôs
│   ├── ssl/
│   │   └── ssl_el_cooperation_attacker.py      # Ambiente SSL-EL (Campo 4.5x3m, Recompensas, Pygame)
│   └── __init__.py                             # Registro de SSL-EL-CooperationAttacker-v0
├── modelos/                                    # Pasta onde os modelos finais são salvos
│   ├── checkpoints/                            # Checkpoints periódicos (mappo_step_X.pt / .onnx)
│   ├── mappo_best.pt / .onnx                   # Melhor modelo histórico por recompensa
│   └── mappo_final.pt / .onnx                  # Modelo consolidado ao fim dos 5M passos
├── runs/                                       # Logs de métricas para o TensorBoard
├── train_attacker.py                           # Script de treinamento vetorizado com PPO + TensorBoard + Resume
├── play_attacker.py                            # Visualizador em tempo real com auto-reload de checkpoints
├── play.py                                     # Avaliador gráfico de modelos salvos
├── README.md                                   # Documentação oficial do projeto (este arquivo)
└── README_ORIGINAL_RSOCCER.md                  # Documentação legada original do rSoccer
```

---

## 📜 Citação e Créditos

Este projeto estende o framework **rSoccer**:
```bibtex
@InProceedings{10.1007/978-3-030-98682-7_14,
  author    = {Martins, Felipe B. and Machado, Mateus G. and Bassani, Hansenclever F. and Braga, Pedro H. M. and Barros, Edna S.},
  title     = {rSoccer: A Framework for Studying Reinforcement Learning in Small and Very Small Size Robot Soccer},
  booktitle = {RoboCup 2021: Robot World Cup XXIV},
  year      = {2022},
  publisher = {Springer International Publishing},
  pages     = {165--176}
}
```
