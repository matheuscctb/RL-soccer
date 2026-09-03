# 🛡️ Documentação da Branch: `feat/parede-virtual-areas`

Este documento consolida todas as modificações, motivações teóricas, correções arquiteturais e resultados de validação implementados na branch **`feat/parede-virtual-areas`** do repositório **RL-Soccer**.

---

## 📌 Sumário Executivo

* **Nome da Branch**: `feat/parede-virtual-areas`
* **Branch Base**: `gpu`
* **Objetivo Principal**: 
  1. Impedir fisicamente e algoritmicamente que os robôs invadam as áreas proibidas (áreas de pênalti) para forçar gols (*Action Shielding* + *Hard Constraint*).
  2. Eliminar o comportamento de paralisia e hesitação ("esperar o parceiro com a bola ao lado") via atribuição dinâmica de papéis com histerese (*Dynamic Role Allocation*).
  3. Equalizar o treinamento com alternância de papéis no spawn inicial (simetria de agentes).
  4. Adicionar gradiente de contenção suave antes das bordas sem punir robôs que salvam a bola viva na linha (*Chase Factor*).
  5. Corrigir o mapeamento de IDs de robôs no simulador Box2D (`rsim.py`).

---

## 🔍 1. Diagnóstico dos Problemas Anteriores

| Problema Identificado | Causa Raiz | Sintoma nos Vídeos / Treino |
| :--- | :--- | :--- |
| **Invasão de Área para Fazer Gol** | O gol concedia $+50.0$ e a invasão penalizava apenas $-3.0$. O algoritmo aprendia que compensava invadir a área proibida para empurrar a bola para o gol (*reward hacking*). | Robôs atropelavam a área adversária para empurrar a bola. |
| **Robô "Espera" com a Bola ao Lado** | Papel de atacante/ala era recalculado sem histerese e baseado apenas na distância euclidiana instantânea. Se um robô virava ala, seu alvo de PBRS era puxado para longe da bola, punindo-o se tentasse pegá-la. | Robô congelava ou hesitava ao lado de uma bola viva. |
| **Assimetria e Vício de Identidade** | No método de reset, `blue_0` nascia sempre atrás da bola e `blue_1` nascia sempre no flanco oposto. | `blue_0` virava um atacante "fominha" e `blue_1` um espectador passivo sem iniciativa. |
| **Término Repentino por Saída de Campo** | Não havia gradiente de aviso antes da linha. Além disso, penalizar bordas sem critério punia o robô justo quando ele corria para salvar a bola de sair. | Robôs saíam de campo abruptamente ou desistiam de disputar a bola perto da linha. |
| **Bug de Reset no Box2D (`rsim.py`)** | `_placement_dict_from_frame` iterava com `.values()`, que depende da ordem de inserção do dicionário Python, desordenando os IDs no Box2D. | Robôs trocavam de posição indevidamente ao nascer no simulador. |

---

## 🛠️ 2. Soluções Implementadas em Detalhes

### 2.1. Parede Virtual Cinemática Rígida (*Action Shielding*)
Implementada no método `_apply_virtual_walls(robot_x, robot_y, v_x, v_y)` em `rsoccer_gym/ssl/ssl_el_cooperation_attacker.py`:
* **Área de Pênalti Adversária (Amarela)**:
  * Início da grande área oficial: $X = 1.75\text{ m}$, $|Y| \le 0.675\text{ m}$.
  * Linha de parada segura com buffer: $X_{stop} = 1.63\text{ m}$ (considerando raio do robô de $0.09\text{ m}$ + margem de segurança física).
  * **Zona de Frenagem Progressiva**: Começa 50 cm antes da linha ($d_{brake} = 0.50\text{ m}$), reduzindo a velocidade máxima permitida de acordo com $v_{max} \cdot \left(\frac{\text{dist}}{d_{brake}}\right)^{1.5}$.
  * **Bloqueio e Força de Restituição**: Se o robô ultrapassar $1.63\text{ m}$, comandos em $+X$ são anulados e uma velocidade de restituição proporcional é aplicada em $-X$.
  * **Paredes Laterais (Flancos)**: Bloqueia qualquer tentativa de invadir a área cortando por cima ($Y > 0.785\text{ m}$) ou por baixo ($Y < -0.785\text{ m}$).
  * **Deslizamento Tangencial Livre**: O robô pode se mover livremente em $Y$, comportando-se exatamente como uma parede física real.
* **Área Própria (Azul) e Borda Perimetral**:
  * Mesma barreira física na grande área azul ($X \le -1.63\text{ m}$) e nas quatro linhas laterais do campo.

### 2.2. Restrição Rígida no MDP (*Hard Constraint*)
No método `_calculate_multi_agent_reward_and_done`:
* **Prioridade Absoluta nº 1**: A validação de infrações de área e limites de campo foi movida para o início do cálculo, **antes** da checagem de gols.
* **Anulação de Gol Imediata**: Se qualquer atacante azul invadir a área de pênalti, o gol é **anulado na hora** (recompensa de gol $= 0.0$).
* **Penalidade Rígida**: Aplica $-30.0$ compartilhado e encerra o episódio no mesmo instante (`done = True`).

### 2.3. Atribuição Dinâmica de Papéis por Função de Custo com Histerese
Implementados `_role_cost` e `_resolve_lead_role`:
* **Função de Custo Cinemática**:
  $$C_i = w_1 \cdot r_i + w_2 \cdot (1 - \cos\theta_i)$$
  * $r_i$: distância euclidiana até a bola.
  * $\theta_i$: desalinhamento angular entre a orientação do robô e o vetor até a bola ($\theta = 0 \implies 1 - \cos\theta = 0$; de costas $\implies 1 - \cos\theta = 2$).
  * Pesos: $w_1 = 1.0$ (distância), $w_2 = 0.4$ (alinhamento).
* **Histerese Multiplicativa (`ROLE_HYSTERESIS_DISCOUNT = 0.85`)**:
  * O custo do robô que já é líder recebe um desconto de $15\%$.
  * O parceiro só assume o papel de condutor se seu custo for claramente menor, eliminando o *ping-pong* de alvos.
* **Consistência no PBRS**:
  * O papel resolvido é mantido idêntico entre a avaliação do estado atual e do `last_frame`, evitando descontinuidades no delta de potencial $\Phi(s') - \Phi(s)$.

### 2.4. Alternância de Papéis no Spawn Inicial
No método `_get_initial_positions_frame`:
* Sorteio estocástico equilibrado:
  ```python
  conductor_id = random.choice([0, 1])
  wing_id = 1 - conductor_id
  ```
* Em média, $50\%$ dos episódios nascem com `blue_0` como condutor e $50\%$ com `blue_1`, forçando os agentes a aprenderem tanto a conduzir quanto a desmarcar/infiltrar.

### 2.5. Barreiras Suaves com Supressão Inteligente (*Chase Factor*)
Na Seção 2b da função de recompensa:
* **Margens de Aviso**: Zona de $15\text{ cm}$ antes do limite rígido do campo e das áreas de pênalti.
* **Supressão por Perseguição da Bola**:
  ```python
  dist_r_b = float(np.linalg.norm([rbt.x - ball.x, rbt.y - ball.y]))
  chase_factor = float(np.clip(dist_r_b / self.BOUNDARY_CHASE_SUPPRESS_RADIUS, 0.0, 1.0))
  boundary_pen = -0.6 * (overshoot / self.BOUNDARY_WARN_MARGIN) * chase_factor
  ```
  * Se o robô estiver perto da bola ($< 0.50\text{ m}$) salvando um lance na linha, a penalidade cai para $\approx 0$.
  * Se estiver vagando sem motivo perto da linha, a penalidade age com força total.
  * Nas áreas de pênalti, a penalidade suave atua **sempre**, pois invasão é estritamente proibida pelas regras da SSL.

### 2.6. Correção de Indexação no Box2D (`rsim.py`)
No arquivo `rsoccer_gym/Simulators/rsim.py`:
* Substituída a iteração sobre `.values()` por indexação ordenada por chave numérica (`frame.robots_blue[i]`).
* Garante correspondência 1:1 rigorosa entre as entidades do Python e os corpos rígidos do simulador Box2D em C++.

---

## 📂 3. Arquivos Modificados

| Arquivo | Modificações Principais |
| :--- | :--- |
| `rsoccer_gym/ssl/ssl_el_cooperation_attacker.py` | • Implementação de `_apply_virtual_walls`<br>• Integração em `convert_actions` e comandos<br>• Inversão de prioridade de checagem no MDP (Áreas antes de Gols)<br>• Métodos `_role_cost` e `_resolve_lead_role`<br>• Seção 2b com barreiras suaves e `chase_factor`<br>• Spawn aleatório de papéis |
| `rsoccer_gym/Simulators/rsim.py` | • Correção na ordenação de IDs dos robôs (`blue_robots_pos` e `yellow_robots_pos`) |

---

## 🧪 4. Testes de Validação Realizados

### Teste 1: Estresse da Parede Virtual (50 Episódios / 6.000 Passos)
* **Condição**: Robôs acelerando para a frente com comandos em velocidade máxima ($1.5\text{ m/s}$) contra as áreas de pênalti.
* **Resultado**:
  * Total de invasões de área: **`0`**
  * Desaceleração suave observada antes da linha e deslizamento lateral estável.

### Teste 2: Anulação de Gol em Caso de Invasão
* **Condição**: Bola posicionada dentro da baliza com robô posicionado deliberadamente dentro da grande área.
* **Resultado**:
  * Recompensa de gol: **`0.0`** (anulado)
  * Penalidade de violação: **`-30.0`**
  * Término imediato: `done = True`

### Teste 3: Alternância no Spawn Inicial (20 Resets)
* **Resultado**:
  * `blue_0` como Condutor: **11 vezes**
  * `blue_1` como Condutor: **9 vezes**
  * Distribuição perfeitamente equilibrada e simétrica.

### Teste 4: Supressão do `chase_factor` na Borda
* **Condição**: Robô na linha lateral ($Y = 1.45\text{ m}$, limite $1.50\text{ m}$).
* **Resultado**:
  * Vagando perto da borda (bola longe): Penalidade de **`-0.3989`**
  * Salvando a bola viva perto da borda: Penalidade de **`-0.0164`** (redução de **96%**).

---

## 🚀 5. Como Executar e Treinar nesta Branch

Para reproduzir ou continuar os testes nesta branch:

```bash
# 1. Garantir que está na branch correta
git checkout feat/parede-virtual-areas

# 2. Executar o script de treino com MAPPO
python3 train_attacker.py

# 3. Ou retomar a partir de um checkpoint existente
python3 train_attacker.py --resume

# 4. Assistir ao comportamento visualmente
python3 play_attacker.py
```
