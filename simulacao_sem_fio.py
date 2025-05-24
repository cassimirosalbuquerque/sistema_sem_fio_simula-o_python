import numpy as np
import matplotlib.pyplot as plt

# PARÂMETROS FIXOS DO SISTEMA
coverage_area_side = 1000  # 1 km x 1 km
total_bandwidth_hz = 100e6  # 100 MHz
ue_tx_power_w = 1
propagation_n = 4
propagation_d0 = 1
propagation_k = 10**-4
noise_k0_mw_hz = 10**-17  # milliWatts/Hz
noise_k0_w_hz = noise_k0_mw_hz * 1e-3 # Watts/Hz
qos_target_cell_border_mbps = 100  # Mbps

# PARÂMETROS 
num_aps_m_values = [1, 9, 36, 64]
num_orthogonal_channels_n_values = [1, 2, 3]
num_monte_carlo_runs = 10000

# Parâmetros Exercício 11
M_exercise11 = 36
N_exercise11 = 1
K_exercise11 = 5
low_sinr_threshold = 1e-4 # limite para considerar um snapshot de baixo SINR


# FUNÇÕES 

def position_aps(M, area_side):
    if np.sqrt(M) != int(np.sqrt(M)):
        print(f"M={M} não é um quadrado perfeito. Espaçamento é feito aproximado.")
    ap_positions = []
    grid_size = int(np.sqrt(M))
    step = area_side / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            ap_x = i * step + step / 2
            ap_y = j * step + step / 2
            ap_positions.append((ap_x, ap_y))
    return ap_positions

def position_ues(K, area_side):
    ue_positions = np.random.uniform(0, area_side, (K, 2))
    return ue_positions.tolist()

def calculate_distance(pos1, pos2):
    pos1_np = np.array(pos1)
    pos2_np = np.array(pos2)
    return np.sqrt(np.sum((pos1_np - pos2_np)**2))

def calculate_path_loss(distance, k, n, d0):
    effective_distance = max(distance, d0)
    return k / (effective_distance**n)

def calculate_received_power(tx_power, path_gain):
    return tx_power * path_gain

def assign_ue_to_ap(ue_position, ap_positions, ue_tx_power, k, n, d0):
    max_received_power = -np.inf
    serving_ap_index = -1
    for i, ap_pos in enumerate(ap_positions):
        distance = calculate_distance(ue_position, ap_pos)
        path_gain = calculate_path_loss(distance, k, n, d0)
        received_power = ue_tx_power * path_gain
        if received_power > max_received_power:
            max_received_power = received_power
            serving_ap_index = i
    return serving_ap_index, max_received_power

def calculate_noise_power(total_bandwidth, num_channels, k0_w_hz):
    channel_bandwidth = total_bandwidth / num_channels
    return k0_w_hz * channel_bandwidth

def calculate_sinr(desired_received_power, total_interference_power, noise_power):
    total_interference_plus_noise = total_interference_power + noise_power
    if total_interference_plus_noise <= 0:
        return np.inf
    return desired_received_power / total_interference_plus_noise

def calculate_channel_capacity(bandwidth_channel, sinr):
    if sinr <= 0:
        return 0.0
    return bandwidth_channel * np.log2(1 + sinr)

# --- FUNÇÃO AUXILIAR PARA CALCULAR CDF ---
def calculate_cdf(data):
    """Calculates the x and y points for a CDF."""
    sorted_data = np.sort(data)
    y = np.arange(len(sorted_data)) / len(sorted_data)
    return sorted_data, y


# SELECIONA O EXERCÍCIO
selected_exercise = None
while selected_exercise not in [9, 10, 11]:
    try:
        selected_exercise = int(input("Qual exercício você quer simular? Digite 9, 10 ou 11: "))
        if selected_exercise not in [9, 10, 11]:
            print("Entrada inválida. Por favor, digite 9, 10 ou 11.")
    except ValueError:
        print("Entrada inválida. Por favor, digite um número.")

# DEFINE OS PARÂMETROS COM BASE NO EXERCÍCIO ESCOLHIDO
if selected_exercise == 9:
    num_ues_k = 1
    exercise_name = "Exercício 9 (Análise de Cobertura)"
    sim_m_values = num_aps_m_values
    sim_n_values = num_orthogonal_channels_n_values
    run_until_low_sinr = False

elif selected_exercise == 10:
    num_ues_k = 13
    exercise_name = "Exercício 10 (Análise de Capacidade)"
    sim_m_values = num_aps_m_values
    sim_n_values = num_orthogonal_channels_n_values
    run_until_low_sinr = False

elif selected_exercise == 11:
    num_ues_k = K_exercise11
    exercise_name = "Exercício 11 (Cenário Extremo de Baixo SINR)"
    sim_m_values = [M_exercise11]
    sim_n_values = [N_exercise11]
    run_until_low_sinr = True


print(f"\nPreparando simulação para: {exercise_name} com K={num_ues_k} UEs.")


# SIMULAÇÃO DE MONTE CARLO

simulation_results = {}
low_sinr_snapshot_data = None
low_sinr_found = False

for m in sim_m_values:
    ap_positions = position_aps(m, coverage_area_side)
    ap_positions_np = np.array(ap_positions)

    for n in sim_n_values:
        print(f"Simulando M={m}, N={n}, K={num_ues_k}, N_canais={n}...")

        sinr_values_config = []
        capacity_values_config = []

        noise_power_w = calculate_noise_power(total_bandwidth_hz, n, noise_k0_w_hz)

        for run in range(num_monte_carlo_runs):
            ue_positions_run = position_ues(num_ues_k, coverage_area_side)
            ue_positions_run_np = np.array(ue_positions_run)

            # Atribuição AP-UE e Canal
            ue_assignments = []
            for ue_index, ue_pos in enumerate(ue_positions_run):
                serving_ap_index, _ = assign_ue_to_ap(
                     ue_pos, ap_positions, ue_tx_power_w, propagation_k, propagation_n, propagation_d0
                )
                assigned_channel = np.random.randint(0, n) if n > 1 else 0
                ue_assignments.append((ue_index, serving_ap_index, assigned_channel))

            # Calcular SINR e Capacidade
            sinr_values_run = []
            capacity_values_run = []

            for ue_index, serving_ap_index, assigned_channel in ue_assignments:
                ue_pos = ue_positions_run[ue_index]
                serving_ap_pos = ap_positions[serving_ap_index]

                desired_received_power = calculate_received_power(
                    ue_tx_power_w,
                    calculate_path_loss(
                        calculate_distance(ue_pos, serving_ap_pos),
                        propagation_k, propagation_n, propagation_d0
                    )
                )

                total_interference_power = 0
                for other_ue_index, other_serving_ap_index, other_assigned_channel in ue_assignments:
                    if ue_index != other_ue_index and assigned_channel == other_assigned_channel:
                        other_ue_pos = ue_positions_run[other_ue_index]
                        interference_received_power = calculate_received_power(
                            ue_tx_power_w,
                            calculate_path_loss(
                                calculate_distance(other_ue_pos, serving_ap_pos),
                                propagation_k, propagation_n, propagation_d0
                            )
                        )
                        total_interference_power += interference_received_power

                sinr = calculate_sinr(desired_received_power, total_interference_power, noise_power_w)
                channel_bandwidth_hz = total_bandwidth_hz / n
                capacity_bps = calculate_channel_capacity(channel_bandwidth_hz, sinr)

                sinr_values_run.append(sinr)
                capacity_values_run.append(capacity_bps)

            # MODIFICAÇÃO (EX11): Verificação e Condição de Parada + Captura de Dados Detalhados do Snapshot 
            if run_until_low_sinr:
                 # Verifica se algum UE nesta run tem SINR abaixo do threshold
                 if any(sinr < low_sinr_threshold for sinr in sinr_values_run):
                     print(f"\nBaixo SINR encontrado no snapshot {run+1}.")

                     # Captura de Dados Detalhados do Snapshot 
                     # Encontra o pior valor de SINR e o UE correspondente NESTE SNAPSHOT
                     worst_sinr_in_snapshot = np.min(sinr_values_run) # Encontra o valor mínimo
                     worst_sinr_ue_index = np.argmin(sinr_values_run) # Encontra o índice do mínimo
                     worst_sinr_ue_pos = ue_positions_run[worst_sinr_ue_index] # Pega a posição do pior UE

                     # Armazena todos os dados relevantes DESTE snapshot, incluindo o pior caso
                     low_sinr_snapshot_data = {
                         'ap_positions': ap_positions,
                         'ue_positions': ue_positions_run,
                         'ue_assignments': ue_assignments,
                         'sinr_values': sinr_values_run,
                         # Armazena informações do pior caso
                         'worst_sinr': worst_sinr_in_snapshot,
                         'worst_sinr_ue_pos': worst_sinr_ue_pos,
                         'snapshot_run_number': run + 1 # Armazena o número da run onde foi encontrado
                     }
                     low_sinr_found = True
                     break # Sai do Monte Carlo

            # Para Ex. 9 e 10, ou se não parou no Ex. 11: acumula resultados estatísticos
            if not run_until_low_sinr or not low_sinr_found:
                 sinr_values_config.extend(sinr_values_run)
                 capacity_values_config.extend(capacity_values_run)

        # Após todas as runs (ou ter parado cedo no Ex. 11):
        if not run_until_low_sinr:
             simulation_results[(m, n)] = {'sinr': sinr_values_config, 'capacity': capacity_values_config}
             print(f"Terminadas {num_monte_carlo_runs} runs para M={m}, N={n}. Coletados {len(sinr_values_config)} pontos de dados (de {num_ues_k} UEs por run).")
        elif low_sinr_found:
             print(f"Simulação para M={m}, N={n} parada após encontrar baixo SINR.")
             # Imprime as informações do pior caso logo após parar
             print(f"  Pior valor de SINR no snapshot: {low_sinr_snapshot_data['worst_sinr']:.4e}") # Usar notação científica
             print(f"  Coordenadas do UE com o pior SINR: ({low_sinr_snapshot_data['worst_sinr_ue_pos'][0]:.2f}, {low_sinr_snapshot_data['worst_sinr_ue_pos'][1]:.2f})")
        else:
             print(f"Simulação para M={m}, N={n} completou {num_monte_carlo_runs} runs sem encontrar baixo SINR abaixo do threshold {low_sinr_threshold:.1e}.") # Mostrar threshold


# ANÁLISE E PLOTAGEM (Condicional ao Exercício)

# Análise Estatística (para Ex. 9 e 10)
if selected_exercise in [9, 10]:
    print(f"\n--- Análise Estatística para {exercise_name} ---")

    for (m, n), data in simulation_results.items():
        sinr_values = data['sinr']
        capacity_values_bps = data['capacity']
        capacity_values_mbps = [c / 1e6 for c in capacity_values_bps]

        percentile_10_sinr = np.percentile(sinr_values, 10)
        percentile_10_capacity_mbps = np.percentile(capacity_values_mbps, 10)

        print(f"\nConfiguração: M={m}, N={n}")
        print(f"  10º Percentil SINR: {percentile_10_sinr:.2f}")
        print(f"  10º Percentil Capacidade de Canal: {percentile_10_capacity_mbps:.2f} Mbps")

        if percentile_10_capacity_mbps >= qos_target_cell_border_mbps:
            print("  Objetivo de QoS (100 Mbps no 10º percentil) ALCANÇADO.")
        else:
            print("  Objetivo de QoS (100 Mbps no 10º percentil) NÃO ALCANÇADO.")

    # CÓDIGO DE PLOTAGEM para Ex. 9 e 10

    print(f"\n--- Gerando Gráficos de CDF para {exercise_name} ---")

    # Plotagem das CDFs do SINR (Linear)
    fig_snr_cdf_linear, axes_snr_cdf_linear = plt.subplots(1, len(sim_m_values), figsize=(5 * len(sim_m_values), 5), squeeze=False)
    fig_snr_cdf_linear.suptitle(f'Curvas da CDF do SINR para {exercise_name}, variando M e N (Escala Linear)')
    for col_idx, m in enumerate(sim_m_values):
        ax = axes_snr_cdf_linear[0, col_idx]
        ax.set_title(f'Pontos de Acesso: {m}')
        ax.set_xlabel('SINR')
        ax.set_ylabel('CDF')
        ax.grid(True, linestyle='--', alpha=0.6)
        for n in sim_n_values:
            if (m, n) in simulation_results:
                sinr_values = simulation_results[(m, n)]['sinr']
                sorted_sinr, sinr_cdf = calculate_cdf(sinr_values)
                ax.plot(sorted_sinr, sinr_cdf, label=f'N: {n}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plotagem das CDFs do SINR (Logarítmica)
    fig_snr_cdf_log, axes_snr_cdf_log = plt.subplots(1, len(sim_m_values), figsize=(5 * len(sim_m_values), 5), squeeze=False)
    fig_snr_cdf_log.suptitle(f'Curvas da CDF do SINR para {exercise_name}, variando M e N (Escala Logarítmica)')
    for col_idx, m in enumerate(sim_m_values):
        ax = axes_snr_cdf_log[0, col_idx]
        ax.set_title(f'Pontos de Acesso: {m}')
        ax.set_xlabel('SINR')
        ax.set_ylabel('CDF')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xscale('log')
        for n in sim_n_values:
            if (m, n) in simulation_results:
                sinr_values = simulation_results[(m, n)]['sinr']
                positive_finite_sinr_values = [s for s in sinr_values if s > 0 and np.isfinite(s)]
                if positive_finite_sinr_values:
                    sorted_sinr, sinr_cdf = calculate_cdf(positive_finite_sinr_values)
                    ax.plot(sorted_sinr, sinr_cdf, label=f'N: {n}')
                else:
                     print(f"Aviso: Não há valores de SINR positivos finitos para plotar para M={m}, N={n} na escala logarítmica.")
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plotagem das CDFs da Capacidade
    fig_capacity_cdf, axes_capacity_cdf = plt.subplots(1, len(sim_n_values), figsize=(5 * len(sim_n_values), 5), squeeze=False)
    fig_capacity_cdf.suptitle(f'Curvas da CDF da Capacidade de canal para {exercise_name}, variando M e N')
    for col_idx, n in enumerate(sim_n_values):
        ax = axes_capacity_cdf[0, col_idx]
        ax.set_title(f'Canal: {n}')
        ax.set_xlabel('Capacidade de Canal (Mbps)')
        ax.set_ylabel('CDF')
        ax.grid(True, linestyle='--', alpha=0.6)
        for m in sim_m_values:
             if (m, n) in simulation_results:
                capacity_values_bps = simulation_results[(m, n)]['capacity']
                capacity_values_mbps = [c / 1e6 for c in capacity_values_bps]
                sorted_capacity_mbps, capacity_cdf = calculate_cdf(capacity_values_mbps)
                ax.plot(sorted_capacity_mbps, capacity_cdf, label=f'M: {m}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Visualização do Snapshot de Baixo SINR (apenas para Ex. 11)
elif selected_exercise == 11:
    print(f"\n--- Visualização do Snapshot para {exercise_name} ---")

    if low_sinr_found and low_sinr_snapshot_data is not None:
        ap_positions_snap = low_sinr_snapshot_data['ap_positions']
        ue_positions_snap = low_sinr_snapshot_data['ue_positions']
        ue_assignments_snap = low_sinr_snapshot_data['ue_assignments']
        sinr_values_snap = low_sinr_snapshot_data['sinr_values']
        # --- Mudança: Recupera os dados do pior caso ---
        worst_sinr_snap = low_sinr_snapshot_data['worst_sinr']
        worst_sinr_ue_pos_snap = low_sinr_snapshot_data['worst_sinr_ue_pos']
        snapshot_run_number_snap = low_sinr_snapshot_data['snapshot_run_number']


        plt.figure(figsize=(8, 8))

        # Plot APs
        ap_positions_np_snap = np.array(ap_positions_snap)
        plt.scatter(ap_positions_np_snap[:, 0], ap_positions_np_snap[:, 1], marker='^', facecolors='none', label='APs', s=100, edgecolors='blue')

        # Plot UEs. Podemos adicionar um destaque para o UE com o pior SINR.
        ue_positions_np_snap = np.array(ue_positions_snap)
        # Plot todos os UEs
        plt.scatter(ue_positions_np_snap[:, 0], ue_positions_np_snap[:, 1], marker='s', facecolors='none', label='UEs', s=50, edgecolors='red', zorder=5)
        # --- Mudança: Destacar o UE com o pior SINR ---
        plt.scatter(worst_sinr_ue_pos_snap[0], worst_sinr_ue_pos_snap[1], marker='*', color='gold', label=f'Pior SINR UE ({worst_sinr_snap:.2e})', s=200, edgecolors='black', zorder=6) # Marcador de estrela dourada


        # Desenha linhas conectando cada UE ao seu AP servidor
        for ue_index, serving_ap_index, assigned_channel in ue_assignments_snap:
             ue_pos = ue_positions_snap[ue_index]
             serving_ap_pos = ap_positions_snap[serving_ap_index]
             plt.plot([ue_pos[0], serving_ap_pos[0]], [ue_pos[1], serving_ap_pos[1]], color='gray', linestyle='-', linewidth=1, alpha=0.7)

        # Adiciona linhas de grade
        grid_size_snap = int(np.sqrt(M_exercise11))
        step_snap = coverage_area_side / grid_size_snap
        for i in range(grid_size_snap + 1):
             plt.axvline(i * step_snap, color='gray', linestyle='--', linewidth=0.5)
             plt.axhline(i * step_snap, color='gray', linestyle='--', linewidth=0.5)

        plt.xlim([0, coverage_area_side])
        plt.ylim([0, coverage_area_side])
        plt.xlabel('Posição X (m)')
        plt.ylabel('Posição Y (m)')
        # Mudança: Título do gráfico inclui o número da run onde o snapshot foi encontrado
        plt.title(f'Snapshot de Baixo SINR (Run {snapshot_run_number_snap}) para {exercise_name}\n(M={M_exercise11}, N={N_exercise11}, K={K_exercise11})')
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()


    else:
        print("\nNão foi possível encontrar um snapshot com SINR abaixo do threshold nas runs simuladas.")
        print(f"Considere aumentar 'num_monte_carlo_runs' ou ajustar 'low_sinr_threshold' ({low_sinr_threshold:.1e}).")


# Exibe todos os gráficos gerados (para Ex. 9/10) ou a figura do snapshot (para Ex. 11)
plt.show()

print(f"\nProcesso para {exercise_name} concluído.")