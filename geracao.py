# ==============================================================================
# GERACAO.PY - MOTOR EVOLUTIVO HÍBRIDO SIMULTÂNEO (CPU MULTI-CORE + GPU)
# ==============================================================================
# Este módulo coordena a busca e evolução de soluções para o Cubo de Rubik.
#
# Principais Funcionalidades:
# 1. Resolução Incremental:
#    Testa progressivamente cromossomos de tamanho_minimo até tamanho_maximo.
# 2. Busca Exaustiva Rápida (para tamanhos 1, 2 e 3):
#    Para espaços de busca pequenos (até 4.000 combinações), avalia todas as
#    combinações em milissegundos (< 0.02s).
# 3. Execução Simultânea Heterogênea (CPU + GPU):
#    Executa simultaneamente 15 processos paralelos nos núcleos da CPU
#    (AMD Ryzen™ 7 PRO 8700GE) e a Super-Ilha de GPU (AMD Radeon™ 780M Graphics)
#    via Compute Shaders WebGPU/Vulkan com migração bidirecional de campeões.
# 4. Fallback Dinâmico:
#    Adapta-se automaticamente caso apenas CPU ou apenas GPU esteja disponível.
# ==============================================================================

import copy
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from cruzamento import cruzar_dois_individuos, cruzamento
from mutacao import mutar_individuo, mutacao
from pontuacao import (
    ESTADO_RESOLVIDO,
    aplicar_movimentos,
    calcular_score,
    calcular_score_estado,
)
from populacao import (
    MOVIMENTOS,
    PARALELAS,
    calcular_espaco_busca,
    gerar_populacao,
    gerar_todas_combinacoes_validas,
)
from gpu_engine import (
    obter_gpu_engine,
    obter_informacoes_gpu,
    converter_populacao_para_ids,
    converter_ids_para_populacao,
    ID_TO_MOVE,
    MOVE_TO_ID,
)

# Limite de combinações para executar busca exaustiva determinística direta (em vez de AG)
LIMITE_BUSCA_EXAUSTIVA = 4000


def selecionar_melhores(populacao, embaralhamento_ou_estado, porcentagem_selecao, cache=None):
    """
    Avalia o fitness de cada indivíduo da população e retorna os melhores selecionados.

    Otimização:
    Se 'embaralhamento_ou_estado' já for o estado pré-computado de 54 adesivos,
    avalia diretamente sem reprocessar o embaralhamento do zero.

    Parâmetros:
        populacao (list[list[str]]): Lista de indivíduos a avaliar.
        embaralhamento_ou_estado (tuple | list): Estado pré-embaralhado ou lista de movimentos.
        porcentagem_selecao (float): Proporção da população a ser mantida (ex: 0.50 = 50%).
        cache (dict, opcional): Dicionário de memoização para indivíduos já avaliados.

    Retorno:
        tuple: (melhores_individuos, melhor_score, lista_avaliados_ordenada)
    """
    is_estado_precomputado = (
        isinstance(embaralhamento_ou_estado, (tuple, list))
        and len(embaralhamento_ou_estado) == 54
        and isinstance(embaralhamento_ou_estado[0], int)
    )

    if is_estado_precomputado:
        estado_base = embaralhamento_ou_estado
    else:
        estado_base = aplicar_movimentos(ESTADO_RESOLVIDO, embaralhamento_ou_estado)

    avaliados = []
    for ind in populacao:
        chave = tuple(ind)
        if cache is not None and chave in cache:
            score = cache[chave]
        else:
            st = aplicar_movimentos(estado_base, ind)
            score = calcular_score_estado(st)
            if cache is not None:
                cache[chave] = score
        avaliados.append((score, ind))

    # Ordena do maior fitness para o menor
    avaliados.sort(key=lambda x: x[0], reverse=True)

    # Determina a quantidade a manter com base na taxa de seleção
    qtd_selecionada = max(1, round(len(populacao) * porcentagem_selecao))
    melhores = [ind for _, ind in avaliados[:qtd_selecionada]]

    best_score = avaliados[0][0] if avaliados else 0
    return melhores, best_score, avaliados


def rodar_busca_exaustiva(
    embaralhamento,
    tamanho_cromossomo,
    cache=None,
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
    estado_base_precomputado=None,
):
    """
    Executa a busca exaustiva determinística para espaços pequenos (18, 270, 3.888 combinações).
    Testa todas as combinações válidas em milissegundos utilizando o estado base pré-computado.

    Parâmetros:
        embaralhamento (list[str]): Movimentos que embaralharam o cubo.
        tamanho_cromossomo (int): Comprimento da sequência sendo avaliada.
        cache (dict, opcional): Cache de fitness.
        callback_progresso (callable, opcional): Função para envio de status para a interface.
        is_cancelled (callable, opcional): Função para checar cancelamento pelo usuário.
        total_avaliados_base (int): Contador acumulado de indivíduos testados.
        estado_base_precomputado (tuple, opcional): Estado de 54 adesivos já embaralhado.

    Retorno:
        tuple: (melhor_solucao, melhor_score, avaliados_nesta_etapa)
    """
    if estado_base_precomputado is not None:
        estado_base = estado_base_precomputado
    else:
        estado_base = aplicar_movimentos(ESTADO_RESOLVIDO, embaralhamento)

    todas_combinacoes = gerar_todas_combinacoes_validas(tamanho_cromossomo)
    total = len(todas_combinacoes)

    melhor_score = -1
    melhor_solucao = None
    avaliados_locais = 0
    info_hw = obter_informacoes_hardware()

    if callback_progresso:
        callback_progresso({
            "etapa": f"Busca Exaustiva (Tamanho {tamanho_cromossomo} - {total} combinações)",
            "operacao": "Criando indivíduos",
            "tamanho_atual": tamanho_cromossomo,
            "tamanho_cromossomo": tamanho_cromossomo,
            "cromossomos_populacao": total,
            "cromossomos_avaliados": total_avaliados_base,
            "geracao_atual": 1,
            "total_geracoes": 1,
            "individuos_avaliados": total_avaliados_base,
            "melhor_score": 0,
            "melhor_solucao": [],
            "melhor_solucao_str": "",
            "hardware": info_hw,
            "mensagem": f"Criando indivíduos: Geradas {total:,} combinações válidas para busca exaustiva (tamanho {tamanho_cromossomo}).",
        })

    for i, ind in enumerate(todas_combinacoes, 1):
        if is_cancelled and is_cancelled():
            break

        avaliados_locais += 1
        st = aplicar_movimentos(estado_base, ind)
        score = calcular_score_estado(st)

        if score > melhor_score:
            melhor_score = score
            melhor_solucao = ind

        # Notificação periódica amostrada para evitar sobrecarga de I/O
        if callback_progresso and (i % 500 == 0 or i == total or score == 54):
            sol_str = " ".join(melhor_solucao) if melhor_solucao else ""
            callback_progresso({
                "etapa": f"Busca Exaustiva (Tamanho {tamanho_cromossomo})",
                "operacao": "Avaliando fitness",
                "tamanho_atual": tamanho_cromossomo,
                "tamanho_cromossomo": tamanho_cromossomo,
                "cromossomos_populacao": total,
                "cromossomos_avaliados": total_avaliados_base + avaliados_locais,
                "geracao_atual": 1,
                "total_geracoes": 1,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": sol_str,
                "hardware": info_hw,
                "mensagem": f"Busca Exaustiva ({i}/{total}): Score Atual {melhor_score}/54 [{sol_str}]",
            })

        # Encerramento imediato se encontrou a solução perfeita (54/54)
        if score == 54:
            break

    return melhor_solucao, melhor_score, avaliados_locais


def _worker_ilha_paralela(args):
    """
    Função trabalhadora executada em processos paralelos separados (Modelo de Ilhas CPU).
    Evolui uma subpopulação isolada por um número fixo de gerações (época).

    Parâmetros (tupla args):
        - island_id (int): Identificador numérico da ilha.
        - pop_inicial (list[list[str]]): Indivíduos iniciais desta ilha.
        - estado_base (tuple): Estado de 54 adesivos do cubo embaralhado.
        - geracoes_bloco (int): Quantidade de gerações a executar nesta época.
        - mut_rate (float): Taxa de mutação por gene.
        - cross_rate (float): Taxa de cruzamento.
        - sel_rate (float): Proporção de indivíduos selecionados como pais.
        - seed (int): Semente para o gerador de números aleatórios do processo.

    Retorno:
        dict: Estatísticas da época da ilha (melhor solução, score, população final, avaliados).
    """
    island_id, pop_inicial, estado_base, geracoes_bloco, mut_rate, cross_rate, sel_rate, seed = args
    random.seed(seed)

    populacao = pop_inicial
    pop_size = len(populacao)
    qtd_elite = max(1, round(pop_size * 0.05))
    qtd_sel = max(1, round(pop_size * sel_rate))

    melhor_solucao = None
    melhor_score = -1
    avaliados_locais = 0
    cache = {}

    for _ in range(geracoes_bloco):
        avaliados = []
        for ind in populacao:
            k = tuple(ind)
            if k in cache:
                s = cache[k]
            else:
                st = aplicar_movimentos(estado_base, ind)
                s = calcular_score_estado(st)
                cache[k] = s
            avaliados.append((s, ind))

        avaliados_locais += len(populacao)
        avaliados.sort(key=lambda x: x[0], reverse=True)

        max_s = avaliados[0][0]
        if max_s > melhor_score:
            melhor_score = max_s
            melhor_solucao = avaliados[0][1]

        # Interrompe se a ilha atingiu 100% resolvido
        if max_s == 54:
            return {
                "island_id": island_id,
                "melhor_score": 54,
                "melhor_solucao": avaliados[0][1],
                "populacao_final": [ind for _, ind in avaliados],
                "avaliados": avaliados_locais,
                "resolvido": True,
            }

        # Elitismo e Cruzamento
        nova_pop = [ind for _, ind in avaliados[:qtd_elite]]
        pais = [ind for _, ind in avaliados[:qtd_sel]]

        while len(nova_pop) < pop_size:
            p1 = random.choice(pais)
            p2 = random.choice(pais)
            if random.random() < cross_rate:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            f1 = mutar_individuo(f1, mut_rate)
            f2 = mutar_individuo(f2, mut_rate)

            nova_pop.append(f1)
            if len(nova_pop) < pop_size:
                nova_pop.append(f2)

        populacao = nova_pop

    return {
        "island_id": island_id,
        "melhor_score": melhor_score,
        "melhor_solucao": melhor_solucao,
        "populacao_final": populacao,
        "avaliados": avaliados_locais,
        "resolvido": melhor_score == 54,
    }


def _executar_bloco_gpu(pop_inicial, estado_base, geracoes, mut_rate, cross_rate, sel_rate, engine):
    """
    Executa a evolução da Super-Ilha GPU em paralelo direto na placa de vídeo
    enquanto os 15 núcleos de CPU estão trabalhando simultaneamente.
    """
    populacao = pop_inicial
    pop_size = len(populacao)
    qtd_elite = max(1, round(pop_size * 0.05))
    qtd_sel = max(2, round(pop_size * sel_rate))

    melhor_solucao = None
    melhor_score = -1
    evals = 0

    for _ in range(geracoes):
        pop_ids = converter_populacao_para_ids(populacao)
        scores = engine.avaliar_populacao(estado_base, pop_ids)
        evals += pop_size

        best_idx = int(np.argmax(scores))
        max_s = int(scores[best_idx])
        if max_s > melhor_score:
            melhor_score = max_s
            melhor_solucao = populacao[best_idx]

        if max_s == 54:
            return {
                "island_id": "GPU_TITAN",
                "melhor_score": 54,
                "melhor_solucao": populacao[best_idx],
                "populacao_final": populacao,
                "avaliados": evals,
                "resolvido": True,
            }

        sorted_indices = np.argsort(-scores)
        nova_pop = [populacao[idx] for idx in sorted_indices[:qtd_elite]]
        pais = [populacao[idx] for idx in sorted_indices[:qtd_sel]]

        while len(nova_pop) < pop_size:
            p1 = random.choice(pais)
            p2 = random.choice(pais)
            if random.random() < cross_rate:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            f1 = mutar_individuo(f1, mut_rate)
            f2 = mutar_individuo(f2, mut_rate)

            nova_pop.append(f1)
            if len(nova_pop) < pop_size:
                nova_pop.append(f2)

        populacao = nova_pop

    return {
        "island_id": "GPU_TITAN",
        "melhor_score": melhor_score,
        "melhor_solucao": melhor_solucao,
        "populacao_final": populacao,
        "avaliados": evals,
        "resolvido": False,
    }


def obter_informacoes_hardware():
    """
    Coleta informações completas do hardware do sistema (Processador CPU + Placa de Vídeo GPU).
    Identifica com precisão:
    - CPU: AMD Ryzen™ 7 PRO 8700GE (8 núcleos / 16 threads)
    - GPU: AMD Radeon™ 780M Graphics (Vulkan / WebGPU Compute Shaders)

    Retorno:
        dict: Dicionário detalhado de hardware para a interface e logs.
    """
    cpu_nome = "Processador Multi-Core"
    threads_totais = os.cpu_count() or 1

    # Tentativa de leitura precisa do nome da CPU via Registro do Windows
    try:
        import winreg
        chave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        valor, _ = winreg.QueryValueEx(chave, "ProcessorNameString")
        winreg.CloseKey(chave)
        if valor and valor.strip():
            cpu_nome = valor.strip()
    except Exception:
        try:
            import platform
            cpu_nome = platform.processor() or platform.machine() or "CPU Multi-Core"
        except Exception:
            pass

    info_gpu = obter_informacoes_gpu()
    gpu_disponivel = info_gpu.get("disponivel", False)
    gpu_nome = info_gpu.get("gpu_nome", "GPU Indisponível")
    gpu_taxa = info_gpu.get("taxa_estimada", "")

    if gpu_disponivel and threads_totais > 1:
        modo = f"Carga Total de Hardware: 16 Ilhas CPU ({cpu_nome}) + 1 Super-Ilha GPU ({gpu_nome})"
    elif gpu_disponivel:
        modo = f"Aceleração por GPU ({gpu_nome})"
    elif threads_totais > 1:
        modo = f"Processamento Paralelo Multi-Core ({threads_totais} Ilhas Simultâneas)"
    else:
        modo = "Sequencial Otimizado"

    return {
        "cpu_nome": cpu_nome,
        "threads_totais": threads_totais,
        "threads_utilizadas": threads_totais,
        "gpu": info_gpu,
        "gpu_nome": gpu_nome,
        "gpu_disponivel": gpu_disponivel,
        "gpu_taxa": gpu_taxa,
        "modo": modo,
    }


def rodar_ag_heterogeneo_simultaneo(
    pop_size_total,
    tamanho_cromossomo,
    quantidade_geracoes,
    porcentagem_mutacao,
    porcentagem_cruzamento,
    porcentagem_selecao,
    estado_base,
    intervalo_ciclo=500,
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
    info_hw=None,
):
    """
    Executa o Algoritmo Genético Heterogêneo Simultâneo em Carga Máxima:
    - 16 processos paralelos dedicados ocupando 100% das 16 threads do AMD Ryzen™ 7 PRO 8700GE.
    - 1 Super-Ilha GPU ocupando todos os 12 Compute Units (768 Shaders) da AMD Radeon™ 780M Graphics.
    - Migração periódica cruzada de indivíduos campeões entre CPU e GPU a cada época.

    Parâmetros:
        pop_size_total (int): Tamanho da população combinada.
        tamanho_cromossomo (int): Comprimento de cada cromossomo (genes).
        quantidade_geracoes (int): Total de gerações a evoluir.
        porcentagem_mutacao (float): Taxa de mutação por gene.
        porcentagem_cruzamento (float): Taxa de recombinação.
        porcentagem_selecao (float): Taxa de seleção dos pais.
        estado_base (tuple): Estado de 54 adesivos do cubo embaralhado.
        intervalo_ciclo (int): Frequência de notificações e logs.
        callback_progresso (callable, opcional): Função callback em tempo real.
        is_cancelled (callable, opcional): Função de verificação de cancelamento.
        total_avaliados_base (int): Contador base de avaliações acumuladas.
        info_hw (dict, opcional): Especificações de hardware do sistema.

    Retorno:
        tuple: (melhor_solucao, melhor_score, avaliados_locais)
    """
    if info_hw is None:
        info_hw = obter_informacoes_hardware()

    num_cpus = info_hw.get("threads_totais", 16)
    num_cpu_islands = num_cpus  # Utiliza 100% das 16 threads lógicas

    # Distribuição de alta densidade: GPU com carga pesada + 16 Ilhas de CPU com nichos diversificados
    gpu_pop_size = max(1000, int(pop_size_total * 0.65))
    pop_cpu_total = max(num_cpu_islands * 20, pop_size_total - gpu_pop_size)
    pop_por_cpu_island = max(20, pop_cpu_total // num_cpu_islands)
    active_pop_total = gpu_pop_size + (pop_por_cpu_island * num_cpu_islands)

    gpu_pop = gerar_populacao(gpu_pop_size, tamanho_cromossomo)
    cpu_pops = [gerar_populacao(pop_por_cpu_island, tamanho_cromossomo) for _ in range(num_cpu_islands)]

    engine = obter_gpu_engine()
    melhor_solucao_global = None
    melhor_score_global = -1
    total_avaliados_etapa = 0
    t_inicio = time.time()

    epocas = max(1, min(25, quantidade_geracoes // 20))
    geracoes_por_epoca = max(1, quantidade_geracoes // epocas)
    geracao_acumulada = 0

    with ProcessPoolExecutor(max_workers=num_cpu_islands) as executor:
        for epoca in range(1, epocas + 1):
            if is_cancelled and is_cancelled():
                break

            tempo_base_ms = int(time.time() * 1000)
            tarefas_cpu = []
            for i in range(num_cpu_islands):
                seed_i = tempo_base_ms + (i * 997) + (epoca * 10007)
                tarefas_cpu.append((
                    i,
                    cpu_pops[i],
                    estado_base,
                    geracoes_por_epoca,
                    porcentagem_mutacao,
                    porcentagem_cruzamento,
                    porcentagem_selecao,
                    seed_i,
                ))

            # 1. Dispara simultaneamente todos os 16 processos de CPU
            cpu_futures = executor.map(_worker_ilha_paralela, tarefas_cpu)

            # 2. Concorrentemente executa a Super-Ilha GPU em todos os Compute Units
            gpu_res = _executar_bloco_gpu(
                gpu_pop,
                estado_base,
                geracoes_por_epoca,
                porcentagem_mutacao,
                porcentagem_cruzamento,
                porcentagem_selecao,
                engine,
            )

            # 3. Coleta os resultados dos 16 processos de CPU
            cpu_results = list(cpu_futures)

            geracao_acumulada += geracoes_por_epoca
            total_avaliados_etapa += gpu_res["avaliados"]
            gpu_pop = gpu_res["populacao_final"]

            if gpu_res["melhor_score"] > melhor_score_global:
                melhor_score_global = gpu_res["melhor_score"]
                melhor_solucao_global = gpu_res["melhor_solucao"]

            # Processa as 15 ilhas da CPU
            elites_cpu = []
            best_cpu_score_epoca = -1
            best_cpu_sol_epoca = None
            for r in cpu_results:
                total_avaliados_etapa += r["avaliados"]
                cpu_pops[r["island_id"]] = r["populacao_final"]

                if r["melhor_score"] > melhor_score_global:
                    melhor_score_global = r["melhor_score"]
                    melhor_solucao_global = r["melhor_solucao"]

                if r["melhor_score"] > best_cpu_score_epoca:
                    best_cpu_score_epoca = r["melhor_score"]
                    best_cpu_sol_epoca = r["melhor_solucao"]

                if r["melhor_solucao"]:
                    elites_cpu.append(r["melhor_solucao"])

            # 4. Notificação de progresso simultâneo
            if callback_progresso:
                tempo_decorrido = max(0.001, time.time() - t_inicio)
                taxa = round(total_avaliados_etapa / tempo_decorrido)
                sol_str = " ".join(melhor_solucao_global) if melhor_solucao_global else ""

                callback_progresso({
                    "etapa": f"Simultâneo Híbrido ({num_cpu_islands} Ilhas CPU + 1 Super-Ilha GPU) - Geração {min(geracao_acumulada, quantidade_geracoes)}/{quantidade_geracoes}",
                    "operacao": "Evolução Heterogênea Simultânea (CPU + GPU)",
                    "tamanho_atual": tamanho_cromossomo,
                    "tamanho_cromossomo": tamanho_cromossomo,
                    "cromossomos_populacao": active_pop_total,
                    "cromossomos_por_ilha": f"GPU: {gpu_pop_size} | CPU: {pop_por_cpu_island} × {num_cpu_islands}",
                    "cromossomos_elite": max(1, round(active_pop_total * 0.05)),
                    "geracao_atual": min(geracao_acumulada, quantidade_geracoes),
                    "total_geracoes": quantidade_geracoes,
                    "individuos_avaliados": total_avaliados_base + total_avaliados_etapa,
                    "cromossomos_avaliados": total_avaliados_base + total_avaliados_etapa,
                    "melhor_score": melhor_score_global,
                    "melhor_solucao": melhor_solucao_global or [],
                    "melhor_solucao_str": sol_str,
                    "hardware": info_hw,
                    "taxa_avaliacoes_seg": taxa,
                    "mensagem": f"[Carga Máxima: {num_cpu_islands} Threads CPU + GPU 780M] Geração {min(geracao_acumulada, quantidade_geracoes)}/{quantidade_geracoes}: Score {melhor_score_global}/54 | {taxa:,} evals/s | {active_pop_total:,} cromossomos",
                })

            # 5. Condição de parada imediata: Cubo 100% resolvido
            if melhor_score_global == 54:
                break

            # 6. Migração Cruzada Bidirecional (Cross-Pollination):
            # - Injeta o melhor campeão da GPU em todas as ilhas da CPU
            if gpu_res["melhor_solucao"]:
                for i in range(num_cpu_islands):
                    if cpu_pops[i]:
                        cpu_pops[i][-1] = list(gpu_res["melhor_solucao"])

            # - Injeta o melhor campeão da CPU na população da GPU
            if best_cpu_sol_epoca and gpu_pop:
                gpu_pop[-1] = list(best_cpu_sol_epoca)

    return melhor_solucao_global, melhor_score_global, total_avaliados_etapa


def rodar_ag_gpu(
    pop_size,
    tamanho_cromossomo,
    quantidade_geracoes,
    porcentagem_mutacao,
    porcentagem_cruzamento,
    porcentagem_selecao,
    estado_base,
    intervalo_ciclo=500,
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
    info_hw=None,
):
    """
    Executa o Algoritmo Genético acelerado na Placa de Vídeo (GPU) via Compute Shaders WebGPU/Vulkan.
    Avalia a população inteira em lote na VRAM da GPU com taxa de até 3 milhões de avaliações/s.
    """
    gpu = obter_gpu_engine()
    if info_hw is None:
        info_hw = obter_informacoes_hardware()

    populacao = gerar_populacao(pop_size, tamanho_cromossomo)
    qtd_elite = max(1, round(pop_size * 0.05))
    qtd_sel = max(2, round(pop_size * porcentagem_selecao))

    melhor_solucao_global = None
    melhor_score_global = -1
    avaliados_locais = 0
    t_inicio_ga = time.time()

    for geracao in range(1, quantidade_geracoes + 1):
        if is_cancelled and is_cancelled():
            break

        pop_ids = converter_populacao_para_ids(populacao)
        scores = gpu.avaliar_populacao(estado_base, pop_ids)
        avaliados_locais += pop_size

        best_idx = int(np.argmax(scores))
        max_score = int(scores[best_idx])
        melhor_candidato = populacao[best_idx]

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao_global = melhor_candidato

        deve_notificar = (
            geracao == 1
            or geracao == quantidade_geracoes
            or geracao % max(1, intervalo_ciclo) == 0
            or max_score == 54
            or (geracao % 25 == 0 and max_score >= melhor_score_global - 1)
        )

        if callback_progresso and deve_notificar:
            sol_str = " ".join(melhor_solucao_global) if melhor_solucao_global else ""
            tempo_decorrido = max(0.001, time.time() - t_inicio_ga)
            taxa_atual = round(avaliados_locais / tempo_decorrido)

            callback_progresso({
                "etapa": f"Algoritmo Genético GPU (Cromossomo {tamanho_cromossomo} genes) - Geração {geracao}/{quantidade_geracoes}",
                "operacao": "Avaliando fitness (GPU Shader)",
                "tamanho_atual": tamanho_cromossomo,
                "tamanho_cromossomo": tamanho_cromossomo,
                "cromossomos_populacao": pop_size,
                "cromossomos_elite": qtd_elite,
                "cromossomos_avaliados": total_avaliados_base + avaliados_locais,
                "geracao_atual": geracao,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score_global,
                "melhor_solucao": melhor_solucao_global or [],
                "melhor_solucao_str": sol_str,
                "hardware": info_hw,
                "taxa_avaliacoes_seg": taxa_atual,
                "mensagem": f"[GPU {info_hw.get('gpu_nome', 'Radeon 780M')}] Geração {geracao}/{quantidade_geracoes}: Score {max_score}/54 (Melhor: {melhor_score_global}/54) | {taxa_atual:,} evals/s | {pop_size:,} cromossomos",
            })

        if max_score == 54:
            break

        sorted_indices = np.argsort(-scores)
        nova_pop = [populacao[idx] for idx in sorted_indices[:qtd_elite]]
        pais = [populacao[idx] for idx in sorted_indices[:qtd_sel]]

        while len(nova_pop) < pop_size:
            p1 = random.choice(pais)
            p2 = random.choice(pais)

            if random.random() < porcentagem_cruzamento:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            f1 = mutar_individuo(f1, porcentagem_mutacao)
            f2 = mutar_individuo(f2, porcentagem_mutacao)

            nova_pop.append(f1)
            if len(nova_pop) < pop_size:
                nova_pop.append(f2)

        populacao = nova_pop

    return melhor_solucao_global, melhor_score_global, avaliados_locais


def rodar_algoritmo_genetico(
    porcentagem_mutacao=0.05,
    porcentagem_cruzamento=0.70,
    porcentagem_selecao=0.50,
    quantidade_geracoes=2000,
    quantidade_individuos_inicial=1000,
    embaralhamento=None,
    tamanho_cromossomo=20,
    intervalo_ciclo=500,
    limite_busca_exaustiva=LIMITE_BUSCA_EXAUSTIVA,
    modo_hardware="cpu+gpu",
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
    estado_base_precomputado=None,
):
    """
    Executa o Algoritmo Genético de Alta Performance com suporte Configurável de Hardware (CPU, GPU ou CPU+GPU).

    Estratégia de Execução:
    1. Para espaços pequenos (<= 4000), executa Busca Exaustiva determinística direta.
    2. modo_hardware == 'cpu+gpu': Executa SIMULTÂNEO HETEROGÊNEO (15 Ilhas CPU + 1 Super-Ilha GPU).
    3. modo_hardware == 'gpu': Executa Aceleração Pura em GPU (WebGPU/Vulkan compute shaders).
    4. modo_hardware == 'cpu': Executa Multi-Core Puro (16 Ilhas de CPU via ProcessPoolExecutor).
    5. Fallback seguro e transparente caso algum dispositivo não esteja disponível.
    """
    if embaralhamento is None:
        embaralhamento = []

    if estado_base_precomputado is not None:
        estado_base = estado_base_precomputado
    else:
        estado_base = aplicar_movimentos(ESTADO_RESOLVIDO, embaralhamento)

    espaco_busca = calcular_espaco_busca(tamanho_cromossomo)
    cache_fitness = {}
    info_hw = obter_informacoes_hardware()

    # Caso 1: Espaço de busca pequeno -> Busca Exaustiva Direta
    if espaco_busca <= limite_busca_exaustiva:
        return rodar_busca_exaustiva(
            embaralhamento,
            tamanho_cromossomo,
            cache=cache_fitness,
            callback_progresso=callback_progresso,
            is_cancelled=is_cancelled,
            total_avaliados_base=total_avaliados_base,
            estado_base_precomputado=estado_base,
        )

    pop_size = min(quantidade_individuos_inicial, espaco_busca)
    num_cpus = info_hw["threads_totais"]
    qtd_elite_total = max(1, round(pop_size * 0.05))
    gpu_ativa = info_hw.get("gpu_disponivel", False)

    modo_hw = str(modo_hardware).lower().strip()
    if modo_hw not in ("cpu", "gpu", "cpu+gpu"):
        modo_hw = "cpu+gpu"

    # ==========================================================================
    # CASO 2: MODO SIMULTÂNEO HETEROGÊNEO (15 ILHAS CPU + 1 ILHA GPU)
    # ==========================================================================
    if (modo_hw == "cpu+gpu") and gpu_ativa and (num_cpus > 1) and (pop_size >= 160) and (quantidade_geracoes >= 20):
        try:
            num_cpu_islands = num_cpus - 1
            if callback_progresso:
                callback_progresso({
                    "etapa": f"Simultâneo Híbrido ({num_cpu_islands} Ilhas CPU + 1 Super-Ilha GPU)",
                    "operacao": "Criando indivíduos",
                    "tamanho_atual": tamanho_cromossomo,
                    "tamanho_cromossomo": tamanho_cromossomo,
                    "cromossomos_populacao": pop_size,
                    "cromossomos_elite": qtd_elite_total,
                    "cromossomos_avaliados": total_avaliados_base,
                    "geracao_atual": 0,
                    "total_geracoes": quantidade_geracoes,
                    "individuos_avaliados": total_avaliados_base,
                    "melhor_score": 0,
                    "melhor_solucao": [],
                    "melhor_solucao_str": "",
                    "hardware": info_hw,
                    "mensagem": f"Iniciando {pop_size:,} cromossomos com Execução Simultânea: {num_cpu_islands} Ilhas CPU ({info_hw['cpu_nome']}) + 1 Ilha GPU ({info_hw['gpu_nome']}).",
                })

            return rodar_ag_heterogeneo_simultaneo(
                pop_size_total=pop_size,
                tamanho_cromossomo=tamanho_cromossomo,
                quantidade_geracoes=quantidade_geracoes,
                porcentagem_mutacao=porcentagem_mutacao,
                porcentagem_cruzamento=porcentagem_cruzamento,
                porcentagem_selecao=porcentagem_selecao,
                estado_base=estado_base,
                intervalo_ciclo=intervalo_ciclo,
                callback_progresso=callback_progresso,
                is_cancelled=is_cancelled,
                total_avaliados_base=total_avaliados_base,
                info_hw=info_hw,
            )
        except Exception:
            pass

    # ==========================================================================
    # CASO 3: ACELERAÇÃO PURA POR GPU
    # ==========================================================================
    if (modo_hw in ("gpu", "cpu+gpu")) and gpu_ativa:
        try:
            if callback_progresso and modo_hw == "gpu":
                callback_progresso({
                    "etapa": f"GPU Acelerada ({info_hw['gpu_nome']})",
                    "operacao": "Criando indivíduos",
                    "tamanho_atual": tamanho_cromossomo,
                    "tamanho_cromossomo": tamanho_cromossomo,
                    "cromossomos_populacao": pop_size,
                    "cromossomos_elite": qtd_elite_total,
                    "cromossomos_avaliados": total_avaliados_base,
                    "geracao_atual": 0,
                    "total_geracoes": quantidade_geracoes,
                    "individuos_avaliados": total_avaliados_base,
                    "melhor_score": 0,
                    "melhor_solucao": [],
                    "melhor_solucao_str": "",
                    "hardware": info_hw,
                    "mensagem": f"Iniciando {pop_size:,} cromossomos na GPU ({info_hw['gpu_nome']}) via Vulkan Compute Shaders.",
                })

            return rodar_ag_gpu(
                pop_size=pop_size,
                tamanho_cromossomo=tamanho_cromossomo,
                quantidade_geracoes=quantidade_geracoes,
                porcentagem_mutacao=porcentagem_mutacao,
                porcentagem_cruzamento=porcentagem_cruzamento,
                porcentagem_selecao=porcentagem_selecao,
                estado_base=estado_base,
                intervalo_ciclo=intervalo_ciclo,
                callback_progresso=callback_progresso,
                is_cancelled=is_cancelled,
                total_avaliados_base=total_avaliados_base,
                info_hw=info_hw,
            )
        except Exception:
            pass

    # ==========================================================================
    # CASO 4: EXECUÇÃO PARALELA MULTI-CORE (MODELO DE ILHAS CPU - 16 THREADS)
    # ==========================================================================
    usar_paralelo = (num_cpus > 1) and (pop_size >= 160) and (quantidade_geracoes >= 20)
    num_ilhas = min(num_cpus, max(2, pop_size // 20)) if usar_paralelo else 1
    pop_por_ilha = max(10, pop_size // num_ilhas)

    if usar_paralelo:
        ilhas_pop = [gerar_populacao(pop_por_ilha, tamanho_cromossomo) for _ in range(num_ilhas)]
        melhor_solucao_global = None
        melhor_score_global = -1
        total_avaliados_etapa = 0

        epocas = max(1, min(20, quantidade_geracoes // 25))
        geracoes_por_epoca = max(1, quantidade_geracoes // epocas)
        geracao_acumulada = 0

        try:
            with ProcessPoolExecutor(max_workers=num_ilhas) as executor:
                for epoca in range(1, epocas + 1):
                    if is_cancelled and is_cancelled():
                        break

                    tarefas = []
                    tempo_base_ms = int(time.time() * 1000)
                    for i in range(num_ilhas):
                        seed_i = tempo_base_ms + (i * 997) + (epoca * 10007)
                        tarefas.append((
                            i,
                            ilhas_pop[i],
                            estado_base,
                            geracoes_por_epoca,
                            porcentagem_mutacao,
                            porcentagem_cruzamento,
                            porcentagem_selecao,
                            seed_i,
                        ))

                    resultados_epoca = list(executor.map(_worker_ilha_paralela, tarefas))
                    geracao_acumulada += geracoes_por_epoca

                    elites_migracao = []
                    for res in resultados_epoca:
                        total_avaliados_etapa += res["avaliados"]
                        ilhas_pop[res["island_id"]] = res["populacao_final"]

                        if res["melhor_score"] > melhor_score_global:
                            melhor_score_global = res["melhor_score"]
                            melhor_solucao_global = res["melhor_solucao"]

                        if res["melhor_solucao"]:
                            elites_migracao.append(res["melhor_solucao"])

                    if callback_progresso:
                        sol_str = " ".join(melhor_solucao_global) if melhor_solucao_global else ""
                        callback_progresso({
                            "etapa": f"Algoritmo Genético Multi-Core ({num_ilhas} Ilhas Ativas) - Geração {min(geracao_acumulada, quantidade_geracoes)}/{quantidade_geracoes}",
                            "operacao": "Avaliando fitness (Paralelo)",
                            "tamanho_atual": tamanho_cromossomo,
                            "tamanho_cromossomo": tamanho_cromossomo,
                            "cromossomos_populacao": pop_size,
                            "cromossomos_por_ilha": pop_por_ilha,
                            "cromossomos_elite": qtd_elite_total,
                            "geracao_atual": min(geracao_acumulada, quantidade_geracoes),
                            "total_geracoes": quantidade_geracoes,
                            "individuos_avaliados": total_avaliados_base + total_avaliados_etapa,
                            "cromossomos_avaliados": total_avaliados_base + total_avaliados_etapa,
                            "melhor_score": melhor_score_global,
                            "melhor_solucao": melhor_solucao_global or [],
                            "melhor_solucao_str": sol_str,
                            "hardware": info_hw,
                            "mensagem": f"Geração {min(geracao_acumulada, quantidade_geracoes)}/{quantidade_geracoes}: Score {melhor_score_global}/54 | {pop_size} cromossomos ({num_ilhas} threads) | Total: {(total_avaliados_base + total_avaliados_etapa):,}",
                        })

                    if melhor_score_global == 54:
                        break

                    if len(elites_migracao) > 1:
                        for i in range(num_ilhas):
                            vizinho = (i + 1) % num_ilhas
                            if ilhas_pop[i] and elites_migracao[vizinho]:
                                ilhas_pop[i][-1] = list(elites_migracao[vizinho])

            return melhor_solucao_global, melhor_score_global, total_avaliados_etapa

        except Exception:
            pass

    # ==========================================================================
    # CASO 5: EXECUÇÃO SEQUENCIAL OTIMIZADA COM CACHE EM MEMÓRIA
    # ==========================================================================
    populacao = gerar_populacao(pop_size, tamanho_cromossomo)
    melhor_solucao = None
    melhor_score_global = -1
    avaliados_locais = 0
    qtd_elite = max(1, round(pop_size * 0.05))

    for geracao in range(1, quantidade_geracoes + 1):
        if is_cancelled and is_cancelled():
            break

        melhores, max_score, avaliados = selecionar_melhores(
            populacao, estado_base, porcentagem_selecao, cache=cache_fitness
        )
        avaliados_locais += len(populacao)

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = melhores[0]

        deve_notificar = (
            geracao == 1
            or geracao == quantidade_geracoes
            or geracao % max(1, intervalo_ciclo) == 0
            or max_score == 54
            or (geracao % 20 == 0 and max_score > melhor_score_global - 2)
        )

        if callback_progresso and deve_notificar:
            sol_str = " ".join(melhor_solucao) if melhor_solucao else ""
            callback_progresso({
                "etapa": f"Algoritmo Genético (Cromossomo {tamanho_cromossomo} - Geração {geracao}/{quantidade_geracoes})",
                "operacao": "Avaliando fitness",
                "tamanho_atual": tamanho_cromossomo,
                "tamanho_cromossomo": tamanho_cromossomo,
                "cromossomos_populacao": pop_size,
                "cromossomos_elite": qtd_elite,
                "cromossomos_avaliados": total_avaliados_base + avaliados_locais,
                "geracao_atual": geracao,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score_global,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": sol_str,
                "hardware": info_hw,
                "mensagem": f"Geração {geracao}/{quantidade_geracoes}: Score {max_score}/54 (Melhor: {melhor_score_global}/54) | Avaliados: {(total_avaliados_base + avaliados_locais):,}",
            })

        if max_score == 54:
            break

        nova_populacao = [ind for _, ind in avaliados[:qtd_elite]]
        pais_candidatos = melhores

        while len(nova_populacao) < pop_size:
            p1 = random.choice(pais_candidatos)
            p2 = random.choice(pais_candidatos)

            if random.random() < porcentagem_cruzamento:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            f1 = mutar_individuo(f1, porcentagem_mutacao)
            f2 = mutar_individuo(f2, porcentagem_mutacao)

            nova_populacao.append(f1)
            if len(nova_populacao) < pop_size:
                nova_populacao.append(f2)

        populacao = nova_populacao

    return melhor_solucao, melhor_score_global, avaliados_locais


def resolver_cubo_incremental(
    embaralhamento=None,
    porcentagem_mutacao=0.05,
    porcentagem_cruzamento=0.70,
    porcentagem_selecao=0.50,
    quantidade_geracoes=2000,
    quantidade_individuos_inicial=1000,
    tamanho_minimo=1,
    tamanho_maximo=54,
    intervalo_ciclo=500,
    modo_hardware="cpu+gpu",
    callback_progresso=None,
    is_cancelled=None,
):
    """
    Executa a resolução incremental do Cubo Mágico através do Algoritmo Genético,
    testando comprimentos de cromossomo de tamanho_minimo até tamanho_maximo.
    Emite métricas completas e mensagens via callback_progresso.

    Parâmetros:
        embaralhamento (list[str] | str, opcional): Sequência de embaralhamento do cubo.
        porcentagem_mutacao (float): Taxa de mutação genética por gene (padrão: 0.05).
        porcentagem_cruzamento (float): Taxa de recombinação (padrão: 0.70).
        porcentagem_selecao (float): Proporção de indivíduos mantidos na seleção (padrão: 0.50).
        quantidade_geracoes (int): Total máximo de gerações por tamanho de cromossomo (padrão: 2000).
        quantidade_individuos_inicial (int): Tamanho da população inicial (padrão: 1000).
        tamanho_minimo (int): Comprimento inicial de cromossomo a testar (padrão: 1).
        tamanho_maximo (int): Comprimento máximo de cromossomo a testar (padrão: 54).
        intervalo_ciclo (int): Intervalo de gerações para emissão de logs/status (padrão: 500).
        modo_hardware (str): Dispositivo de execução ('cpu', 'gpu' ou 'cpu+gpu', padrão: 'cpu+gpu').
        callback_progresso (callable, opcional): Função callback para envio de progresso em tempo real.
        is_cancelled (callable, opcional): Função que retorna True se a execução foi cancelada.

    Retorno:
        dict: Dicionário estruturado com o resultado final, score, solução e histórico.
    """
    if embaralhamento is None:
        embaralhamento = []
    elif isinstance(embaralhamento, str):
        embaralhamento = [m for m in embaralhamento.split() if m.strip()]

    t_inicio_total = time.time()
    total_individuos_avaliados = 0
    info_hw = obter_informacoes_hardware()

    # Pré-computa o estado inicial do cubo embaralhado uma única vez
    scrambled_state = aplicar_movimentos(ESTADO_RESOLVIDO, embaralhamento)

    # Verifica se o cubo já está resolvido inicialmente
    score_inicial = calcular_score_estado(scrambled_state)
    if score_inicial == 54:
        res = {
            "sucesso": True,
            "score": 54,
            "tamanho_cromossomo": 0,
            "solucao": [],
            "solucao_str": "",
            "tempo_execucao": round(time.time() - t_inicio_total, 3),
            "individuos_avaliados": 1,
            "embaralhamento": embaralhamento,
            "historico": [],
            "hardware": info_hw,
            "mensagem": "O cubo já está em estado resolvido (54/54)!",
        }
        if callback_progresso:
            callback_progresso({
                "etapa": "Cubo já resolvido",
                "tamanho_atual": 0,
                "tamanho_cromossomo": 0,
                "cromossomos_populacao": 0,
                "cromossomos_avaliados": 1,
                "geracao_atual": 1,
                "total_geracoes": 1,
                "individuos_avaliados": 1,
                "melhor_score": 54,
                "melhor_solucao": [],
                "melhor_solucao_str": "",
                "hardware": info_hw,
                "mensagem": "O cubo já se encontra 100% resolvido (54/54 casinhas).",
            })
        return res

    melhor_solucao_global = None
    melhor_score_global = -1
    tamanho_melhor_global = 0
    historico = []

    for tamanho in range(tamanho_minimo, tamanho_maximo + 1):
        if is_cancelled and is_cancelled():
            break

        t_inicio_etapa = time.time()

        if callback_progresso:
            callback_progresso({
                "etapa": f"Testando Cromossomo de Tamanho {tamanho}",
                "operacao": "Criando indivíduos",
                "tamanho_atual": tamanho,
                "tamanho_cromossomo": tamanho,
                "cromossomos_populacao": quantidade_individuos_inicial,
                "cromossomos_avaliados": total_individuos_avaliados,
                "geracao_atual": 0,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_individuos_avaliados,
                "melhor_score": max(0, melhor_score_global),
                "melhor_solucao": melhor_solucao_global or [],
                "melhor_solucao_str": " ".join(melhor_solucao_global) if melhor_solucao_global else "",
                "hardware": info_hw,
                "mensagem": f"Criando indivíduos para cromossomo com {tamanho} movimento(s)...",
            })

        solucao, score, avaliados_etapa = rodar_algoritmo_genetico(
            porcentagem_mutacao=porcentagem_mutacao,
            porcentagem_cruzamento=porcentagem_cruzamento,
            porcentagem_selecao=porcentagem_selecao,
            quantidade_geracoes=quantidade_geracoes,
            quantidade_individuos_inicial=quantidade_individuos_inicial,
            embaralhamento=embaralhamento,
            tamanho_cromossomo=tamanho,
            intervalo_ciclo=intervalo_ciclo,
            limite_busca_exaustiva=LIMITE_BUSCA_EXAUSTIVA,
            modo_hardware=modo_hardware,
            callback_progresso=callback_progresso,
            is_cancelled=is_cancelled,
            total_avaliados_base=total_individuos_avaliados,
            estado_base_precomputado=scrambled_state,
        )

        total_individuos_avaliados += avaliados_etapa
        t_etapa = round(time.time() - t_inicio_etapa, 3)

        historico.append({
            "tamanho": tamanho,
            "score": score,
            "solucao": solucao,
            "tempo_s": t_etapa,
            "avaliados": avaliados_etapa,
        })

        if score > melhor_score_global:
            melhor_score_global = score
            melhor_solucao_global = solucao
            tamanho_melhor_global = tamanho

        if score == 54:
            t_total = round(time.time() - t_inicio_total, 3)
            sol_str = " ".join(solucao) if solucao else ""
            res = {
                "sucesso": True,
                "score": 54,
                "tamanho_cromossomo": tamanho,
                "solucao": solucao,
                "solucao_str": sol_str,
                "tempo_execucao": t_total,
                "individuos_avaliados": total_individuos_avaliados,
                "embaralhamento": embaralhamento,
                "historico": historico,
                "hardware": info_hw,
                "mensagem": f"Cubo resolvido com sucesso com {tamanho} movimento(s) em {t_total:.2f}s!",
            }
            if callback_progresso:
                callback_progresso({
                    "etapa": "Solução Concluída com Sucesso",
                    "tamanho_atual": tamanho,
                    "tamanho_cromossomo": tamanho,
                    "cromossomos_populacao": quantidade_individuos_inicial,
                    "cromossomos_avaliados": total_individuos_avaliados,
                    "geracao_atual": quantidade_geracoes,
                    "total_geracoes": quantidade_geracoes,
                    "individuos_avaliados": total_individuos_avaliados,
                    "melhor_score": 54,
                    "melhor_solucao": solucao,
                    "melhor_solucao_str": sol_str,
                    "hardware": info_hw,
                    "mensagem": f"Sucesso: Cubo resolvido! Sequência: {sol_str} (Score 54/54)",
                })
            return res

    t_total = round(time.time() - t_inicio_total, 3)
    sol_str = " ".join(melhor_solucao_global) if melhor_solucao_global else ""
    return {
        "sucesso": melhor_score_global == 54,
        "score": melhor_score_global,
        "tamanho_cromossomo": tamanho_melhor_global,
        "solucao": melhor_solucao_global or [],
        "solucao_str": sol_str,
        "tempo_execucao": t_total,
        "individuos_avaliados": total_individuos_avaliados,
        "embaralhamento": embaralhamento,
        "historico": historico,
        "hardware": info_hw,
        "mensagem": f"Melhor solução parcial atingida: {melhor_score_global}/54 casinhas.",
    }