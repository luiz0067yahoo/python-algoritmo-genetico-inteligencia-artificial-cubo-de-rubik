import copy
import random
import time
from cruzamento import cruzar_dois_individuos, cruzamento
from mutacao import mutar_individuo, mutacao
from pontuacao import calcular_score
from populacao import (
    MOVIMENTOS,
    PARALELAS,
    calcular_espaco_busca,
    gerar_populacao,
    gerar_todas_combinacoes_validas,
)

# Limite de combinações para executar busca exaustiva direta (em vez de AG)
LIMITE_BUSCA_EXAUSTIVA = 4000


def selecionar_melhores(populacao, embaralhamento, porcentagem_selecao, cache=None):
    """
    Avalia a população calculando o fitness de cada indivíduo (com suporte a cache)
    e retorna os melhores indivíduos com base no percentual de seleção.
    """
    avaliados = []
    for ind in populacao:
        score = calcular_score(embaralhamento, ind, cache=cache)
        avaliados.append((score, ind))

    # Ordena do maior fitness para o menor
    avaliados.sort(key=lambda x: x[0], reverse=True)

    # Determina a quantidade a manter
    qtd_selecionada = max(1, round(len(populacao) * porcentagem_selecao))
    melhores = [ind for _, ind in avaliados[:qtd_selecionada]]

    best_score = avaliados[0][0]
    return melhores, best_score, avaliados


def rodar_busca_exaustiva(
    embaralhamento,
    tamanho_cromossomo,
    cache=None,
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
):
    """
    Executa a busca exaustiva determinística para espaços pequenos (18, 270, 3888).
    Atualiza o callback_progresso com o progresso de indivíduos e etapas.
    """
    todas_combinacoes = gerar_todas_combinacoes_validas(tamanho_cromossomo)
    total = len(todas_combinacoes)
    t_inicio = time.time()

    melhor_score = -1
    melhor_solucao = None
    avaliados_locais = 0

    if callback_progresso:
        callback_progresso({
            "etapa": f"Busca Exaustiva (Tamanho {tamanho_cromossomo} - {total} combinações)",
            "operacao": "Criando indivíduos",
            "tamanho_atual": tamanho_cromossomo,
            "geracao_atual": 1,
            "total_geracoes": 1,
            "individuos_avaliados": total_avaliados_base,
            "melhor_score": 0,
            "melhor_solucao": [],
            "melhor_solucao_str": "",
            "mensagem": f"Criando indivíduos: Geradas {total:,} combinações válidas para busca exaustiva (tamanho {tamanho_cromossomo}).",
        })

    for i, ind in enumerate(todas_combinacoes, 1):
        if is_cancelled and is_cancelled():
            break

        avaliados_locais += 1
        score = calcular_score(embaralhamento, ind, cache=cache)

        if score > melhor_score:
            melhor_score = score
            melhor_solucao = ind

        # Notificação periódica durante a busca
        if callback_progresso and (i % 250 == 0 or i == total or score == 54):
            sol_str = " ".join(melhor_solucao) if melhor_solucao else ""
            callback_progresso({
                "etapa": f"Busca Exaustiva (Tamanho {tamanho_cromossomo})",
                "operacao": "Avaliando fitness",
                "tamanho_atual": tamanho_cromossomo,
                "geracao_atual": 1,
                "total_geracoes": 1,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": sol_str,
                "mensagem": f"Busca Exaustiva ({i}/{total}): Score Atual {melhor_score}/54 [{sol_str}]",
            })

        # Se encontrou solução perfeita (54/54), encerra imediatamente
        if score == 54:
            break

    return melhor_solucao, melhor_score, avaliados_locais


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
    callback_progresso=None,
    is_cancelled=None,
    total_avaliados_base=0,
):
    """
    Executa o Algoritmo Genético com Elitismo, Cruzamento, Mutação e Cache de Fitness.
    Atualiza as variáveis de progresso através de callback_progresso sem prints no terminal.
    """
    if embaralhamento is None:
        embaralhamento = []

    espaco_busca = calcular_espaco_busca(tamanho_cromossomo)
    cache_fitness = {}

    # Caso 1: Espaço de busca pequeno -> Busca Exaustiva Direta
    if espaco_busca <= limite_busca_exaustiva:
        return rodar_busca_exaustiva(
            embaralhamento,
            tamanho_cromossomo,
            cache=cache_fitness,
            callback_progresso=callback_progresso,
            is_cancelled=is_cancelled,
            total_avaliados_base=total_avaliados_base,
        )

    # Caso 2: Espaço de busca grande -> Algoritmo Genético
    pop_size = min(quantidade_individuos_inicial, espaco_busca)

    if callback_progresso:
        callback_progresso({
            "etapa": f"Algoritmo Genético (Cromossomo {tamanho_cromossomo})",
            "operacao": "Criando indivíduos",
            "tamanho_atual": tamanho_cromossomo,
            "geracao_atual": 0,
            "total_geracoes": quantidade_geracoes,
            "individuos_avaliados": total_avaliados_base,
            "melhor_score": 0,
            "melhor_solucao": [],
            "melhor_solucao_str": "",
            "mensagem": f"Criando indivíduos: População inicial de {pop_size} indivíduos gerada para cromossomo tamanho {tamanho_cromossomo}.",
        })

    populacao = gerar_populacao(pop_size, tamanho_cromossomo)

    melhor_solucao = None
    melhor_score_global = -1
    geracao_resolvido = -1
    avaliados_locais = 0
    t_inicio = time.time()

    qtd_elite = max(1, round(pop_size * 0.05))

    for geracao in range(1, quantidade_geracoes + 1):
        if is_cancelled and is_cancelled():
            break

        # 1. Avaliação e ordenação por fitness
        melhores, max_score, avaliados = selecionar_melhores(
            populacao, embaralhamento, porcentagem_selecao, cache=cache_fitness
        )
        avaliados_locais += len(populacao)

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = melhores[0]

        # Notificação periódica de avaliação
        deve_notificar = (
            geracao == 1
            or geracao == quantidade_geracoes
            or geracao % max(1, intervalo_ciclo) == 0
            or max_score == 54
            or (geracao % 10 == 0 and max_score > melhor_score_global - 2)
        )

        if callback_progresso and deve_notificar:
            sol_str = " ".join(melhor_solucao) if melhor_solucao else ""
            callback_progresso({
                "etapa": f"Algoritmo Genético (Cromossomo {tamanho_cromossomo} - Geração {geracao}/{quantidade_geracoes})",
                "operacao": "Avaliando fitness",
                "tamanho_atual": tamanho_cromossomo,
                "geracao_atual": geracao,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score_global,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": sol_str,
                "mensagem": f"Geração {geracao}/{quantidade_geracoes}: Score {max_score}/54 (Melhor: {melhor_score_global}/54) | Avaliados: {(total_avaliados_base + avaliados_locais):,}",
            })

        # Condição de parada imediata: Cubo 100% resolvido
        if max_score == 54:
            geracao_resolvido = geracao
            break

        # 2. Elitismo
        nova_populacao = [ind for _, ind in avaliados[:qtd_elite]]
        pais_candidatos = melhores

        # Notifica início das operações genéticas de cruzamento e mutação
        if callback_progresso and (geracao % max(1, intervalo_ciclo) == 0 or geracao == 1):
            callback_progresso({
                "etapa": f"Algoritmo Genético (Cromossomo {tamanho_cromossomo} - Geração {geracao}/{quantidade_geracoes})",
                "operacao": "Fazendo cruzamento",
                "tamanho_atual": tamanho_cromossomo,
                "geracao_atual": geracao,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score_global,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": " ".join(melhor_solucao) if melhor_solucao else "",
                "mensagem": f"Geração {geracao}: Fazendo cruzamento entre os melhores indivíduos selecionados...",
            })

        # 3. Cruzamento e Mutação
        houve_mutacao = False
        while len(nova_populacao) < pop_size:
            p1 = random.choice(pais_candidatos)
            p2 = random.choice(pais_candidatos)

            if random.random() < porcentagem_cruzamento:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            # Mutação
            f1_antigo = list(f1)
            f1 = mutar_individuo(f1, porcentagem_mutacao)
            f2 = mutar_individuo(f2, porcentagem_mutacao)
            if f1 != f1_antigo:
                houve_mutacao = True

            nova_populacao.append(f1)
            if len(nova_populacao) < pop_size:
                nova_populacao.append(f2)

        if callback_progresso and (geracao % max(1, intervalo_ciclo) == 0 or geracao == 1) and houve_mutacao:
            callback_progresso({
                "etapa": f"Algoritmo Genético (Cromossomo {tamanho_cromossomo} - Geração {geracao}/{quantidade_geracoes})",
                "operacao": "Fazendo mutação",
                "tamanho_atual": tamanho_cromossomo,
                "geracao_atual": geracao,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_avaliados_base + avaliados_locais,
                "melhor_score": melhor_score_global,
                "melhor_solucao": melhor_solucao or [],
                "melhor_solucao_str": " ".join(melhor_solucao) if melhor_solucao else "",
                "mensagem": f"Geração {geracao}: Fazendo mutação com taxa de {porcentagem_mutacao * 100:.1f}% nos novos descendentes.",
            })

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
    callback_progresso=None,
    is_cancelled=None,
):
    """
    Executa a resolução incremental do Cubo Mágico através do Algoritmo Genético,
    testando comprimentos de cromossomo de tamanho_minimo até tamanho_maximo.
    Emite métricas completas e mensagens via callback_progresso sem prints.
    """
    if embaralhamento is None:
        embaralhamento = []
    elif isinstance(embaralhamento, str):
        embaralhamento = [m for m in embaralhamento.split() if m.strip()]

    t_inicio_total = time.time()
    total_individuos_avaliados = 0

    # Verifica se o cubo já está resolvido inicialmente
    score_inicial = calcular_score(embaralhamento, [])
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
            "mensagem": "O cubo já está em estado resolvido (54/54)!",
        }
        if callback_progresso:
            callback_progresso({
                "etapa": "Cubo já resolvido",
                "tamanho_atual": 0,
                "geracao_atual": 1,
                "total_geracoes": 1,
                "individuos_avaliados": 1,
                "melhor_score": 54,
                "melhor_solucao": [],
                "melhor_solucao_str": "",
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
                "geracao_atual": 0,
                "total_geracoes": quantidade_geracoes,
                "individuos_avaliados": total_individuos_avaliados,
                "melhor_score": max(0, melhor_score_global),
                "melhor_solucao": melhor_solucao_global or [],
                "melhor_solucao_str": " ".join(melhor_solucao_global) if melhor_solucao_global else "",
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
            callback_progresso=callback_progresso,
            is_cancelled=is_cancelled,
            total_avaliados_base=total_individuos_avaliados,
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
                "mensagem": f"Cubo resolvido com sucesso com {tamanho} movimento(s) em {t_total:.2f}s!",
            }
            if callback_progresso:
                callback_progresso({
                    "etapa": "Solução Concluída com Sucesso",
                    "tamanho_atual": tamanho,
                    "geracao_atual": quantidade_geracoes,
                    "total_geracoes": quantidade_geracoes,
                    "individuos_avaliados": total_individuos_avaliados,
                    "melhor_score": 54,
                    "melhor_solucao": solucao,
                    "melhor_solucao_str": sol_str,
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
        "mensagem": f"Melhor solução parcial atingida: {melhor_score_global}/54 casinhas.",
    }