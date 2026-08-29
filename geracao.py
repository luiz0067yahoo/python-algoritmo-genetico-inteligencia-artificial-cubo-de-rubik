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


def rodar_busca_exaustiva(embaralhamento, tamanho_cromossomo, cache=None):
    """
    Executa a busca exaustiva determinística quando o espaço de combinações
    é pequeno (ex: 18 para tamanho 1, 270 para tamanho 2, 3888 para tamanho 3).
    Garante 100% de precisão e rapidez máxima sem loops desnecessários.
    """
    todas_combinacoes = gerar_todas_combinacoes_validas(tamanho_cromossomo)
    total = len(todas_combinacoes)

    print(f"=== BUSCA EXAUSTIVA DIRETA ({tamanho_cromossomo} MOVIMENTO(S) | {total:,} COMBINAÇÕES) ===")
    t_inicio = time.time()

    melhor_score = -1
    melhor_solucao = None

    for i, ind in enumerate(todas_combinacoes, 1):
        score = calcular_score(embaralhamento, ind, cache=cache)
        if score > melhor_score:
            melhor_score = score
            melhor_solucao = ind

        # Se encontrou solução perfeita (54/54), encerra imediatamente
        if score == 54:
            t_total = time.time() - t_inicio
            print(f"\n[SOLUÇÃO ÓTIMA ENCONTRADA NA BUSCA EXAUSTIVA!] em {t_total:.3f}s")
            print(f"Avaliadas: {i}/{total} combinações | Score Máximo: 54/54")
            print(f"Sequência: {melhor_solucao}")
            return melhor_solucao, melhor_score

    t_total = time.time() - t_inicio
    print(f"Busca exaustiva concluída em {t_total:.3f}s. Maior Score Atingido: {melhor_score}/54")
    return melhor_solucao, melhor_score


def rodar_algoritmo_genetico(
    porcentagem_mutacao=0.05,
    porcentagem_cruzamento=0.70,
    porcentagem_selecao=0.50,
    quantidade_geracoes=5000,
    quantidade_individuos_inicial=1000,
    embaralhamento=None,
    tamanho_cromossomo=20,
    intervalo_ciclo=500,
    limite_busca_exaustiva=LIMITE_BUSCA_EXAUSTIVA,
):
    """
    Executa o Algoritmo Genético de forma adaptativa e inteligente:
    - Se o espaço total de busca for menor ou igual a 'limite_busca_exaustiva' (ex: tamanhos 1, 2, 3),
      utiliza busca exaustiva direta instantânea (eliminando duplicatas e loops infinitos).
    - Para tamanhos maiores, roda o AG otimizado com Elitismo, Seleção por Torneio/Ranking e Cache de Fitness.
    """
    if embaralhamento is None:
        embaralhamento = []

    espaco_busca = calcular_espaco_busca(tamanho_cromossomo)
    cache_fitness = {}

    # Caso 1: Espaço de busca pequeno -> Busca Exaustiva Direta
    if espaco_busca <= limite_busca_exaustiva:
        return rodar_busca_exaustiva(embaralhamento, tamanho_cromossomo, cache=cache_fitness)

    # Caso 2: Espaço de busca grande -> Algoritmo Genético Otimizado
    pop_size = min(quantidade_individuos_inicial, espaco_busca)

    print(f"=== INICIANDO ALGORITMO GENÉTICO ({tamanho_cromossomo} MOVIMENTOS) ===")
    print(f"Espaço de Busca Estimado: {espaco_busca:,} indivíduos possíveis")
    print(f"Tamanho da População: {pop_size} indivíduos únicos")
    print(
        f"Taxa de Mutação: {porcentagem_mutacao * 100:.1f}% | Cruzamento:"
        f" {porcentagem_cruzamento * 100:.1f}% | Seleção:"
        f" {porcentagem_selecao * 100:.1f}%"
    )
    print(f"Máximo de Gerações: {quantidade_geracoes}")
    print("-" * 60)

    # Geração inicial sem repetição
    populacao = gerar_populacao(pop_size, tamanho_cromossomo)

    melhor_solucao = None
    melhor_score_global = -1
    geracao_resolvido = -1
    t_inicio = time.time()

    # Taxa de elitismo: mantém os melhores 5%
    qtd_elite = max(1, round(pop_size * 0.05))

    for geracao in range(1, quantidade_geracoes + 1):
        # 1. Avaliação e ordenação por fitness
        melhores, max_score, avaliados = selecionar_melhores(
            populacao, embaralhamento, porcentagem_selecao, cache=cache_fitness
        )

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = melhores[0]

        # Condição de parada imediata: Cubo 100% resolvido
        if max_score == 54:
            geracao_resolvido = geracao
            t_total = time.time() - t_inicio
            print(
                f"\n[SOLUÇÃO ENCONTRADA!] Na geração {geracao} em {t_total:.2f}s com Score Máximo (54/54)!"
            )
            break

        # 2. Elitismo: preserva os melhores indivíduos intactos
        nova_populacao = [ind for _, ind in avaliados[:qtd_elite]]

        # 3. Conjunto de pais selecionados para cruzamento
        pais_candidatos = melhores

        # 4. Reprodução (Cruzamento e Mutação) para preencher a próxima geração
        while len(nova_populacao) < pop_size:
            p1 = random.choice(pais_candidatos)
            p2 = random.choice(pais_candidatos)

            # Cruzamento
            if random.random() < porcentagem_cruzamento:
                f1, f2 = cruzar_dois_individuos(p1, p2)
            else:
                f1, f2 = list(p1), list(p2)

            # Mutação
            f1 = mutar_individuo(f1, porcentagem_mutacao)
            f2 = mutar_individuo(f2, porcentagem_mutacao)

            nova_populacao.append(f1)
            if len(nova_populacao) < pop_size:
                nova_populacao.append(f2)

        populacao = nova_populacao

        # Exibe status a cada ciclo
        if geracao % intervalo_ciclo == 0 or geracao == 1:
            print(
                f"Geração {geracao}/{quantidade_geracoes} - Melhor Score Atual:"
                f" {max_score}/54 | Melhor Global: {melhor_score_global}/54 | Cache: {len(cache_fitness):,} estados"
            )

    t_total = time.time() - t_inicio
    print("=" * 60)
    print("=== RESUMO DA EXECUÇÃO ===")
    print(f"Tamanho do Cromossomo: {tamanho_cromossomo}")
    print(f"Tempo Total: {t_total:.2f}s")
    print(f"Maior Score Atingido: {melhor_score_global}/54")
    print(f"Melhor Sequência de Movimentos: {melhor_solucao}")
    if geracao_resolvido != -1:
        print(f"Status: PROBLEMA RESOLVIDO NA GERAÇÃO {geracao_resolvido}!")
    else:
        print("Status: Limite de gerações atingido sem resolver 100% o cubo.")

    return melhor_solucao, melhor_score_global


if __name__ == "__main__":
    # Parâmetros Globais do Algoritmo Genético
    PORCENTAGEM_MUTACAO = 0.05       # 5%
    PORCENTAGEM_CRUZAMENTO = 0.70    # 70%
    PORCENTAGEM_SELECAO = 0.50       # 50%
    QUANTIDADE_GERACOES = 2000
    QUANTIDADE_INDIVIDUOS_INICIAL = 1000

    # Sequência de embaralhamento do cubo para teste
    EMBARALHAMENTO = ["U", "R", "F'"]

    # Limites para o teste incremental do cromossomo
    TAMANHO_MINIMO = 1
    TAMANHO_MAXIMO = 54

    print("=" * 60)
    print(
        f"INICIANDO TESTES INCREMENTAIS (TAMANHO {TAMANHO_MINIMO} ATÉ"
        f" {TAMANHO_MAXIMO})"
    )
    print("=" * 60)

    for tamanho in range(TAMANHO_MINIMO, TAMANHO_MAXIMO + 1):
        print("\n" + "#" * 60)
        print(f" TESTANDO TAMANHO DE CROMOSSOMO: {tamanho} MOVIMENTO(S) ")
        print("#" * 60)

        melhor_solucao, melhor_score = rodar_algoritmo_genetico(
            porcentagem_mutacao=PORCENTAGEM_MUTACAO,
            porcentagem_cruzamento=PORCENTAGEM_CRUZAMENTO,
            porcentagem_selecao=PORCENTAGEM_SELECAO,
            quantidade_geracoes=QUANTIDADE_GERACOES,
            quantidade_individuos_inicial=QUANTIDADE_INDIVIDUOS_INICIAL,
            embaralhamento=EMBARALHAMENTO,
            tamanho_cromossomo=tamanho,
            intervalo_ciclo=200,
        )

        # Se atingiu a pontuação máxima (54/54 casinhas), encerra os testes
        if melhor_score == 54:
            print("\n" + "*" * 60)
            print(
                "SUCESSO ABSOLUTO: Cubo resolvido com um cromossomo de tamanho"
                f" {tamanho}!"
            )
            print(f"Sequência Encontrada: {melhor_solucao}")
            print("*" * 60)
            break