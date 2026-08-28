import copy
import random
from cruzamento import cruzamento
from mutacao import mutacao
from pontuacao import calcular_score
from populacao import MOVIMENTOS, PARALELAS, gerar_populacao


def selecionar_melhores(populacao, embaralhamento, porcentagem_selecao):
    """Avalia a população calculando o fitness de cada indivíduo e

    retorna os melhores indivíduos com base no percentual de seleção.
    """
    # Avalia cada indivíduo: tupla (score, individuo)
    avaliados = []
    for ind in populacao:
        score = calcular_score(embaralhamento, ind)
        avaliados.append((score, ind))

    # Ordena do maior fitness para o menor
    avaliados.sort(key=lambda x: x[0], reverse=True)

    # Determina a quantidade a manter
    qtd_selecionada = max(1, round(len(populacao) * porcentagem_selecao))
    melhores = [ind for _, ind in avaliados[:qtd_selecionada]]

    best_score = avaliados[0][0]
    return melhores, best_score


def rodar_algoritmo_genetico(
    porcentagem_mutacao,
    porcentagem_cruzamento,
    porcentagem_selecao,
    quantidade_geracoes,
    quantidade_individuos_inicial,
    embaralhamento,
    tamanho_cromossomo=20,
    intervalo_ciclo=1000,
):
    """Executa o Algoritmo Genético até resolver o cubo mágico (score == 54) ou

    atingir a quantidade máxima de gerações.
    """
    print(
        f"=== INICIANDO EXECUÇÃO (CROMOSSOMO: {tamanho_cromossomo} MOVIMENTOS) ==="
    )
    print(f"População Inicial: {quantidade_individuos_inicial}")
    print(
        f"Mutação: {porcentagem_mutacao * 100}% | Cruzamento:"
        f" {porcentagem_cruzamento * 100}% | Seleção:"
        f" {porcentagem_selecao * 100}%"
    )
    print(f"Máximo de Gerações: {quantidade_geracoes}")
    print("-" * 55)

    # 1. População inicial
    populacao = gerar_populacao(
        quantidade_individuos_inicial, tamanho_cromossomo
    )

    melhor_solucao = None
    melhor_score_global = -1
    geracao_resolvido = -1

    for geracao in range(1, quantidade_geracoes + 1):
        # Passo A: Seleção inicial da geração
        populacao, max_score = selecionar_melhores(
            populacao, embaralhamento, porcentagem_selecao
        )

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = populacao[0]

        # Condição de parada: Cubo resolvido (54 casinhas corretas)
        if max_score == 54:
            geracao_resolvido = geracao
            print(
                f"\n[SOLUÇÃO ENCONTRADA!] Na geração {geracao} com Score Máximo"
                " (54/54)!"
            )
            break

        # Passo B: Mutação e Seleção
        populacao_mutada = mutacao(
            copy.deepcopy(populacao), porcentagem_mutacao
        )
        populacao, max_score = selecionar_melhores(
            populacao_mutada, embaralhamento, porcentagem_selecao
        )

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = populacao[0]

        if max_score == 54:
            geracao_resolvido = geracao
            print(
                "\n[SOLUÇÃO ENCONTRADA NA MUTAÇÃO!] Na geração"
                f" {geracao} com Score Máximo (54/54)!"
            )
            break

        # Passo C: Cruzamento e Seleção
        populacao_cruzada = cruzamento(
            copy.deepcopy(populacao), porcentagem_cruzamento
        )
        populacao, max_score = selecionar_melhores(
            populacao_cruzada, embaralhamento, porcentagem_selecao
        )

        if max_score > melhor_score_global:
            melhor_score_global = max_score
            melhor_solucao = populacao[0]

        if max_score == 54:
            geracao_resolvido = geracao
            print(
                "\n[SOLUÇÃO ENCONTRADA NO CRUZAMENTO!] Na geração"
                f" {geracao} com Score Máximo (54/54)!"
            )
            break

        # Passo D: Completar a população até a quantidade inicial
        faltantes = quantidade_individuos_inicial - len(populacao)
        if faltantes > 0:
            novos_individuos = gerar_populacao(faltantes, tamanho_cromossomo)
            populacao.extend(novos_individuos)

        # Exibe status a cada ciclo (ex: a cada 1000 gerações)
        if geracao % intervalo_ciclo == 0 or geracao == 1:
            print(
                f"Geração {geracao}/{quantidade_geracoes} - Melhor Score Atual:"
                f" {max_score}/54 | Melhor Global: {melhor_score_global}/54"
            )

    print("=" * 55)
    print("=== RESUMO DA EXECUÇÃO ===")
    print(f"Tamanho do Cromossomo: {tamanho_cromossomo}")
    print(f"Maior Score Atingido: {melhor_score_global}/54")
    print(f"Melhor Sequência de Movimentos: {melhor_solucao}")
    if geracao_resolvido != -1:
        print(
            f"Status: PROBLEMA RESOLVIDO NA GERAÇÃO {geracao_resolvido}!"
        )
    else:
        print(
            "Status: Limite de gerações atingido sem resolver 100% o cubo."
        )

    return melhor_solucao, melhor_score_global


if __name__ == "__main__":
    # Parâmetros Globais do Algoritmo Genético
    PORCENTAGEM_MUTACAO = 0.04  # 4%
    PORCENTAGEM_CRUZAMENTO = 0.04  # 4%
    PORCENTAGEM_SELECAO = 0.50  # 50%
    QUANTIDADE_GERACOES = 5000
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
            intervalo_ciclo=1000,
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