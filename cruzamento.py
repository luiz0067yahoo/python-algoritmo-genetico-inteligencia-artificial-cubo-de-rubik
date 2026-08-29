import random
from populacao import MOVIMENTOS, PARALELAS, validar_proximo_movimento


def obter_face(movimento):
    """Extrai a letra que representa a face do Cubo Mágico a partir de um movimento."""
    return movimento[0]


def reparar_individuo(individuo):
    """
    Garante que uma sequência de movimentos respeite rigorosamente as regras
    de não redundância do Cubo Mágico, substituindo movimentos inválidos.
    """
    reparado = []
    for mov in individuo:
        if validar_proximo_movimento(reparado, mov):
            reparado.append(mov)
        else:
            candidatos = [m for m in MOVIMENTOS if validar_proximo_movimento(reparado, m)]
            if candidatos:
                reparado.append(random.choice(candidatos))
            else:
                reparado.append(mov)
    return reparado


def cruzar_dois_individuos(pai1, pai2):
    """
    Executa o cruzamento de um ponto entre dois indivíduos e repara os filhos gerados.
    """
    tamanho = len(pai1)
    if tamanho <= 1:
        return list(pai1), list(pai2)

    ponto_corte = random.randint(1, tamanho - 1)
    filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
    filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]

    return reparar_individuo(filho1), reparar_individuo(filho2)


def cruzamento(populacao, porcentagem_cruzamento):
    """
    Aplica o operador de cruzamento em pares da população com base na taxa estipulada.
    """
    populacao_cruzada = [list(ind) for ind in populacao]
    n = len(populacao_cruzada)

    if n < 2 or len(populacao_cruzada[0]) <= 1:
        return populacao_cruzada

    # Processa os indivíduos em pares
    for i in range(0, n - 1, 2):
        if random.random() < porcentagem_cruzamento:
            f1, f2 = cruzar_dois_individuos(populacao_cruzada[i], populacao_cruzada[i + 1])
            populacao_cruzada[i] = f1
            populacao_cruzada[i + 1] = f2

    return populacao_cruzada
