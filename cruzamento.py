import random
from populacao import MOVIMENTOS, PARALELAS, VALID_NEXT_MOVES, VALID_NEXT_MOVES_SET


def obter_face(movimento):
    """Extrai a letra que representa a face do Cubo Mágico a partir de um movimento."""
    return movimento[0]


def reparar_individuo(individuo):
    """
    Garante que uma sequência de movimentos respeite rigorosamente as regras
    de não redundância do Cubo Mágico em uma única passada O(N), substituindo
    movimentos inválidos através da tabela de transição O(1).
    """
    reparado = []
    face_ant = None
    face_ret = None

    for mov in individuo:
        chave = (face_ant, face_ret)
        if mov in VALID_NEXT_MOVES_SET[chave]:
            reparado.append(mov)
            face_ret = face_ant
            face_ant = mov[0]
        else:
            opcoes = VALID_NEXT_MOVES[chave]
            novo_mov = random.choice(opcoes) if opcoes else mov
            reparado.append(novo_mov)
            face_ret = face_ant
            face_ant = novo_mov[0]

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
