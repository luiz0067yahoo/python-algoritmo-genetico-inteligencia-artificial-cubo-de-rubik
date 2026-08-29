import random
from populacao import MOVIMENTOS, PARALELAS


def obter_face(movimento):
    """Extrai a letra que representa a face do Cubo Mágico a partir de um movimento."""
    return movimento[0]


def movimento_valido_no_indice(individuo, indice, novo_movimento):
    """
    Verifica se substituir o gene em 'indice' por 'novo_movimento' mantém
    o indivíduo válido segundo as regras de não redundância.
    """
    face = obter_face(novo_movimento)

    # 1. Checagem com o elemento anterior (i - 1)
    if indice > 0:
        if face == obter_face(individuo[indice - 1]):
            return False

    # 2. Checagem com o elemento retrasado (i - 2)
    if indice > 1:
        face_ant = obter_face(individuo[indice - 1])
        face_ret = obter_face(individuo[indice - 2])
        if face == face_ret and PARALELAS.get(face) == face_ant:
            return False

    # 3. Checagem com o elemento posterior (i + 1)
    if indice < len(individuo) - 1:
        if face == obter_face(individuo[indice + 1]):
            return False

    # 4. Checagem com o elemento pós-posterior (i + 2)
    if indice < len(individuo) - 2:
        face_pos = obter_face(individuo[indice + 1])
        face_pos_pos = obter_face(individuo[indice + 2])
        if face == face_pos_pos and PARALELAS.get(face) == face_pos:
            return False

    return True


def mutar_individuo(individuo, porcentagem_mutacao):
    """
    Aplica mutação em um indivíduo com probabilidade 'porcentagem_mutacao' por gene.
    """
    individuo_mutado = list(individuo)
    for i in range(len(individuo_mutado)):
        if random.random() < porcentagem_mutacao:
            candidatos = [
                m for m in MOVIMENTOS
                if m != individuo_mutado[i] and movimento_valido_no_indice(individuo_mutado, i, m)
            ]
            if candidatos:
                individuo_mutado[i] = random.choice(candidatos)
    return individuo_mutado


def mutacao(populacao, porcentagem_mutacao):
    """
    Aplica o operador de mutação em toda a população, retornando uma nova lista de indivíduos.
    """
    return [mutar_individuo(ind, porcentagem_mutacao) for ind in populacao]
