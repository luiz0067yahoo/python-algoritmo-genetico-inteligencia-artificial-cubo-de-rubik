# pip install pycuber
import pycuber as pc

# Gabarito de cores pré-computado a partir de um cubo resolvido
_FACES = ('L', 'U', 'F', 'D', 'R', 'B')
_CUBO_GABARITO = pc.Cube()
_CORES_GABARITO = {
    face: [[_CUBO_GABARITO.get_face(face)[i][j].colour for j in range(3)] for i in range(3)]
    for face in _FACES
}


def calcular_score(lista_de_movimentos_estado_atual, lista_de_movimentos_proposta_solucao, cache=None):
    """
    Simula um cubo mágico, aplica o embaralhamento e a solução proposta,
    e retorna a quantidade de casas (stickers) na posição correta (Máximo: 54).

    Suporta um dicionário de cache opcional para evitar recalcular o fitness
    de indivíduos idênticos já avaliados anteriormente.
    """
    if cache is not None:
        chave = tuple(lista_de_movimentos_proposta_solucao)
        if chave in cache:
            return cache[chave]

    # Cria um novo cubo
    cubo = pc.Cube()

    # Aplica o embaralhamento
    if lista_de_movimentos_estado_atual:
        if isinstance(lista_de_movimentos_estado_atual, str):
            cubo(lista_de_movimentos_estado_atual)
        else:
            formula_embaralhamento = " ".join(lista_de_movimentos_estado_atual)
            if formula_embaralhamento.strip():
                cubo(formula_embaralhamento)

    # Aplica a solução proposta
    if lista_de_movimentos_proposta_solucao:
        if isinstance(lista_de_movimentos_proposta_solucao, str):
            cubo(lista_de_movimentos_proposta_solucao)
        else:
            formula_solucao = " ".join(lista_de_movimentos_proposta_solucao)
            if formula_solucao.strip():
                cubo(formula_solucao)

    # Contagem de casinhas corretas comparando com o gabarito pré-computado
    score = 0
    for face in _FACES:
        face_atual = cubo.get_face(face)
        gab_face = _CORES_GABARITO[face]
        for linha in range(3):
            for coluna in range(3):
                if face_atual[linha][coluna].colour == gab_face[linha][coluna]:
                    score += 1

    if cache is not None:
        cache[chave] = score

    return score