# pip install pycuber
import pycuber as pc


def calcular_score(lista_de_movimentos_estado_atual, lista_de_movimentos_proposta_solucao):
    """
    Simula um cubo mágico, aplica o embaralhamento e a solução proposta,
    e retorna a quantidade de casas (stickers) na posição correta (Máximo: 54).
    """

    # Cria um cubo no estado resolvido (montado)
    cubo = pc.Cube()

    # O pycuber aceita strings separadas por espaço (ex: "U R2 F'")
    # Transformamos as suas listas do Python nesse formato
    formula_embaralhamento = " ".join(lista_de_movimentos_estado_atual)
    formula_solucao = " ".join(lista_de_movimentos_proposta_solucao)

    # 1. Aplica os movimentos que embaralharam o cubo
    if formula_embaralhamento.strip():
        cubo(formula_embaralhamento)

    # 2. Aplica os movimentos do cromossomo (indivíduo gerado)
    if formula_solucao.strip():
        cubo(formula_solucao)

    # 3. Calcula o Score (Fitness)
    cubo_resolvido = pc.Cube()  # Cria um cubo montado para servir de gabarito
    score = 0

    # As 6 faces do cubo no pycuber: L (Left), U (Up), F (Front), D (Down), R (Right), B (Back)
    faces = ['L', 'U', 'F', 'D', 'R', 'B']

    # Percorre cada face e cada uma das 9 casinhas (3x3) daquela face
    for face in faces:
        face_atual = cubo.get_face(face)
        face_gabarito = cubo_resolvido.get_face(face)

        for linha in range(3):
            for coluna in range(3):
                # Compara a cor da casinha atual com a cor que deveria estar lá
                if face_atual[linha][coluna].colour == face_gabarito[linha][coluna].colour:
                    score += 1

    return score