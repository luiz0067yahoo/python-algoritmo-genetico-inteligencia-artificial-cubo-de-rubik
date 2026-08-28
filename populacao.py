import random

MOVIMENTOS = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
    "R", "R'", "R2",
    "L", "L'", "L2"
)

PARALELAS = {
    "U": "D",
    "D": "U",
    "F": "B",
    "B": "F",
    "R": "L",
    "L": "R"
}


def gerar_individuo(qtd_cromossomos):
    cromossomos = []
    for _ in range(qtd_cromossomos):
        while True:
            cromossomo = random.choice(MOVIMENTOS)
            face_atual = cromossomo[0]

            # Regra 1: Evitar a mesma face consecutiva
            if len(cromossomos) >= 1:
                if face_atual == cromossomos[-1][0]:
                    continue

            # Regra 2: Evitar faces iguais separadas por uma paralela
            if len(cromossomos) >= 2:
                if face_atual == cromossomos[-2][0] and PARALELAS[face_atual] == cromossomos[-1][0]:
                    continue

            break
        cromossomos.append(cromossomo)
    return cromossomos


def gerar_populacao(qtd_individuos, qtd_cromossomos):
    """
    Gera uma população de indivíduos únicos.
    """
    populacao = []

    # Continua executando até a lista atingir a quantidade desejada
    while len(populacao) < qtd_individuos:
        novo_individuo = gerar_individuo(qtd_cromossomos)

        # Verifica se o indivíduo gerado já existe na população
        if novo_individuo not in populacao:
            populacao.append(novo_individuo)

    return populacao