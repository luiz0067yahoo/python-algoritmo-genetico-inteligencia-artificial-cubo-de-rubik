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


def validar_proximo_movimento(sequencia_atual, novo_movimento):
    """
    Verifica se a adição de 'novo_movimento' à 'sequencia_atual' respeita as regras:
    - Regra 1: Evitar movimentos consecutivos na mesma face (ex: U U, U U')
    - Regra 2: Evitar faces iguais intercaladas por face paralela oposta (ex: U D U)
    """
    face_atual = novo_movimento[0]

    # Regra 1: Evitar a mesma face consecutiva
    if len(sequencia_atual) >= 1:
        if face_atual == sequencia_atual[-1][0]:
            return False

    # Regra 2: Evitar faces iguais separadas por uma paralela
    if len(sequencia_atual) >= 2:
        face_anterior = sequencia_atual[-1][0]
        face_retrasada = sequencia_atual[-2][0]
        if face_atual == face_retrasada and PARALELAS.get(face_atual) == face_anterior:
            return False

    return True


def calcular_espaco_busca(tamanho_cromossomo):
    """
    Calcula a quantidade exata de sequências válidas para um determinado tamanho de cromossomo.
    - Tamanho 1: 18
    - Tamanho 2: 18 * 15 = 270
    - Tamanho 3: 3.888
    - Tamanho 4: 56.376
    ...
    """
    if tamanho_cromossomo <= 0:
        return 0
    if tamanho_cromossomo == 1:
        return 18
    if tamanho_cromossomo == 2:
        return 18 * 15

    faces = ("U", "D", "F", "B", "R", "L")
    estado_anterior = {}
    for f in faces:
        for p in faces:
            if f != p:
                estado_anterior[(f, p)] = 3 * 3

    for _ in range(3, tamanho_cromossomo + 1):
        novo_estado = {}
        for f in faces:
            for p in faces:
                if f == p:
                    continue
                total = 0
                for pp in faces:
                    if (p, pp) in estado_anterior:
                        if PARALELAS[pp] == p and f == pp:
                            continue
                        total += estado_anterior[(p, pp)]
                if total > 0:
                    novo_estado[(f, p)] = total * 3
        estado_anterior = novo_estado

    return sum(estado_anterior.values())


def gerar_todas_combinacoes_validas(tamanho_cromossomo):
    """
    Gera deterministicamente todas as sequências válidas de movimentos para um dado tamanho.
    Ideal para tamanhos pequenos (1, 2, 3) onde a busca exaustiva é instantânea.
    """
    if tamanho_cromossomo <= 0:
        return []
    if tamanho_cromossomo == 1:
        return [[m] for m in MOVIMENTOS]

    menores = gerar_todas_combinacoes_validas(tamanho_cromossomo - 1)
    resultado = []
    for seq in menores:
        for m in MOVIMENTOS:
            if validar_proximo_movimento(seq, m):
                resultado.append(seq + [m])
    return resultado


def gerar_individuo(qtd_cromossomos):
    """
    Gera um indivíduo aleatório respeitando as regras de não redundância.
    """
    cromossomos = []
    for _ in range(qtd_cromossomos):
        while True:
            cromossomo = random.choice(MOVIMENTOS)
            if validar_proximo_movimento(cromossomos, cromossomo):
                cromossomos.append(cromossomo)
                break
    return cromossomos


def gerar_populacao(qtd_individuos, qtd_cromossomos):
    """
    Gera uma população de indivíduos únicos e válidos.
    - Se o espaço total de busca for menor ou igual à quantidade solicitada,
      retorna diretamente todas as combinações existentes (evita loop infinito e redundância).
    - Para espaços maiores, utiliza conjunto (set) para verificação O(1) com limite de tentativas.
    """
    if qtd_cromossomos <= 0:
        return []

    espaco_busca = calcular_espaco_busca(qtd_cromossomos)

    # Se a quantidade pedida cobrir todo o espaço de busca, retorna todas as combinações
    if espaco_busca <= qtd_individuos:
        return gerar_todas_combinacoes_validas(qtd_cromossomos)

    populacao_set = set()
    populacao = []
    tentativas = 0
    max_tentativas = qtd_individuos * 50

    while len(populacao) < qtd_individuos and tentativas < max_tentativas:
        novo_individuo = gerar_individuo(qtd_cromossomos)
        chave = tuple(novo_individuo)

        if chave not in populacao_set:
            populacao_set.add(chave)
            populacao.append(novo_individuo)

        tentativas += 1

    return populacao