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

# Tabela pré-computada de próximos movimentos válidos para evitar verificações repetitivas
# Chave: (face_anterior, face_retrasada) ou (None, None)
VALID_NEXT_MOVES = {}
VALID_NEXT_MOVES_SET = {}

_FACES_COM_NONE = (None, "U", "D", "F", "B", "R", "L")

for face_ant in _FACES_COM_NONE:
    for face_ret in _FACES_COM_NONE:
        validos = []
        for mov in MOVIMENTOS:
            f_mov = mov[0]
            # Regra 1: Evitar movimentos consecutivos na mesma face
            if face_ant is not None and f_mov == face_ant:
                continue
            # Regra 2: Evitar faces iguais intercaladas por face paralela oposta (ex: U D U)
            if face_ret is not None and face_ant is not None:
                if f_mov == face_ret and PARALELAS.get(f_mov) == face_ant:
                    continue
            validos.append(mov)
        chave = (face_ant, face_ret)
        VALID_NEXT_MOVES[chave] = tuple(validos)
        VALID_NEXT_MOVES_SET[chave] = set(validos)


def validar_proximo_movimento(sequencia_atual, novo_movimento):
    """
    Verifica em O(1) se a adição de 'novo_movimento' à 'sequencia_atual' respeita as regras:
    - Regra 1: Evitar movimentos consecutivos na mesma face (ex: U U, U U')
    - Regra 2: Evitar faces iguais intercaladas por face paralela oposta (ex: U D U)
    """
    face_ant = sequencia_atual[-1][0] if len(sequencia_atual) >= 1 else None
    face_ret = sequencia_atual[-2][0] if len(sequencia_atual) >= 2 else None
    return novo_movimento in VALID_NEXT_MOVES_SET[(face_ant, face_ret)]


def calcular_espaco_busca(tamanho_cromossomo):
    """
    Calcula a quantidade exata de sequências válidas para um determinado tamanho de cromossomo.
    - Tamanho 1: 18
    - Tamanho 2: 18 * 15 = 270
    - Tamanho 3: 3.888
    - Tamanho 4: 56.376
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
    Utiliza a tabela pré-calculada para geração ultrarrápida.
    """
    if tamanho_cromossomo <= 0:
        return []
    if tamanho_cromossomo == 1:
        return [[m] for m in MOVIMENTOS]

    menores = gerar_todas_combinacoes_validas(tamanho_cromossomo - 1)
    resultado = []
    for seq in menores:
        face_ant = seq[-1][0]
        face_ret = seq[-2][0] if len(seq) >= 2 else None
        for m in VALID_NEXT_MOVES[(face_ant, face_ret)]:
            resultado.append(seq + [m])
    return resultado


def gerar_individuo(qtd_cromossomos):
    """
    Gera um indivíduo aleatório respeitando as regras de não redundância em O(N) direto,
    sem loops de rejeição ou tentativas inválidas.
    """
    cromossomos = []
    face_ant = None
    face_ret = None

    for _ in range(qtd_cromossomos):
        opcoes = VALID_NEXT_MOVES[(face_ant, face_ret)]
        mov = random.choice(opcoes)
        cromossomos.append(mov)
        face_ret = face_ant
        face_ant = mov[0]

    return cromossomos


def gerar_populacao(qtd_individuos, qtd_cromossomos):
    """
    Gera uma população de indivíduos únicos e válidos.
    - Se o espaço total de busca for menor ou igual à quantidade solicitada,
      retorna diretamente todas as combinações existentes.
    - Para espaços maiores, utiliza conjunto (set) de tuplas para verificação O(1).
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