# ==============================================================================
# POPULACAO.PY - GERAÇÃO DE INDIVÍDUOS, POPULAÇÕES E TRANSIÇÕES VÁLIDAS (WCA)
# ==============================================================================
# Este módulo gerencia a criação de cromossomos (sequências de movimentos)
# para o Algoritmo Genético, garantindo que não existam redundâncias ou
# cancelamentos algébricos óbvios no cubo mágico.
#
# Regras de Não Redundância:
# - Regra 1: Duas rotações consecutivas na mesma face são inválidas (ex: U U ou U U').
# - Regra 2: Três rotações onde a 1ª e a 3ª pertencem à mesma face e a 2ª pertence
#            à face paralela oposta são inválidas (ex: U D U ou R L R').
# ==============================================================================

import random

# Conjunto completo dos 18 movimentos canônicos do Cubo de Rubik
MOVIMENTOS = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
    "R", "R'", "R2",
    "L", "L'", "L2"
)

# Mapeamento de pares de faces paralelas e opostas no Cubo Mágico:
# - U (Up/Topo) é oposto a D (Down/Base)
# - F (Front/Frente) é oposto a B (Back/Atrás)
# - R (Right/Direita) é oposto a L (Left/Esquerda)
PARALELAS = {
    "U": "D",
    "D": "U",
    "F": "B",
    "B": "F",
    "R": "L",
    "L": "R"
}

# ------------------------------------------------------------------------------
# TABELA PRÉ-COMPUTADA DE TRANSIÇÕES VÁLIDAS O(1)
# ------------------------------------------------------------------------------
# Chave: (face_anterior, face_retrasada) -> Tupla de movimentos permitidos
# Isso elimina o custo computacional de validações repetidas durante o AG.
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
    Verifica em tempo constante O(1) se a adição de 'novo_movimento' à sequência
    respeita as regras de não redundância do Cubo Mágico.

    Parâmetros:
        sequencia_atual (list): Lista de movimentos já efetuados (ex: ['R', 'U']).
        novo_movimento (str): Próximo movimento candidato (ex: 'R'').

    Retorno:
        bool: True se o movimento for válido e não redundante, False caso contrário.
    """
    face_ant = sequencia_atual[-1][0] if len(sequencia_atual) >= 1 else None
    face_ret = sequencia_atual[-2][0] if len(sequencia_atual) >= 2 else None
    return novo_movimento in VALID_NEXT_MOVES_SET[(face_ant, face_ret)]


def calcular_espaco_busca(tamanho_cromossomo):
    """
    Calcula a quantidade matemática exata de combinações válidas possíveis
    para um dado tamanho de cromossomo (comprimento da sequência de movimentos).

    Valores conhecidos:
    - Tamanho 1: 18 movimentos
    - Tamanho 2: 18 * 15 = 270 movimentos
    - Tamanho 3: 3.888 movimentos
    - Tamanho 4: 56.376 movimentos
    ...

    Parâmetros:
        tamanho_cromossomo (int): Quantidade de movimentos na sequência.

    Retorno:
        int: Número total exato de combinações sem redundâncias.
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
    Gera de forma determinística todas as sequências válidas de movimentos para
    um determinado tamanho. Utilizado na Busca Exaustiva para comprimentos pequenos (1, 2, 3).

    Parâmetros:
        tamanho_cromossomo (int): Comprimento das sequências a serem geradas.

    Retorno:
        list[list[str]]: Lista contendo todas as sequências possíveis de movimentos.
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
    Gera um único indivíduo (cromossomo) composto por 'qtd_cromossomos' genes (movimentos).
    Executa em O(N) direto selecionando apenas movimentos válidos da tabela pré-computada,
    com garantia de 0 rejeições e máxima performance.

    Parâmetros:
        qtd_cromossomos (int): Quantidade de movimentos que o indivíduo terá.

    Retorno:
        list[str]: Sequência de movimentos representando o cromossomo gerado.
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
    Gera uma população inicial de indivíduos únicos e válidos para o Algoritmo Genético.

    Comportamento:
    - Se a quantidade solicitada cobrir todo o espaço de busca, retorna diretamente
      todas as combinações determinísticas existentes.
    - Caso contrário, gera indivíduos aleatórios e garante unicidade utilizando um set em O(1).

    Parâmetros:
        qtd_individuos (int): Número de indivíduos a serem criados na população.
        qtd_cromossomos (int): Tamanho de cada cromossomo (número de movimentos).

    Retorno:
        list[list[str]]: População de indivíduos.
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


def gerar_embaralhamento_wca(tamanho=25):
    """
    Gera uma sequência de embaralhamento oficial conforme o regulamento da
    World Cube Association (WCA - Artigo 12 / Regulação 4b).

    Por padrão, gera 25 movimentos aleatórios válidos sem redundâncias ou cancelamentos.

    Parâmetros:
        tamanho (int): Quantidade de movimentos do embaralhamento (padrão: 25).

    Retorno:
        list[str]: Lista com os movimentos do embaralhamento oficial WCA.
    """
    return gerar_individuo(tamanho)