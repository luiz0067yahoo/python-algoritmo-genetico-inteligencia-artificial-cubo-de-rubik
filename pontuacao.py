import pycuber as pc

# Definição das faces do cubo e gabarito
_FACES = ('L', 'U', 'F', 'D', 'R', 'B')

MOVIMENTOS = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
    "R", "R'", "R2",
    "L", "L'", "L2"
)

# Estado resolvido: 6 faces x 9 adesivos = 54 adesivos (0=L, 1=U, 2=F, 3=D, 4=R, 5=B)
ESTADO_RESOLVIDO = tuple(i // 9 for i in range(54))
SOLVED_STATE = ESTADO_RESOLVIDO

# Mapeamento pré-computado de permutações para os 18 movimentos canônicos
# Cada movimento é representado como uma tupla de 54 índices.
MOVE_PERMUTATIONS = {}


def _inicializar_tabela_permutacoes():
    """Pré-computa as permutações dos 54 adesivos para todos os 18 movimentos possíveis."""
    for m in MOVIMENTOS:
        cube = pc.Cube()
        idx = 0
        for face in _FACES:
            f_grid = cube.get_face(face)
            for r in range(3):
                for c in range(3):
                    f_grid[r][c].orig_id = idx
                    idx += 1
        cube(m)
        perm = []
        for face in _FACES:
            f_grid = cube.get_face(face)
            for r in range(3):
                for c in range(3):
                    perm.append(f_grid[r][c].orig_id)
        MOVE_PERMUTATIONS[m] = tuple(perm)


_inicializar_tabela_permutacoes()


def aplicar_movimento(estado, movimento):
    """Aplica um único movimento (ex: 'U', 'R'') ao estado de 54 adesivos em O(1)."""
    perm = MOVE_PERMUTATIONS[movimento]
    return tuple(estado[p] for p in perm)


def aplicar_movimentos(estado, lista_movimentos):
    """Aplica uma sequência de movimentos ao estado de 54 adesivos em tempo ultrarrápido."""
    if isinstance(lista_movimentos, str):
        lista_movimentos = [m for m in lista_movimentos.split() if m.strip()]

    for mov in lista_movimentos:
        perm = MOVE_PERMUTATIONS.get(mov)
        if perm:
            estado = tuple(estado[p] for p in perm)
    return estado


def calcular_score_estado(estado):
    """
    Retorna a quantidade de adesivos na posição correta (0 a 54)
    comparando o estado atual com o estado resolvido.
    """
    return sum(1 for i in range(54) if estado[i] == ESTADO_RESOLVIDO[i])


def calcular_score(lista_de_movimentos_estado_atual, lista_de_movimentos_proposta_solucao, cache=None):
    """
    Simula o cubo mágico com permutação direta de adesivos em memória e
    retorna a quantidade de casas na posição correta (Máximo: 54).

    Totalmente compatível com a interface anterior, suportando cache opcional.
    """
    if cache is not None:
        chave = tuple(lista_de_movimentos_proposta_solucao) if not isinstance(lista_de_movimentos_proposta_solucao, str) else lista_de_movimentos_proposta_solucao
        if chave in cache:
            return cache[chave]

    # Inicia do estado resolvido
    estado = ESTADO_RESOLVIDO

    # Aplica o embaralhamento
    if lista_de_movimentos_estado_atual:
        estado = aplicar_movimentos(estado, lista_de_movimentos_estado_atual)

    # Aplica a proposta de solução
    if lista_de_movimentos_proposta_solucao:
        estado = aplicar_movimentos(estado, lista_de_movimentos_proposta_solucao)

    score = calcular_score_estado(estado)

    if cache is not None:
        cache[chave] = score

    return score