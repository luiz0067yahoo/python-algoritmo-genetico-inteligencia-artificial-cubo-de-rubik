# ==============================================================================
# PONTUACAO.PY - MOTOR DE SIMULAÇÃO E AVALIAÇÃO DE FITNESS DO CUBO DE RUBIK
# ==============================================================================
# Este módulo é responsável pelo cálculo ultrarrápido da pontuação (fitness)
# de qualquer sequência de movimentos aplicada ao Cubo Mágico 3x3x3.
#
# Em vez de instanciar estruturas de objetos pesadas (como classes de cubies e faces)
# a cada avaliação, utilizamos um motor de permutação direta O(1) dos 54 adesivos
# (stickers) do cubo, atingindo centenas de milhares de avaliações por segundo.
# ==============================================================================

import pycuber as pc

# Ordem canônica das 6 faces utilizadas para indexação interna
_FACES = ('L', 'U', 'F', 'D', 'R', 'B')

# Tupla dos 18 movimentos possíveis no Cubo Mágico (Notação Padrão WCA)
# Cada face (U, D, F, B, R, L) possui 3 variações:
# - Simples (giro de 90° horário, ex: U)
# - Linha (giro de 90° anti-horário, ex: U')
# - Duplo (giro de 180°, ex: U2)
MOVIMENTOS = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
    "R", "R'", "R2",
    "L", "L'", "L2"
)

# Estado Resolvido do Cubo Mágico:
# Um cubo 3x3x3 possui 6 faces x 9 adesivos por face = 54 adesivos.
# Mapeamos cada adesivo para um número de 0 a 5 representando a cor da sua face:
# - Índices 0..8   (9 adesivos): Face Left   (L) -> Valor 0
# - Índices 9..17  (9 adesivos): Face Up     (U) -> Valor 1
# - Índices 18..26 (9 adesivos): Face Front  (F) -> Valor 2
# - Índices 27..35 (9 adesivos): Face Down   (D) -> Valor 3
# - Índices 36..44 (9 adesivos): Face Right  (R) -> Valor 4
# - Índices 45..53 (9 adesivos): Face Back   (B) -> Valor 5
ESTADO_RESOLVIDO = tuple(i // 9 for i in range(54))
SOLVED_STATE = ESTADO_RESOLVIDO

# Dicionário global de permutações pré-computadas para cada um dos 18 movimentos.
# Exemplo: MOVE_PERMUTATIONS["U"] contém uma tupla de 54 índices indicando a nova
# posição de cada adesivo após um giro na face U.
MOVE_PERMUTATIONS = {}


def _inicializar_tabela_permutacoes():
    """
    Inicializa a tabela de permutações uma única vez no momento da importação.
    Utiliza o PyCuber para calcular com precisão matemática o vetor de deslocamento
    de cada um dos 54 adesivos para todos os 18 movimentos oficiais.
    """
    for m in MOVIMENTOS:
        cube = pc.Cube()
        idx = 0
        # Marca cada adesivo do cubo com seu identificador original único (0 a 53)
        for face in _FACES:
            f_grid = cube.get_face(face)
            for r in range(3):
                for c in range(3):
                    f_grid[r][c].orig_id = idx
                    idx += 1

        # Aplica o movimento no PyCuber
        cube(m)

        # Lê a nova disposição dos adesivos após o giro
        perm = []
        for face in _FACES:
            f_grid = cube.get_face(face)
            for r in range(3):
                for c in range(3):
                    perm.append(f_grid[r][c].orig_id)
        MOVE_PERMUTATIONS[m] = tuple(perm)


# Executa a inicialização da tabela no momento da carga do módulo
_inicializar_tabela_permutacoes()


def aplicar_movimento(estado, movimento):
    """
    Aplica um único movimento (ex: 'U', 'R'', 'F2') a um estado de 54 adesivos em O(1).

    Parâmetros:
        estado (tuple): Tupla de 54 inteiros representando as cores atuais das faces.
        movimento (str): Código do movimento a ser executado.

    Retorno:
        tuple: Novo estado resultante após a permutação dos adesivos.
    """
    perm = MOVE_PERMUTATIONS[movimento]
    return tuple(estado[p] for p in perm)


def aplicar_movimentos(estado, lista_movimentos):
    """
    Aplica uma sequência encadeada de movimentos a um estado de 54 adesivos.

    Parâmetros:
        estado (tuple): Estado inicial de 54 adesivos.
        lista_movimentos (list | tuple | str): Sequência de movimentos (ex: ['R', 'U', "R'"]).

    Retorno:
        tuple: Estado final após a execução de todos os giros.
    """
    if isinstance(lista_movimentos, str):
        lista_movimentos = [m for m in lista_movimentos.split() if m.strip()]

    for mov in lista_movimentos:
        perm = MOVE_PERMUTATIONS.get(mov)
        if perm:
            estado = tuple(estado[p] for p in perm)
    return estado


def calcular_score_estado(estado):
    """
    Calcula a função de Fitness (aptidão) de um estado comparando-o com o cubo resolvido.

    Parâmetros:
        estado (tuple): Tupla de 54 inteiros representando as cores atuais do cubo.

    Retorno:
        int: Quantidade de adesivos que estão na cor e posição correta (de 0 a 54).
             Um score de 54 indica que o Cubo Mágico está 100% resolvido.
    """
    return sum(1 for i in range(54) if estado[i] == ESTADO_RESOLVIDO[i])


def calcular_score(lista_de_movimentos_estado_atual, lista_de_movimentos_proposta_solucao, cache=None):
    """
    Avalia a qualidade de uma proposta de solução em relação a um estado inicial embaralhado.

    Funcionamento:
    1. Parte do estado resolvido canônico.
    2. Aplica os movimentos de embaralhamento (se houver).
    3. Aplica a sequência de movimentos proposta como solução.
    4. Conta quantos adesivos (stickers) terminaram na face correta (máx 54).

    Parâmetros:
        lista_de_movimentos_estado_atual (list | str): Sequência que embaralhou o cubo.
        lista_de_movimentos_proposta_solucao (list | str): Sequência de resolução candidata.
        cache (dict, opcional): Dicionário para memoização de scores já avaliados.

    Retorno:
        int: Pontuação de 0 a 54 casinhas corretas.
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

    # Avalia a quantidade de adesivos corretos
    score = calcular_score_estado(estado)

    # Armazena no cache para consultas futuras
    if cache is not None:
        cache[chave] = score

    return score