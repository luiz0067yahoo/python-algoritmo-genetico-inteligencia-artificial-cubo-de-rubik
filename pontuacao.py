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


# ==============================================================================
# ESTRUTURAS DE PEÇAS (CUBIES) E PRÉ-COMPUTAÇÃO DE PERFORMANCE
# ==============================================================================
# Mapeamento dos 20 cubies móveis (8 cantos e 12 meios) com suas coordenadas 3D (x, y, z)
# e índices de adesivos para cálculo de distância Manhattan, orientação e blocos F2L.
# Cores de referência: L=0, U=1, F=2, D=3, R=4, B=5
CUBIE_SLOTS = {
    # 8 Cantos (Corners): ((x,y,z), ((face, index, target_color), ...))
    'ULF': ((-1,  1,  1), (('L', 2, 0), ('U', 15, 1), ('F', 18, 2))),
    'ULB': ((-1,  1, -1), (('L', 0, 0), ('U',  9, 1), ('B', 47, 5))),
    'URF': (( 1,  1,  1), (('R', 36, 4), ('U', 17, 1), ('F', 20, 2))),
    'URB': (( 1,  1, -1), (('R', 38, 4), ('U', 11, 1), ('B', 45, 5))),
    'DLF': ((-1, -1,  1), (('L', 8, 0), ('D', 27, 3), ('F', 24, 2))),
    'DLB': ((-1, -1, -1), (('L', 6, 0), ('D', 33, 3), ('B', 53, 5))),
    'DRF': (( 1, -1,  1), (('R', 42, 4), ('D', 29, 3), ('F', 26, 2))),
    'DRB': (( 1, -1, -1), (('R', 44, 4), ('D', 35, 3), ('B', 51, 5))),
    # 12 Meios (Edges):
    'UL': ((-1,  1,  0), (('L', 1, 0), ('U', 12, 1))),
    'UB': (( 0,  1, -1), (('U', 10, 1), ('B', 46, 5))),
    'UR': (( 1,  1,  0), (('U', 14, 1), ('R', 37, 4))),
    'UF': (( 0,  1,  1), (('U', 16, 1), ('F', 19, 2))),
    'FL': ((-1,  0,  1), (('L', 5, 0), ('F', 21, 2))),
    'FR': (( 1,  0,  1), (('F', 23, 2), ('R', 39, 4))),
    'BL': ((-1,  0, -1), (('L', 3, 0), ('B', 50, 5))),
    'BR': (( 1,  0, -1), (('R', 41, 4), ('B', 48, 5))),
    'DL': ((-1, -1,  0), (('L', 7, 0), ('D', 30, 3))),
    'DB': (( 0, -1, -1), (('D', 34, 3), ('B', 52, 5))),
    'DR': (( 1, -1,  0), (('D', 32, 3), ('R', 43, 4))),
    'DF': (( 0, -1,  1), (('D', 28, 3), ('F', 25, 2))),
}

CORNER_HOME = {}
EDGE_HOME = {}
for name, (coords, stickers) in CUBIE_SLOTS.items():
    colors = frozenset(t[2] for t in stickers)
    if len(stickers) == 3:
        CORNER_HOME[colors] = (name, coords, stickers)
    else:
        EDGE_HOME[colors] = (name, coords, stickers)

CORNER_DATA = {}
for _c_name in ('ULF', 'ULB', 'URF', 'URB', 'DLF', 'DLB', 'DRF', 'DRB'):
    _coords, _stks = CUBIE_SLOTS[_c_name]
    _indices = tuple(s[1] for s in _stks)
    _targets = tuple(s[2] for s in _stks)
    _target_set = frozenset(_targets)
    CORNER_DATA[_c_name] = (_coords, _indices, _targets, _target_set)

EDGE_DATA = {}
for _e_name in ('UL', 'UB', 'UR', 'UF', 'FL', 'FR', 'BL', 'BR', 'DL', 'DB', 'DR', 'DF'):
    _coords, _stks = CUBIE_SLOTS[_e_name]
    _indices = tuple(s[1] for s in _stks)
    _targets = tuple(s[2] for s in _stks)
    _target_set = frozenset(_targets)
    EDGE_DATA[_e_name] = (_coords, _indices, _targets, _target_set)

F2L_PAIRS = (
    ('DLF', 'FL'),
    ('DRF', 'FR'),
    ('DLB', 'BL'),
    ('DRB', 'BR'),
)

SCORE_RESOLVIDO = 2110.0
SCORE_MAXIMO = 2110.0


def contar_adesivos_corretos(estado):
    """Conta a quantidade bruta de adesivos que estão na cor e face correta (0 a 54)."""
    return sum(1 for i in range(54) if estado[i] == ESTADO_RESOLVIDO[i])


def cubo_esta_resolvido(estado):
    """Verifica se o cubo está 100% resolvido."""
    return estado == ESTADO_RESOLVIDO or contar_adesivos_corretos(estado) == 54


def calcular_score_estado(estado, qtd_movimentos=0, lambda_mov=0.5, retornar_detalhes=False):
    """
    Calcula a função de Fitness / Score do Cubo de Rubik avaliando os 6 componentes essenciais,
    estruturados segundo a hierarquia do MÉTODO DE JESSICA FRIDRICH (CFOP):

    Mapeamento direto com os 4 Estágios do Método Fridrich (CFOP):
    ----------------------------------------------------------------
    • Estágio 1 (Cross / Cruz):
        - Arestas da base D alinhadas aos centros correspondentes (+20 pts/aresta + 50 pts bônus cruz = máx 130 pts).
    • Estágio 2 (F2L / First Two Layers):
        - 4 pares formados por canto inferior e aresta intermediária (+50 pts por par acoplado = máx 200 pts).
    • Estágio 3 (OLL / Orientation of Last Layer):
        - Orientação correta dos 8 cantos (+25 pts/canto = máx 200 pts) e 12 arestas (+20 pts/aresta = máx 240 pts),
          recompensando o alinhamento de todos os adesivos amarelos no topo sem exigir permutação.
    • Estágio 4 (PLL / Permutation of Last Layer):
        - Posição final dos 8 cantos (+25 pts/canto = máx 200 pts) e 12 arestas (+20 pts/aresta = máx 240 pts)
          somadas aos 54 adesivos (+108 pts) e bônus de solução perfeita (+600 pts) atingindo 2110.0 pts.

    Componentes da Decomposição do Score:
    1. + posição correta dos cantos: 8 peças nos slots corretos (+25 pts cada, máx 200).
    2. + orientação correta dos cantos: adesivos alinhados à face de referência (+25 pts cada, máx 200).
    3. + posição correta das arestas: 12 peças nos slots corretos (+20 pts cada, máx 240).
    4. + orientação correta das arestas: arestas sem inversão/flip (+20 pts cada, máx 240).
    5. + pares de peças corretos: 4 pares F2L (+50 pts cada) + Cruz da Base (+80 pts + 50 bônus) = máx 330 pts.
    6. - penalidade pelo tamanho da solução: -lambda_mov * qtd_movimentos (privilegia parcimônia).

    Parâmetros:
        estado (tuple | list): Tupla de 54 inteiros representando o cubo.
        qtd_movimentos (int): Quantidade de movimentos na proposta de solução.
        lambda_mov (float): Coeficiente de penalidade por movimento.
        retornar_detalhes (bool): Se True, retorna (score, detalhes_dict).

    Retorno:
        float | tuple: Pontuação (score/fitness) calculada, ou (score, detalhes).
    """
    # 1 e 2: Cantos (Corners - 8 peças)
    cantos_pos_correta = 0
    cantos_ori_correta = 0
    pontos_dist_cantos = 0
    for c_name, (coords, indices, targets, target_set) in CORNER_DATA.items():
        c0, c1, c2 = estado[indices[0]], estado[indices[1]], estado[indices[2]]
        c_set = frozenset((c0, c1, c2))
        home_info = CORNER_HOME.get(c_set)
        if home_info:
            _, home_coords, _ = home_info
            dist = abs(coords[0] - home_coords[0]) + abs(coords[1] - home_coords[1]) + abs(coords[2] - home_coords[2])
            pontos_dist_cantos += (6 - dist) * 2

        if c_set == target_set:
            cantos_pos_correta += 1
            if c0 == targets[0] and c1 == targets[1] and c2 == targets[2]:
                cantos_ori_correta += 1

    # 3 e 4: Arestas (Edges - 12 peças)
    arestas_pos_correta = 0
    arestas_ori_correta = 0
    pontos_dist_arestas = 0
    for e_name, (coords, indices, targets, target_set) in EDGE_DATA.items():
        c0, c1 = estado[indices[0]], estado[indices[1]]
        e_set = frozenset((c0, c1))
        home_info = EDGE_HOME.get(e_set)
        if home_info:
            _, home_coords, _ = home_info
            dist = abs(coords[0] - home_coords[0]) + abs(coords[1] - home_coords[1]) + abs(coords[2] - home_coords[2])
            pontos_dist_arestas += (4 - dist) * 2

        if e_set == target_set:
            arestas_pos_correta += 1
            if c0 == targets[0] and c1 == targets[1]:
                arestas_ori_correta += 1

    # 5: Pares de peças corretos (Pares F2L e Cruz da Base)
    pares_f2l_corretos = 0
    for c_slot, e_slot in F2L_PAIRS:
        _, c_indices, c_targets, _ = CORNER_DATA[c_slot]
        _, e_indices, e_targets, _ = EDGE_DATA[e_slot]
        if (estado[c_indices[0]] == c_targets[0] and
            estado[c_indices[1]] == c_targets[1] and
            estado[c_indices[2]] == c_targets[2] and
            estado[e_indices[0]] == e_targets[0] and
            estado[e_indices[1]] == e_targets[1]):
            pares_f2l_corretos += 1

    cruz_arestas_corretas = 0
    for e in ('DF', 'DB', 'DL', 'DR'):
        _, indices, targets, _ = EDGE_DATA[e]
        if estado[indices[0]] == targets[0] and estado[indices[1]] == targets[1]:
            cruz_arestas_corretas += 1

    bonus_cruz = cruz_arestas_corretas * 20 + (50 if cruz_arestas_corretas == 4 else 0)
    bonus_f2l = pares_f2l_corretos * 50
    pontos_pares = bonus_f2l + bonus_cruz

    # 6 Componentes da Decomposição do Score
    pts_pos_cantos = float(cantos_pos_correta * 25)
    pts_ori_cantos = float(cantos_ori_correta * 25)
    pts_pos_arestas = float(arestas_pos_correta * 20)
    pts_ori_arestas = float(arestas_ori_correta * 20)
    pts_pares = float(bonus_f2l + bonus_cruz)
    penalidade_tamanho = float(lambda_mov * qtd_movimentos)

    # Adesivos corretos e bônus de término 100% resolvido
    adesivos_corretos = sum(1 for i in range(54) if estado[i] == ESTADO_RESOLVIDO[i])
    bonus_resolvido = 600.0 if adesivos_corretos == 54 else 0.0

    score = (
        pts_pos_cantos
        + pts_ori_cantos
        + pts_pos_arestas
        + pts_ori_arestas
        + pontos_dist_cantos
        + pontos_dist_arestas
        + pts_pares
        + (adesivos_corretos * 2)
        + bonus_resolvido
        - penalidade_tamanho
    )

    if retornar_detalhes:
        detalhes = {
            # 6 Componentes da Decomposição do Score: Quantidades
            "posicao_cantos": cantos_pos_correta,
            "orientacao_cantos": cantos_ori_correta,
            "posicao_arestas": arestas_pos_correta,
            "orientacao_arestas": arestas_ori_correta,
            "pares_f2l": pares_f2l_corretos,
            "cruz_arestas": cruz_arestas_corretas,
            "penalidade_tamanho": penalidade_tamanho,

            # 6 Componentes da Decomposição do Score: Pontos
            "pts_posicao_cantos": pts_pos_cantos,
            "pts_orientacao_cantos": pts_ori_cantos,
            "pts_posicao_arestas": pts_pos_arestas,
            "pts_orientacao_arestas": pts_ori_arestas,
            "pts_pares_corretos": pts_pares,
            "pts_penalidade_tamanho": -penalidade_tamanho,

            # Métricas Globais de Estado
            "adesivos_corretos": adesivos_corretos,
            "resolvido": adesivos_corretos == 54,
            "score": score,
            "score_total": score,

            # Objeto Estruturado: Decomposição do Score Completa
            "decomposicao_score": {
                "posicao_cantos": {
                    "rotulo": "Posição correta dos cantos",
                    "qtd": cantos_pos_correta,
                    "max_qtd": 8,
                    "pontos": pts_pos_cantos,
                    "max_pontos": 200.0,
                    "texto": f"{cantos_pos_correta}/8 (+{pts_pos_cantos:.0f} pts)",
                },
                "orientacao_cantos": {
                    "rotulo": "Orientação correta dos cantos",
                    "qtd": cantos_ori_correta,
                    "max_qtd": 8,
                    "pontos": pts_ori_cantos,
                    "max_pontos": 200.0,
                    "texto": f"{cantos_ori_correta}/8 (+{pts_ori_cantos:.0f} pts)",
                },
                "posicao_arestas": {
                    "rotulo": "Posição correta das arestas",
                    "qtd": arestas_pos_correta,
                    "max_qtd": 12,
                    "pontos": pts_pos_arestas,
                    "max_pontos": 240.0,
                    "texto": f"{arestas_pos_correta}/12 (+{pts_pos_arestas:.0f} pts)",
                },
                "orientacao_arestas": {
                    "rotulo": "Orientação correta das arestas",
                    "qtd": arestas_ori_correta,
                    "max_qtd": 12,
                    "pontos": pts_ori_arestas,
                    "max_pontos": 240.0,
                    "texto": f"{arestas_ori_correta}/12 (+{pts_ori_arestas:.0f} pts)",
                },
                "pares_corretos": {
                    "rotulo": "Pares de peças corretos",
                    "f2l": pares_f2l_corretos,
                    "max_f2l": 4,
                    "cruz": cruz_arestas_corretas,
                    "max_cruz": 4,
                    "pontos": pts_pares,
                    "max_pontos": 330.0,
                    "texto": f"{pares_f2l_corretos}/4 F2L + Cruz {cruz_arestas_corretas}/4 (+{pts_pares:.0f} pts)",
                },
                "penalidade_tamanho": {
                    "rotulo": "Penalidade pelo tamanho da solução",
                    "movimentos": qtd_movimentos,
                    "pontos": -penalidade_tamanho,
                    "texto": f"-{penalidade_tamanho:.1f} pts ({qtd_movimentos} movs)",
                },
                "score_total": score,
            },
        }
        return score, detalhes

    return score


def calcular_score(lista_de_movimentos_estado_atual, lista_de_movimentos_proposta_solucao=None, cache=None, lambda_mov=0.5, retornar_detalhes=False):
    """
    Avalia a qualidade (Score / Fitness) de uma proposta de solução em relação a um estado inicial embaralhado.

    Avalia os 6 objetivos do Algoritmo Genético:
    + posição correta dos cantos
    + orientação correta dos cantos
    + posição correta das arestas
    + orientação correta das arestas
    + pares de peças corretos
    + penalidade pelo tamanho da solução

    Parâmetros:
        lista_de_movimentos_estado_atual (list | str): Sequência que embaralhou o cubo.
        lista_de_movimentos_proposta_solucao (list | str): Sequência de resolução candidata.
        cache (dict, opcional): Dicionário para memoização de scores já avaliados.
        lambda_mov (float): Peso da penalidade pelo tamanho da solução.
        retornar_detalhes (bool): Se True, retorna (score, detalhes).

    Retorno:
        float | tuple: Score calculado ou (score, detalhes).
    """
    # Tratamento caso o primeiro argumento já seja um estado pré-computado
    if (lista_de_movimentos_proposta_solucao is None and
        isinstance(lista_de_movimentos_estado_atual, (tuple, list)) and
        len(lista_de_movimentos_estado_atual) == 54 and
        isinstance(lista_de_movimentos_estado_atual[0], int)):
        return calcular_score_estado(
            lista_de_movimentos_estado_atual,
            qtd_movimentos=0,
            lambda_mov=lambda_mov,
            retornar_detalhes=retornar_detalhes,
        )

    if cache is not None and lista_de_movimentos_proposta_solucao is not None:
        chave = tuple(lista_de_movimentos_proposta_solucao) if not isinstance(lista_de_movimentos_proposta_solucao, str) else lista_de_movimentos_proposta_solucao
        if chave in cache:
            return cache[chave]

    # Inicia do estado resolvido
    estado = ESTADO_RESOLVIDO

    # Aplica o embaralhamento
    if lista_de_movimentos_estado_atual:
        estado = aplicar_movimentos(estado, lista_de_movimentos_estado_atual)

    # Aplica a proposta de solução
    qtd_movs = len(lista_de_movimentos_proposta_solucao) if lista_de_movimentos_proposta_solucao else 0
    if lista_de_movimentos_proposta_solucao:
        estado = aplicar_movimentos(estado, lista_de_movimentos_proposta_solucao)

    res = calcular_score_estado(
        estado,
        qtd_movimentos=qtd_movs,
        lambda_mov=lambda_mov,
        retornar_detalhes=retornar_detalhes,
    )

    if cache is not None and lista_de_movimentos_proposta_solucao is not None:
        cache[chave] = res

    return res


def calcular_fitness_avancado(estado, qtd_movimentos=0, lambda_mov=0.5, retornar_detalhes=False):
    """
    Função de fitness utilizada internamente pelo motor evolutivo,
    retornando a tupla (score, adesivos_corretos).
    """
    score, det = calcular_score_estado(
        estado,
        qtd_movimentos=qtd_movimentos,
        lambda_mov=lambda_mov,
        retornar_detalhes=True,
    )
    if retornar_detalhes:
        return score, det["adesivos_corretos"], det
    return score, det["adesivos_corretos"]


calcular_fitness = calcular_score