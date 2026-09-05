"""
==============================================================================
CRUZAMENTO.PY - OPERADOR GENÉTICO DE RECOMBINAÇÃO (CROSSOVER) E REPARAÇÃO
==============================================================================
Este módulo implementa o operador de recombinação genética (Single-Point Crossover)
e o mecanismo determinístico de reparação sintática O(N) de sequências de giros.

1. Fundamentação Teórica - Teorema dos Esquemas de Holland:
   Em Algoritmos Genéticos aplicados a problemas de busca em grafos de Cayley,
   sequências de rotações formam "Building Blocks" (sub-sequências de alta aptidão
   que posicionam e orientam peças específicas, como pares de F2L ou arestas da cruz).
   O crossover de ponto único permite que blocos benéficos descobertos em linhagens
   distintas se combinem em um único descendente, promovendo o salto qualitativo
   no espaço de estados:
       Pai 1:  [ g_1, g_2, ..., g_k | g_{k+1}, ..., g_N ]
       Pai 2:  [ h_1, h_2, ..., h_k | h_{k+1}, ..., h_N ]
       --------------------------------------------------
       Filho 1: [ g_1, ..., g_k | h_{k+1}, ..., h_N ]
       Filho 2: [ h_1, ..., h_k | g_{k+1}, ..., g_N ]

2. Problema da Incoerência na Fronteira de Junção (Boundary Condition):
   Quando a cauda do Pai 1 é unida à cabeça do Pai 2 no índice de corte k, pode surgir
   uma violação sintática na transição g_k -> h_{k+1}, tais como:
   a) Mesma Face Consecutiva: Ex: g_k = 'R' e h_{k+1} = "R'". Isso gera redundância
      ou auto-cancelamento desnecessário, desperdiçando genes.
   b) Comutatividade Paralela Degradada: Ex: g_{k-1} = 'R', g_k = 'L' e h_{k+1} = 'R',
      gerando repetição da mesma face em eixos paralelos independentes.

3. Mecanismo de Reparação Linear O(N):
   Para manter a população estritamente no espaço viável sem o custo proibitivo de
   rejeitar e regerar indivíduos, a função `reparar_individuo` varre o cromossomo em
   um único passo linear O(N). Utilizando a tabela pré-computada de transições válidas
   `VALID_NEXT_MOVES_SET`, qualquer gene que viole o contexto (face_ant, face_ret) é
   imediatamente substituído por um alelo válido equivalente, preservando a integridade
   estrutural do cromossomo.
==============================================================================
"""

import random
from populacao import MOVIMENTOS, PARALELAS, VALID_NEXT_MOVES, VALID_NEXT_MOVES_SET



def obter_face(movimento):
    """
    Extrai a letra que representa a face do Cubo Mágico a partir de um código de movimento.
    Exemplo: 'R'' -> 'R', 'U2' -> 'U', 'F' -> 'F'.

    Parâmetros:
        movimento (str): Notação do movimento.

    Retorno:
        str: Letra da face ('U', 'D', 'F', 'B', 'R', 'L').
    """
    return movimento[0]


def reparar_individuo(individuo):
    """
    Garante que uma sequência de movimentos respeite rigorosamente as regras
    de não redundância do Cubo Mágico em uma única passada linear O(N).

    Caso um gene viole as regras (mesma face consecutiva ou face paralela oposta intercalada),
    ele é substituído por um movimento válido selecionado aleatoriamente da tabela
    de transições permitidas para aquele contexto.

    Parâmetros:
        individuo (list[str]): Sequência de movimentos a ser validada e reparada.

    Retorno:
        list[str]: Nova sequência de movimentos 100% válida e não redundante.
    """
    reparado = []
    face_ant = None
    face_ret = None

    for mov in individuo:
        chave = (face_ant, face_ret)
        # Se o movimento atual for válido no contexto anterior, mantém
        if mov in VALID_NEXT_MOVES_SET[chave]:
            reparado.append(mov)
            face_ret = face_ant
            face_ant = mov[0]
        else:
            # Caso contrário, seleciona um substituto válido compatível
            opcoes = VALID_NEXT_MOVES[chave]
            novo_mov = random.choice(opcoes) if opcoes else mov
            reparado.append(novo_mov)
            face_ret = face_ant
            face_ant = novo_mov[0]

    return reparado


def cruzar_dois_individuos(pai1, pai2):
    """
    Executa o Cruzamento de Ponto Único (Single-Point Crossover) entre dois pais.

    Procedimento:
    1. Escolhe aleatoriamente um ponto de corte entre os genes dos pais.
    2. Filho 1 herda a primeira metade do Pai 1 e a segunda metade do Pai 2.
    3. Filho 2 herda a primeira metade do Pai 2 e a segunda metade do Pai 1.
    4. Aplica o reparo genético em ambos os filhos para garantir a não redundância.

    Parâmetros:
        pai1 (list[str]): Primeiro indivíduo progenitor.
        pai2 (list[str]): Segundo indivíduo progenitor.

    Retorno:
        tuple[list[str], list[str]]: Tupla com os dois filhos gerados e reparados (filho1, filho2).
    """
    tamanho = len(pai1)
    if tamanho <= 1:
        return list(pai1), list(pai2)

    # Ponto de corte aleatório
    ponto_corte = random.randint(1, tamanho - 1)

    # Recombinação dos blocos de genes
    filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
    filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]

    # Reparação de possíveis incoerências na junção do corte
    return reparar_individuo(filho1), reparar_individuo(filho2)


def cruzamento(populacao, porcentagem_cruzamento):
    """
    Aplica o operador de cruzamento em pares de indivíduos da população com base
    na taxa de probabilidade estipulada (porcentagem_cruzamento).

    Parâmetros:
        populacao (list[list[str]]): População de indivíduos a cruzar.
        porcentagem_cruzamento (float): Probabilidade de cruzamento para cada par (ex: 0.70 = 70%).

    Retorno:
        list[list[str]]: Nova população contendo os indivíduos após o processo de cruzamento.
    """
    populacao_cruzada = [list(ind) for ind in populacao]
    n = len(populacao_cruzada)

    if n < 2 or len(populacao_cruzada[0]) <= 1:
        return populacao_cruzada

    # Processa os indivíduos em pares consecutivos (0 e 1, 2 e 3, etc.)
    for i in range(0, n - 1, 2):
        if random.random() < porcentagem_cruzamento:
            f1, f2 = cruzar_dois_individuos(populacao_cruzada[i], populacao_cruzada[i + 1])
            populacao_cruzada[i] = f1
            populacao_cruzada[i + 1] = f2

    return populacao_cruzada
