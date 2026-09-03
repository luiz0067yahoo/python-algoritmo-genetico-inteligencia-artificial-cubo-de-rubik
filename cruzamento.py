# ==============================================================================
# CRUZAMENTO.PY - OPERADOR GENÉTICO DE RECOMBINAÇÃO (CROSSOVER) E REPARAÇÃO
# ==============================================================================
# Este módulo implementa a recombinação genética de ponto único entre pares
# de indivíduos selecionados (pais) para produzir novos descendentes (filhos).
#
# Após o corte e união dos genes dos pais, pode ocorrer de a junção criar
# movimentos redundantes (ex: juntar um final 'R' com um início 'R'').
# Para corrigir isso, aplicamos a função 'reparar_individuo', que percorre o filho
# em uma única passada O(N) e substitui qualquer gene inválido por uma alternativa
# válida e compatível da tabela de transições.
# ==============================================================================

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
