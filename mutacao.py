# ==============================================================================
# MUTACAO.PY - OPERADOR GENÉTICO DE MUTAÇÃO E DIVERSIDADE
# ==============================================================================
# Este módulo implementa o operador de mutação pontual do Algoritmo Genético.
# A mutação é essencial para introduzir nova variabilidade genética na população,
# evitando que o algoritmo fique preso em ótimos locais (estagnação).
#
# Para cada gene do cromossomo que sofre mutação, verificamos a compatibilidade
# tanto com os movimentos anteriores (i-1 e i-2) quanto com os posteriores (i+1 e i+2),
# garantindo que o cromossomo permaneça 100% válido e sem redundâncias.
# ==============================================================================

import random
from populacao import MOVIMENTOS, PARALELAS, VALID_NEXT_MOVES


def obter_face(movimento):
    """
    Extrai a letra que representa a face do Cubo Mágico a partir de um código de movimento.

    Parâmetros:
        movimento (str): Código do movimento (ex: 'F2', 'D'', 'U').

    Retorno:
        str: Letra inicial da face.
    """
    return movimento[0]


def movimento_valido_no_indice(individuo, indice, novo_movimento):
    """
    Verifica se a substituição do gene no índice 'indice' pelo 'novo_movimento'
    mantém o indivíduo válido perante seus vizinhos anteriores e posteriores.

    Regras testadas:
    1. Vizinhança anterior imediata: face != face anterior (i - 1).
    2. Vizinhança anterior retrasada: face != face retrasada (i - 2) se a anterior for paralela.
    3. Vizinhança posterior imediata: face != face posterior (i + 1).
    4. Vizinhança posterior pós-posterior: face != face pós-posterior (i + 2) se a posterior for paralela.

    Parâmetros:
        individuo (list[str]): Sequência completa de movimentos.
        indice (int): Posição do gene que está sendo testado para substituição.
        novo_movimento (str): Novo movimento candidato para a posição.

    Retorno:
        bool: True se for seguro e válido substituir o gene, False caso contrário.
    """
    face = novo_movimento[0]

    # 1. Checagem com o elemento anterior (i - 1)
    if indice > 0:
        face_ant = individuo[indice - 1][0]
        if face == face_ant:
            return False
        # 2. Checagem com o elemento retrasado (i - 2)
        if indice > 1:
            face_ret = individuo[indice - 2][0]
            if face == face_ret and PARALELAS.get(face) == face_ant:
                return False

    # 3. Checagem com o elemento posterior (i + 1)
    if indice < len(individuo) - 1:
        face_pos = individuo[indice + 1][0]
        if face == face_pos:
            return False
        # 4. Checagem com o elemento pós-posterior (i + 2)
        if indice < len(individuo) - 2:
            face_pos_pos = individuo[indice + 2][0]
            if face == face_pos_pos and PARALELAS.get(face) == face_pos:
                return False

    return True


def mutar_individuo(individuo, porcentagem_mutacao):
    """
    Aplica mutação aleatória nos genes de um indivíduo com probabilidade 'porcentagem_mutacao'.

    Procedimento:
    1. Para cada gene do indivíduo, sorteia um número de 0 a 1.
    2. Se o número for menor que a taxa de mutação, busca alternativas válidas
       que respeitem os limites da vizinhança na sequência.
    3. Seleciona aleatoriamente um dos candidatos válidos para substituir o gene atual.

    Parâmetros:
        individuo (list[str]): Indivíduo original.
        porcentagem_mutacao (float): Probabilidade de mutação por gene (ex: 0.05 = 5%).

    Retorno:
        list[str]: Novo indivíduo após o processo de mutação.
    """
    if not individuo or porcentagem_mutacao <= 0:
        return list(individuo)

    individuo_mutado = list(individuo)
    n = len(individuo_mutado)

    for i in range(n):
        if random.random() < porcentagem_mutacao:
            # Obtém as faces dos genes anteriores
            face_ant = individuo_mutado[i - 1][0] if i > 0 else None
            face_ret = individuo_mutado[i - 2][0] if i > 1 else None

            # Candidatos preliminares compatíveis com os antecessores
            candidatos_base = VALID_NEXT_MOVES.get((face_ant, face_ret), MOVIMENTOS)

            # Obtém as faces dos genes posteriores para filtro de compatibilidade
            face_pos = individuo_mutado[i + 1][0] if i < n - 1 else None
            face_pos_pos = individuo_mutado[i + 2][0] if i < n - 2 else None

            # Filtra os candidatos que também respeitam os genes posteriores
            candidatos = []
            for m in candidatos_base:
                if m == individuo_mutado[i]:
                    continue
                f = m[0]
                if face_pos is not None:
                    if f == face_pos:
                        continue
                    if face_pos_pos is not None and f == face_pos_pos and PARALELAS.get(f) == face_pos:
                        continue
                candidatos.append(m)

            # Substitui pelo novo gene sorteado entre os válidos
            if candidatos:
                individuo_mutado[i] = random.choice(candidatos)

    return individuo_mutado


def mutacao(populacao, porcentagem_mutacao):
    """
    Aplica o operador de mutação a todos os indivíduos de uma população.

    Parâmetros:
        populacao (list[list[str]]): Lista de indivíduos que compõem a população.
        porcentagem_mutacao (float): Taxa de mutação por gene.

    Retorno:
        list[list[str]]: Nova população com as mutações aplicadas.
    """
    return [mutar_individuo(ind, porcentagem_mutacao) for ind in populacao]
