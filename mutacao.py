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



# ==============================================================================
# COMUTADORES E MACROS CANÔNICOS DE SPEEDCUBING (MÉTODO DE JESSICA FRIDRICH / CFOP)
# ==============================================================================
# No método de Jessica Fridrich (CFOP), para alterar peças de camadas superiores
# ou inserir pares F2L sem destruir a Cruz da base (D) ou blocos já consolidados,
# utilizam-se COMUTADORES e CONJUGADOS da Teoria de Grupos:
#
# 1. Comutadores: [A, B] = A * B * A' * B'
#    Afetam exclusivamente a intersecção das peças movidas por A e B, mantendo
#    o restante do cubo 100% invariante (protegendo o progresso genético anterior).
#
# 2. Conjugados: A * B * A'
#    Trazem uma peça para a camada de trabalho (A), aplicam a modificação (B) e
#    desfazem a preparação (A').
#
# Ao injetar estas macros na mutação avançada, o Algoritmo Genético ganha "saltos
# quânticos" no espaço de busca, evitando desmanchar estágios anteriores de Fridrich.
# ==============================================================================
MACROS_SPEEDCUBING = (
    ("R", "U", "R'", "U'"),                   # Sexy Move: [R, U] - comutador elementar de F2L / OLL
    ("U", "R", "U'", "R'"),                   # Reverse Sexy: inserção ágil de pares
    ("R'", "F", "R", "F'"),                   # Sledgehammer: orienta arestas sem afetar slots protegidos
    ("F", "R", "F'", "R'"),                   # Hedgeslammer: variação frontal do Sledgehammer
    ("R", "U", "R'", "U", "R", "U2", "R'"),   # Sune: algoritmo canônico de Fridrich para orientação de cantos (OLL)
    ("R", "U2", "R'", "U'", "R", "U'", "R'"), # Anti-Sune: simétrico inverso do Sune
    ("L'", "U'", "L", "U"),                   # Lefty Sexy: espelhamento para a mão esquerda
    ("R", "U'", "R'"),                        # Inserção Direita de Par F2L (preserva a cruz)
    ("L'", "U", "L"),                         # Inserção Esquerda de Par F2L (preserva a cruz)
    ("F'", "U'", "F"),                        # Inserção Frontal Direta
    ("F", "U", "F'"),                         # Inserção Frontal Esquerda
    ("R2", "U", "R", "U", "R'", "U'", "R'", "U'", "R'", "U", "R'"), # Allan / U-perm (PLL de permutação de 3 arestas)
)


def mutar_individuo_avancado(individuo, porcentagem_mutacao, chance_macro=0.35):
    """
    Aplica mutação avançada combinando mutação pontual por gene e
    macro-mutações com comutadores de speedcubing que preservam a integridade estrutural.

    Parâmetros:
        individuo (list[str]): Sequência de movimentos do indivíduo.
        porcentagem_mutacao (float): Probabilidade de mutação pontual.
        chance_macro (float): Probabilidade de aplicar uma substituição por comutador.

    Retorno:
        list[str]: Novo cromossomo mutado.
    """
    if not individuo:
        return []

    n = len(individuo)
    if random.random() < chance_macro and n >= 4:
        from populacao import simplificar_movimentos
        macro = list(random.choice(MACROS_SPEEDCUBING))
        idx = random.randint(0, max(0, n - len(macro)))
        cand = individuo[:idx] + macro + individuo[idx + len(macro):]
        cand = simplificar_movimentos(cand)
        if len(cand) == n:
            return cand
        elif len(cand) > n:
            return cand[:n]
        else:
            # Completa até o tamanho original com genes válidos se necessário
            from populacao import VALID_NEXT_MOVES
            while len(cand) < n:
                f_ant = cand[-1][0] if cand else None
                f_ret = cand[-2][0] if len(cand) >= 2 else None
                cand.append(random.choice(VALID_NEXT_MOVES[(f_ant, f_ret)]))
            return cand

    return mutar_individuo(individuo, porcentagem_mutacao)


def mutacao(populacao, porcentagem_mutacao):
    """
    Aplica o operador de mutação a todos os indivíduos de uma população.

    Parâmetros:
        populacao (list[list[str]]): Lista de indivíduos que compõem a população.
        porcentagem_mutacao (float): Taxa de mutação por gene.

    Retorno:
        list[list[str]]: Nova população com as mutações aplicadas.
    """
    return [mutar_individuo_avancado(ind, porcentagem_mutacao) for ind in populacao]

