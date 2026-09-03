import random
from populacao import MOVIMENTOS, PARALELAS, VALID_NEXT_MOVES


def obter_face(movimento):
    """Extrai a letra que representa a face do Cubo Mágico a partir de um movimento."""
    return movimento[0]


def movimento_valido_no_indice(individuo, indice, novo_movimento):
    """
    Verifica se substituir o gene em 'indice' por 'novo_movimento' mantém
    o indivíduo válido segundo as regras de não redundância.
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
    Aplica mutação em um indivíduo com probabilidade 'porcentagem_mutacao' por gene
    utilizando a tabela de transição para filtrar candidatos válidos instantaneamente.
    """
    if not individuo or porcentagem_mutacao <= 0:
        return list(individuo)

    individuo_mutado = list(individuo)
    n = len(individuo_mutado)

    for i in range(n):
        if random.random() < porcentagem_mutacao:
            face_ant = individuo_mutado[i - 1][0] if i > 0 else None
            face_ret = individuo_mutado[i - 2][0] if i > 1 else None

            # Candidatos compatíveis com os antecessores
            candidatos_base = VALID_NEXT_MOVES.get((face_ant, face_ret), MOVIMENTOS)

            # Filtra com sucessores (se existirem)
            face_pos = individuo_mutado[i + 1][0] if i < n - 1 else None
            face_pos_pos = individuo_mutado[i + 2][0] if i < n - 2 else None

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

            if candidatos:
                individuo_mutado[i] = random.choice(candidatos)

    return individuo_mutado


def mutacao(populacao, porcentagem_mutacao):
    """
    Aplica o operador de mutação em toda a população, retornando uma nova lista de indivíduos.
    """
    return [mutar_individuo(ind, porcentagem_mutacao) for ind in populacao]
