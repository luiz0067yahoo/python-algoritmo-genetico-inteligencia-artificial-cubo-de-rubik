import random
from populacao import MOVIMENTOS, gerar_populacao

# Dicionário que mapeia faces opostas/paralelas do Cubo Mágico
PARALELAS = {"U": "D", "D": "U", "F": "B", "B": "F", "R": "L", "L": "R"}


def buscar_posicao(posicoes, posicao):
    """Verifica se uma posição específica já está presente na lista de posições."""
    resultado = False
    for x in posicoes:
        if posicao == x:
            resultado = True
            break
    return resultado


def criar_posicoes(porcentagem_mutacao):
    """Gera uma lista ordenada de índices aleatórios (de 0 a 99) sem repetição

    para definir onde ocorrem as mutações, com base na taxa percentual.
    """
    posicoes = []
    quantidade_posicoes = porcentagem_mutacao * 100
    for i in range(round(quantidade_posicoes)):
        while True:
            # Seleciona um índice aleatório entre 0 e 99
            posicao = round((random.random() * 99))
            # Garante que a posição sorteada seja única na lista
            if not (buscar_posicao(posicoes, posicao)):
                break
        posicoes.append(posicao)
    # Ordena os índices para permitir verificação sequencial durante o loop
    posicoes.sort()
    return posicoes


def obter_face(movimento):
    """Extrai a letra que representa a face do Cubo Mágico a partir de um movimento."""
    return movimento[0]


def mutacao(populacao, porcentagem_mutacao):
    """Aplica o operador de mutação em uma população de soluções do Cubo Mágico,

    respeitando restrições de redundância de movimentos.
    """
    # Gera os pontos de mutação para o primeiro bloco de 100 genes
    posicoes = criar_posicoes(porcentagem_mutacao)
    posicao = 0
    contador_posicao = 0

    # Iteração sobre cada indivíduo da população
    for contador_individuo in range(len(populacao)):
        individuo = populacao[contador_individuo]

        # Iteração sobre cada cromossomo (movimento) do indivíduo
        for contador_cromossomo in range(len(individuo)):
            # A cada 100 genes processados, reinicia o contador e recalcula as posições
            if (contador_posicao > 0) and (contador_posicao % 100 == 0):
                contador_posicao = 0
                posicoes = criar_posicoes(porcentagem_mutacao)
                posicao = 0

            # Verifica se o gene atual foi selecionado para sofrer mutação
            if contador_posicao == posicoes[posicao]:
                valido = False

                # Sorteia um novo movimento que cumpra todas as regras de validação
                while not valido:
                    cromossomo_mutante = random.choice(MOVIMENTOS)

                    # Condição original: o novo movimento deve ser diferente do atual
                    if cromossomo_mutante == individuo[contador_cromossomo]:
                        continue

                    face_mutante = obter_face(cromossomo_mutante)

                    # Regra 1: Evita movimentos consecutivos na mesma face (ex: U U)
                    if contador_cromossomo > 0:
                        face_anterior = obter_face(
                            individuo[contador_cromossomo - 1]
                        )
                        if face_mutante == face_anterior:
                            continue

                    # Regra 2: Evita sequências redundantes intercaladas por paralelas (ex: U D U)
                    if contador_cromossomo > 1:
                        face_anterior = obter_face(
                            individuo[contador_cromossomo - 1]
                        )
                        face_retrasada = obter_face(
                            individuo[contador_cromossomo - 2]
                        )

                        if (
                            PARALELAS.get(face_anterior) == face_mutante
                            and face_mutante == face_retrasada
                        ):
                            continue

                    valido = True

                # Aplica o cromossomo mutante no indivíduo
                individuo[contador_cromossomo] = cromossomo_mutante
                populacao[contador_individuo] = individuo

                # Avança para a próxima posição mutável da lista
                if posicao < len(posicoes) - 1:
                    posicao += 1
                else:
                    posicao = 0

            contador_posicao += 1

    return populacao


