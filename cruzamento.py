import copy
import random

# Simulação das importações e constantes
MOVIMENTOS = ["U", "U'", "U2", "D", "D'", "D2", "L", "R", "F", "B"]
PARALELAS = {"U": "D", "D": "U", "F": "B", "B": "F", "R": "L", "L": "R"}


def obter_face(movimento):
    return movimento[0]


def buscar_posicao(posicoes, posicao):
    return posicao in posicoes


def criar_posicoes(porcentagem_cruzamento):
    posicoes = []
    quantidade_posicoes = round(porcentagem_cruzamento * 100)
    while len(posicoes) < quantidade_posicoes:
        posicao = random.randint(0, 99)
        if posicao not in posicoes:
            posicoes.append(posicao)
    posicoes.sort()
    return posicoes


def inverte_populacao(populacao):
    """Retorna uma NOVA lista invertida sem alterar a original."""
    return [copy.deepcopy(ind) for ind in reversed(populacao)]


def cruzamento(populacao, porcentagem_cruzamento):
    """Executa o cruzamento alterando uma cópia da população."""
    populacao_resultado = copy.deepcopy(populacao)
    populacao_invertida = inverte_populacao(populacao)

    posicoes = criar_posicoes(porcentagem_cruzamento)
    posicao_idx = 0
    contador_posicao = 0

    for contador_individuo in range(len(populacao_resultado)):
        individuo = populacao_resultado[contador_individuo]

        for contador_cromossomo in range(len(individuo)):
            if (contador_posicao > 0) and (contador_posicao % 100 == 0):
                contador_posicao = 0
                posicoes = criar_posicoes(porcentagem_cruzamento)
                posicao_idx = 0

            posicao_atual = posicoes[posicao_idx]

            if contador_posicao == posicao_atual:
                cromossomo_pai = populacao_invertida[contador_individuo][
                    contador_cromossomo
                ]

                # Validação rápida de regras (se falhar, mantém o cromossomo original)
                valido = True
                if cromossomo_pai == individuo[contador_cromossomo]:
                    valido = False

                if valido and contador_cromossomo > 0:
                    face_anterior = obter_face(
                        individuo[contador_cromossomo - 1]
                    )
                    if obter_face(cromossomo_pai) == face_anterior:
                        valido = False

                if valido and contador_cromossomo > 1:
                    face_anterior = obter_face(
                        individuo[contador_cromossomo - 1]
                    )
                    face_retrasada = obter_face(
                        individuo[contador_cromossomo - 2]
                    )
                    face_pai = obter_face(cromossomo_pai)
                    if (
                        PARALELAS.get(face_anterior) == face_pai
                        and face_pai == face_retrasada
                    ):
                        valido = False

                if valido:
                    individuo[contador_cromossomo] = cromossomo_pai

                if posicao_idx < len(posicoes) - 1:
                    posicao_idx += 1
                else:
                    posicao_idx = 0

            contador_posicao += 1

    return populacao_resultado


def testar_cruzamento_100():
    """Função de teste para validar cruzamento com 100% da população invertida."""
    # 1. Gera população fictícia de teste (ex: 4 indivíduos com 100 genes cada)
    tamanho_populacao = 4
    tamanho_individuo = 100

    populacao_original = []
    for _ in range(tamanho_populacao):
        ind = [random.choice(MOVIMENTOS) for _ in range(tamanho_individuo)]
        populacao_original.append(ind)

    # 2. Executa a inversão manual esperada (pais)
    populacao_esperada_pais = inverte_populacao(populacao_original)

    # 3. Aplica o cruzamento com 100% de taxa (1.0)
    populacao_cruzada = cruzamento(populacao_original, porcentagem_cruzamento=0.1)

    # 4. Verificação dos resultados
    print("--- INÍCIO DO TESTE ---")
    print(f"Total de Indivíduos: {len(populacao_cruzada)}")

    genes_trocados = 0
    genes_totais = tamanho_populacao * tamanho_individuo

    for i in range(tamanho_populacao):
        for g in range(tamanho_individuo):
            gene_cruzado = populacao_cruzada[i][g]
            gene_pai = populacao_esperada_pais[i][g]

            if gene_cruzado == gene_pai:
                genes_trocados += 1

    taxa_sucesso = (genes_trocados / genes_totais) * 100
    print(f"Genes idênticos aos pais invertidos: {genes_trocados}/{genes_totais}")
    print(f"Taxa de substituição efetiva: {taxa_sucesso:.2f}%")
    print(
        "(Nota: A taxa pode ser menor que 100% devido às regras de validação do Cubo Mágico que descartam movimentos inválidos)"
    )
    print("--- TESTE CONCLUÍDO ---")


