readme_content = """# Implementação e Otimização de Algoritmo Genético em Python para Resolução Heurística do Cubo de Rubik

**Autor:** Luiz Fernando Brogliatto Ferreira  
**Instituição:** SENAC PARANÁ  
**Endereço:** R. Guaíra, 3332 – Jardim La Salle, CEP 85903-000 – Toledo – PR – Brasil  
**Contato:** luiz0067@gmail.com  
**Repositório:** [python-algoritmo-genetico-inteligencia-artificial-cubo-de-rubik](https://github.com/luiz0067yahoo/python-algoritmo-genetico-inteligencia-artificial-cubo-de-rubik)

---

## 📌 Resumo

O Cubo de Rubik é um quebra-cabeça tridimensional com aproximadamente $4,32 \\times 10^{19}$ estados possíveis e diâmetro de grafo comprovado em no máximo 20 movimentos (*Número de Deus*). Este projeto implementa um resolvedor em **Python** fundamentado em **Algoritmos Genéticos (AGs)**, explorando conceitos de seleção natural, cruzamento e mutação genética para restaurar as faces do cubo sem recorrer a buscas exaustivas.

Inspirado nas bases da biologia molecular e genética clássica (Mendel, Miescher, Watson & Crick) e nos sistemas adaptativos artificiais de John Holland, o algoritmo codifica rotações espaciais em cromossomos computacionais, aplicando seleção por torneio e filtragem de redundância cinemática.

---

## 🧬 Fundamentação Teórica e Inspiração Bioinspirada

O fluxo da computação evolucionária reflete os principais marcos históricos da transmissão de informação biológica:

1. **Leis da Hereditariedade e Segregação (Gregor Mendel, 1866):** Padrões probabilísticos e transmissão discreta de fatores dominantes e recessivos determinando o fenótipo.
2. **Identificação da Nucleína e Base Química (Friedrich Miescher, 1871):** Isolamento da base molecular da informação biológica.
3. **Evolução Social e Seleção Natural (Charles Darwin, 1859; Lewis H. Morgan, 1877; John Ball, 1878):** Competição, sobrevivência dos mais aptos e adaptação populacional ao meio.
4. **Estrutura em Dupla Hélice (Watson & Crick, 1953):** A informação celular estruturada digitalmente via sequências lineares de bases nitrogenadas.
5. **Algoritmos Genéticos e Teorema dos Esquemas (John H. Holland, 1975/1992):** Formalização dos operadores computacionais de seleção, *crossover* e mutação aplicados a vetores binários/alfanuméricos.

---

## 🧩 O Problema do Cubo de Rubik

- **Espaço Amostral:** 
  $$\\frac{8! \\times 3^7 \\times 12! \\times 2^{11}}{2} \\approx 43.252.003.274.489.856.000 \\text{ combinações}$$
- **Número de Deus:** Qualquer configuração desordenada pode ser resolvida em no máximo 20 movimentos (Rokicki et al., 2010).
- **Notação Padrão dos Movimentos:**
  - Faces: **F** (Front), **B** (Back), **U** (Up), **D** (Down), **L** (Left), **R** (Right).
  - Sentidos: Horário ($X$), Anti-horário ($X'$) e Giro Duplo ($X2$), totalizando 18 movimentos elementares.
- **Comparativo Heurístico:**
  - **CFOP / Jessica Fridrich (2003):** Abordagem clássica humana em camadas (*Cross, F2L, OLL, PLL*), rápida mas dependente de padrões memorizados.
  - **Algoritmo Genético:** Abordagem estocástica global livre de tabelas pré-definidas, adaptando-se via cálculo posicional de aptidão.

---

## ⚙️ Arquitetura do Sistema e Algoritmo

```text
+-----------------------------------------------------------+
|                    População Inicial                      |
|       (N indivíduos compostos por sequências de giros)    |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|               Avaliação de Aptidão (Fitness)              |
|        F(x) = Facetas Móveis Alinhadas / 48 peças         |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|               Seleção por Torneio (K=3)                   |
|        Competição estocástica entre indivíduos            |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|              Recombinação Genética (Crossover)            |
|       Cruzamento de ponto único / dois pontos de corte    |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|               Operador de Mutação com Sanitização         |
|      Troca estocástica e eliminação de movimentos nulos   |
+-----------------------------+-----------------------------+
                              |
                              v
               +--------------+--------------+
               |  Critério de Parada Atingido? |
               +--------------+--------------+
                     |                     |
                   [Sim]                 [Não]
                     |                     |
                     v                     +---> (Nova Geração)
       +-----------------------------+
       |   Solução Ótima Encontrada  |
       |       (Fitness = 1.0)       |
       +-----------------------------+