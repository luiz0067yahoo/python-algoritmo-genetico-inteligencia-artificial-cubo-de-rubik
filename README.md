# 🧩 RubikLab AI — Solucionador de Cubo de Rubik 3x3x3 com Algoritmo Genético

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Backend-Flask-green?logo=flask)
![Three.js](https://img.shields.io/badge/Frontend-Three.js-black?logo=three.js)
![WCA Compliance](https://img.shields.io/badge/Scramble-WCA%20Official-orange)
![Performance](https://img.shields.io/badge/Performance-%2B400k%20evals%2Fs-purple)

Um sistema completo de Inteligência Artificial e Computação Evolutiva para resolução e visualização 3D do **Cubo de Rubik (Cubo Mágico 3x3x3)** através de **Algoritmos Genéticos de Alta Performance com Processamento Paralelo Multi-Core**.

---

## 📋 Sumário
- [Destaques do Projeto](#-destaques-do-projeto)
- [Fundamentação Teórica e Algoritmo Genético](#-fundamentação-teórica-e-algoritmo-genético)
- [Arquitetura e Otimizações de Performance](#-arquitetura-e-otimizações-de-performance)
- [Estrutura de Arquivos](#-estrutura-de-arquivos)
- [Instalação e Execução](#-instalação-e-execução)
- [Interface Gráfica 3D](#-interface-gráfica-3d)
- [Documentação da API REST](#-documentação-da-api-rest)
- [Métricas de Desempenho](#-métricas-de-desempenho)

---

## 🚀 Destaques do Projeto

- **Motor de Permutação Direta $O(1)$**: Substituição de simulações orientadas a objetos por permutações de 54 adesivos em memória, elevando a velocidade de avaliação de ~310 para mais de **400.000 avaliações por segundo**.
- **Processamento Paralelo Multi-Core Dinâmico (100% de Hardware)**: Detecção automática do processador do sistema (ex: **AMD Ryzen™ 7 PRO 8700GE** com 8 núcleos e 16 threads), alocando **16 ilhas paralelas de evolução simultânea** com migração periódica de elites.
- **Métricas Completas de Cromossomos no Dashboard**: Exibição em tempo real da quantidade de cromossomos por geração (população ativa), comprimento do cromossomo (genes/movimentos), cromossomos de elite preservados e total acumulado avaliado.
- **Conformidade Oficial WCA**: Gerador de embaralhamento oficial segundo o Regulamento Internacional da *World Cube Association* (Artigo 12 / Regulação 4b).
- **Interface 3D Interativa**: Renderização com Three.js, iluminação realista, planificação 2D em tempo real e animação passo a passo da solução encontrada.
- **Resolução Incremental Automática**: Testa progressivamente comprimentos de cromossomo de 1 a 54 movimentos com busca exaustiva instantânea para espaços pequenos ($< 0.02s$).

---

## 🧬 Fundamentação Teórica e Algoritmo Genético

O Algoritmo Genético busca encontrar a sequência de movimentos que transforma um cubo embaralhado no estado resolvido.

```mermaid
graph TD
    A[Cubo Embaralhado WCA] --> B[População Inicial de Cromossomos]
    B --> C[Avaliação de Fitness - Score 0 a 54]
    C --> D{Score == 54?}
    D -- Sim --> E[Solução Ótima Encontrada]
    D -- Não --> F[Seleção dos Melhores Indivíduos]
    F --> G[Elitismo - Preservação dos Top 5%]
    F --> H[Cruzamento / Crossover de Ponto Único]
    H --> I[Reparo Linear O(N) sem Redundâncias]
    I --> J[Mutação Adaptativa por Gene]
    J --> K[Migração de Elites entre Ilhas CPU]
    K --> C
    E --> L[Animação e Resolução Automática no Cubo 3D]
```

### 1. Representação do Cromossomo (Genótipo)
- Cada **gene** é um movimento em Notação Canônica WCA: `U, U', U2, D, D', D2, F, F', F2, B, B', B2, R, R', R2, L, L', L2`.
- O **cromossomo** é uma sequência de $N$ movimentos (comprimento incremental).

### 2. Função de Aptidão (Fitness)
- A função de fitness compara o estado resultante da aplicação dos movimentos com o cubo resolvido.
- **Score máximo**: $54$ (6 faces $\times$ 9 adesivos na cor e posição correta).

### 3. Regras de Não Redundância
- **Regra 1**: Proíbe movimentos consecutivos na mesma face ($F_i \neq F_{i-1}$, ex: nunca gera $U\ U'$ ou $R\ R2$).
- **Regra 2**: Proíbe faces opostas intercaladas ($F_i \neq F_{i-2}$ quando $F_{i-1}$ for paralela a $F_i$, ex: nunca gera $U\ D\ U$).

---

## ⚡ Arquitetura e Otimizações de Performance

| Módulo | Antes | Otimização Implementada | Ganho |
| :--- | :--- | :--- | :--- |
| **Simulação do Cubo** | Objetos PyCuber dinâmicos | Array $O(1)$ de 54 inteiros pré-computados | **~675x mais rápido** |
| **Embaralhamento Base** | Recalculado do zero a cada indivíduo | Pré-aplicado uma única vez por execução | **Elimina redundância $O(N)$** |
| **Geração / Transição** | Loops `while True` com rejeições | Tabela de transições válidas $O(1)$ | **0 tentativas rejeitadas** |
| **Crossover & Reparo** | Filtro de listas e strings | Reparo linear $O(N)$ em uma única passada | **Execução instantânea** |
| **Escalabilidade CPU** | 1 único núcleo (GIL do Python) | `ProcessPoolExecutor` com Modelo de Ilhas | **Aproveita 100% dos núcleos** |

---

## 📂 Estrutura de Arquivos

```
.
├── controlador.py    # Servidor web Flask, API REST e threads assíncronas
├── geracao.py        # Motor do AG paralelo, Modelo de Ilhas e Busca Incremental
├── populacao.py      # Geração de cromossomos, tabelas O(1) e embaralhador WCA
├── pontuacao.py      # Motor de permutação dos 54 adesivos e cálculo de fitness
├── cruzamento.py     # Operador de recombinação e reparo linear O(N)
├── mutacao.py        # Operador de mutação com preservação de validade
├── index.html        # Interface gráfica web 3D interativa (Three.js)
└── README.md         # Documentação do projeto
```

---

## 🛠 Instalação e Execução

### Pré-requisitos
- Python 3.8 ou superior instalado.

### 1. Clonar o repositório
```bash
git clone https://github.com/luiz0067yahoo/python-algoritmo-genetico-inteligencia-artificial-cubo-de-rubik.git
cd python-algoritmo-genetico-inteligencia-artificial-cubo-de-rubik
```

### 2. Instalar dependências
```bash
pip install flask flask-cors pycuber
```

### 3. Iniciar o servidor
```bash
python controlador.py
```

### 4. Acessar a aplicação
Abra o navegador em:
```
http://localhost:5000
```

---

## 🎮 Interface Gráfica 3D

A interface web desenvolvida com Three.js oferece:
- **Cubo 3D com Física e Rotações Suaves**: Controle de rotação livre com OrbitControls e atalhos de teclado (`U, D, F, B, R, L` + `Shift` para anti-horário e `Alt` para giros duplos).
- **Planificação 2D em Tempo Real**: Visualização plana das 6 faces simultaneamente.
- **Painel de Monitoramento ao Vivo**: Acompanhamento de tempo decorrido, taxa de indivíduos avaliados, logs do terminal e gráfico de progresso.
- **Execução Automática da Solução**: Ao encontrar a solução, o cubo é automaticamente animado e finalizado no estado $54/54$.
- **Botões de Replay**: Permitem re-executar a solução no Cubo 3D ou re-embaralhar para novos experimentos.

---

## 📡 Documentação da API REST

### `POST /iniciar_solucao`
Inicia a resolução assíncrona com o Algoritmo Genético em background.

**Payload JSON:**
```json
{
  "embaralhamento": ["R", "U", "R'", "U'", "F'", "U", "F"],
  "porcentagem_mutacao": 0.05,
  "porcentagem_cruzamento": 0.70,
  "porcentagem_selecao": 0.50,
  "quantidade_geracoes": 2000,
  "quantidade_individuos_inicial": 1000,
  "tamanho_minimo": 1,
  "tamanho_maximo": 54,
  "intervalo_ciclo": 500
}
```

**Resposta:**
```json
{
  "sucesso": true,
  "session_id": "9b188127-75f3-4825-95c4-dec0d8fbcda5",
  "status": "executando",
  "mensagem": "Processamento do Algoritmo Genético iniciado com sucesso."
}
```

---

### `GET /status/<session_id>`
Retorna o snapshot das métricas em tempo real da sessão.

**Resposta:**
```json
{
  "status": "concluido",
  "geracao_atual": 1,
  "total_geracoes": 2000,
  "individuos_avaliados": 2906,
  "melhor_score": 54,
  "melhor_solucao": ["R", "U'", "R'"],
  "melhor_solucao_str": "R U' R'",
  "tempo_decorrido": 0.026
}
```

---

### `GET /gerar_embaralhamento_wca?tamanho=25`
Gera uma sequência de embaralhamento oficial no padrão da World Cube Association.

**Resposta:**
```json
{
  "sucesso": true,
  "embaralhamento": ["B2", "U2", "R", "D'", "L'", "R", "F2", "L", "R'", "U2", "D'", "R'", "L", "F", "D'", "F2", "D'", "U'", "R", "L", "B", "L2", "D2", "L2", "U2"],
  "embaralhamento_str": "B2 U2 R D' L' R F2 L R' U2 D' R' L F D' F2 D' U' R L B L2 D2 L2 U2",
  "tamanho": 25
}
```

---

## 📊 Métricas de Desempenho

```
Benchmark de Avaliações por Segundo:
--------------------------------------------------
Versão Original (PyCuber):           311 evals/segundo
Versão Otimizada (Single-Core):   90.579 evals/segundo (+290x)
Versão Paralela (Multi-Core):    418.359 evals/segundo (+1.345x)
--------------------------------------------------
```

---

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo de licença para mais informações.
