# ==============================================================================
# CONTROLADOR.PY - SERVIDOR WEB FLASK, GERENCIAMENTO DE SESSÕES E API REST
# ==============================================================================
# Este módulo implementa a camada controladora e servidora do RubikLab AI.
# Atua como a ponte de comunicação bidirecional de baixa latência entre a interface
# gráfica 3D no navegador (Three.js/WebGL) e o motor computacional de Algoritmos
# Genéticos de Alta Performance (Multi-Core CPU e WebGPU/Vulkan).
#
# Arquitetura e Ciclo de Vida do Sistema:
# 1. Requisição Inicial do Cliente:
#    - O navegador requisita 'GET /' e recebe o index.html com o renderizador 3D.
#    - O navegador consulta 'GET /info_hardware' para exibir o banner com as
#      especificações de CPU (AMD Ryzen 7 PRO 8700GE) e GPU (AMD Radeon 780M Graphics).
#
# 2. Despacho Assíncrono de Resolução:
#    - Ao clicar em "Iniciar Solução", o frontend envia 'POST /iniciar_solucao'.
#    - O controlador gera um UUID único (`session_id`), instancia o snapshot inicial
#      e dispara uma daemon thread de background (`worker_solucao`), liberando a
#      resposta HTTP instantaneamente em < 5ms sem travar a interface.
#
# 3. Telemetria e Polling em Tempo Real (1 em 1 segundo):
#    - A cada 1000ms, o frontend faz 'GET /status/<session_id>' para obter:
#      * Progresso do Algoritmo Genético (geração atual, indivíduos avaliados, taxa/s).
#      * Decomposição do Score em 6 Pilares (pos/ori cantos, pos/ori arestas, F2L+Cruz, tamanho).
#      * Melhor sequência de movimentos encontrada até o momento.
#      * Tempo decorrido formatado no padrão canônico HH:MM:SS (00:00:00).
#
# 4. Encerramento e Animação:
#    - Quando o AG atinge 54/54 adesivos (ou esgota os ciclos), o status torna-se 'concluido'.
#    - O frontend recebe o resultado final, ativa o canvas-confetti e executa os giros
#      passo a passo no Cubo 3D interativo.
#
# Concorrência e Thread-Safety:
# - Mutex Lock (`LOCK_SESSAO`) garante consistência atômica no dicionário global de sessões.
# - Flag de cancelamento atômica (`cancelado: True`) permite abortar execuções a qualquer instante.
# ==============================================================================

import threading
import time
import uuid
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from geracao import resolver_cubo_incremental, obter_informacoes_hardware

# Inicialização da aplicação Flask
app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = 'rubik_secret_ai_optimizer_key_2026'

# Habilita CORS para permitir requisições de diferentes origens se necessário
CORS(app, origins='*')

# Dicionário em memória para rastrear o estado e progresso de cada sessão ativa
# Chave: session_id (str) -> Valor: dict contendo métricas, histórico e status
SESSOES_PROGRESSO = {}

# Mutex Lock para garantir thread safety no acesso concorrente às variáveis de sessão
LOCK_SESSAO = threading.Lock()


def formatar_tempo_hhmmss(segundos):
    """
    Formata um valor de tempo em segundos para a notação canônica HH:MM:SS.
    
    Exemplos:
        formatar_tempo_hhmmss(75)    -> "00:01:15"
        formatar_tempo_hhmmss(3665)  -> "01:01:05"
        formatar_tempo_hhmmss(86400) -> "24:00:00"

    Parâmetros:
        segundos (float | int): Quantidade de segundos decorridos.

    Retorno:
        str: String formatada com dois dígitos para horas, minutos e segundos.
    """
    seg = max(0, int(segundos or 0))
    horas = seg // 3600
    minutos = (seg % 3600) // 60
    segs = seg % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def atualizar_sessao(session_id, dados):
    """
    Atualiza o estado de progresso de uma sessão ativa de forma thread-safe.

    Parâmetros:
        session_id (str): Identificador único da sessão.
        dados (dict): Dicionário contendo os campos e métricas a serem atualizados.
    """
    with LOCK_SESSAO:
        if session_id in SESSOES_PROGRESSO:
            estado = SESSOES_PROGRESSO[session_id]
            for chave, valor in dados.items():
                if chave == 'mensagem' and valor:
                    mensagens = estado.get('mensagens', [])
                    timestamp = time.strftime('%H:%M:%S')
                    mensagens.append(f"[{timestamp}] {valor}")
                    # Limita o histórico a 150 mensagens para evitar consumo excessivo de memória
                    if len(mensagens) > 150:
                        mensagens = mensagens[-150:]
                    estado['mensagens'] = mensagens
                else:
                    estado[chave] = valor


def worker_solucao(session_id, payload):
    """
    Função executada em uma thread em segundo plano (background worker)
    para rodar o Algoritmo Genético sem travar o servidor web.

    Parâmetros:
        session_id (str): Identificador único da sessão.
        payload (dict): Parâmetros do AG (embaralhamento, taxas, gerações, etc.).
    """
    def callback(info):
        """Callback invocado pelo algoritmo genético a cada geração/ciclo."""
        atualizar_sessao(session_id, info)

    def is_cancelled():
        """Função para verificar se o usuário solicitou o cancelamento da sessão."""
        with LOCK_SESSAO:
            sess = SESSOES_PROGRESSO.get(session_id)
            return sess.get('cancelado', False) if sess else True

    try:
        # Executa a resolução evolutiva hierárquica baseada no Método de Jessica Fridrich (CFOP):
        # 1. Cruz (Cross na base D) -> 2. F2L (2 primeiras camadas) -> 3. OLL (orientação topo) -> 4. PLL (permutação final)
        # O callback despacha atualizações periódicas a cada 1.0 segundo com a Decomposição do Score completa.
        resultado = resolver_cubo_incremental(
            embaralhamento=payload.get('embaralhamento', []),
            porcentagem_mutacao=payload.get('porcentagem_mutacao', 0.05),
            porcentagem_cruzamento=payload.get('porcentagem_cruzamento', 0.70),
            porcentagem_selecao=payload.get('porcentagem_selecao', 0.50),
            quantidade_geracoes=payload.get('quantidade_geracoes', 2000),
            quantidade_individuos_inicial=payload.get('quantidade_individuos_inicial', 1000),
            tamanho_minimo=payload.get('tamanho_minimo', 1),
            tamanho_maximo=payload.get('tamanho_maximo', 54),
            intervalo_ciclo=payload.get('intervalo_ciclo', 100),
            modo_hardware=payload.get('modo_hardware', 'cpu+gpu'),
            callback_progresso=callback,
            is_cancelled=is_cancelled,
        )

        # Atualiza a sessão com os resultados finais
        if resultado and isinstance(resultado, dict):
            if 'embaralhamento' in resultado and isinstance(resultado['embaralhamento'], list):
                resultado['embaralhamento'] = [str(m) for m in resultado['embaralhamento']]
            if 'solucao' in resultado and isinstance(resultado['solucao'], list):
                resultado['solucao'] = [str(m) for m in resultado['solucao']]

        with LOCK_SESSAO:
            if session_id in SESSOES_PROGRESSO:
                if SESSOES_PROGRESSO[session_id].get('cancelado'):
                    SESSOES_PROGRESSO[session_id]['status'] = 'cancelado'
                    SESSOES_PROGRESSO[session_id]['mensagem'] = 'Execução cancelada pelo usuário.'
                else:
                    SESSOES_PROGRESSO[session_id]['status'] = 'concluido'
                    SESSOES_PROGRESSO[session_id]['resultado_final'] = resultado
                    SESSOES_PROGRESSO[session_id]['melhor_score'] = resultado.get('score', 0)
                    SESSOES_PROGRESSO[session_id]['melhor_solucao'] = resultado.get('solucao', [])
                    SESSOES_PROGRESSO[session_id]['melhor_solucao_str'] = resultado.get('solucao_str', '')
                    if resultado and 'detalhes_fitness' in resultado:
                        SESSOES_PROGRESSO[session_id]['detalhes_fitness'] = resultado['detalhes_fitness']
                    SESSOES_PROGRESSO[session_id]['mensagem'] = resultado.get('mensagem', 'Processamento finalizado.')

    except Exception as e:
        # Registra erros imprevistos de execução
        with LOCK_SESSAO:
            if session_id in SESSOES_PROGRESSO:
                SESSOES_PROGRESSO[session_id]['status'] = 'erro'
                SESSOES_PROGRESSO[session_id]['erro'] = str(e)
                SESSOES_PROGRESSO[session_id]['mensagem'] = f"Erro na execução: {str(e)}"


# ==============================================================================
# ROTAS HTTP / API REST
# ==============================================================================

@app.route('/')
def index():
    """Renderiza a interface web 3D interativa do Cubo de Rubik."""
    return render_template('index.html')


@app.route('/iniciar_solucao', methods=['POST'])
@app.route('/solucionar', methods=['POST'])
def iniciar_solucao():
    """
    Inicia a execução assíncrona do Algoritmo Genético em background.
    Cria a sessão e retorna imediatamente o session_id para acompanhamento via polling.
    """
    dados = request.get_json(silent=True) or {}

    session_id = dados.get('session_id') or str(uuid.uuid4())
    session['session_id'] = session_id

    # Normalização e validação dos parâmetros recebidos
    embaralhamento = dados.get('embaralhamento', [])
    if isinstance(embaralhamento, str):
        embaralhamento = [m for m in embaralhamento.split() if m.strip()]

    modo_hw = str(dados.get('modo_hardware', 'cpu+gpu')).lower().strip()
    if modo_hw not in ('cpu', 'gpu', 'cpu+gpu'):
        modo_hw = 'cpu+gpu'

    payload = {
        'embaralhamento': embaralhamento,
        'porcentagem_mutacao': float(dados.get('porcentagem_mutacao', 0.05)),
        'porcentagem_cruzamento': float(dados.get('porcentagem_cruzamento', 0.70)),
        'porcentagem_selecao': float(dados.get('porcentagem_selecao', 0.50)),
        'quantidade_geracoes': int(dados.get('quantidade_geracoes', 2000)),
        'quantidade_individuos_inicial': int(dados.get('quantidade_individuos_inicial', 1000)),
        'tamanho_minimo': max(1, int(dados.get('tamanho_minimo', 1))),
        'tamanho_maximo': max(1, int(dados.get('tamanho_maximo', 54))),
        'intervalo_ciclo': max(1, int(dados.get('intervalo_ciclo', 500))),
        'modo_hardware': modo_hw,
    }

    info_hardware = obter_informacoes_hardware()

    from pontuacao import ESTADO_RESOLVIDO, aplicar_movimentos, calcular_score_estado
    st_init = aplicar_movimentos(ESTADO_RESOLVIDO, embaralhamento)
    _, det_init = calcular_score_estado(st_init, qtd_movimentos=0, retornar_detalhes=True)

    # Inicializa o snapshot de progresso da sessão
    with LOCK_SESSAO:
        SESSOES_PROGRESSO[session_id] = {
            'status': 'executando',
            'session_id': session_id,
            'etapa': 'Iniciando Algoritmo Genético...',
            'operacao': 'Criando indivíduos',
            'tamanho_atual': payload['tamanho_minimo'],
            'tamanho_cromossomo': payload['tamanho_minimo'],
            'cromossomos_populacao': payload['quantidade_individuos_inicial'],
            'cromossomos_avaliados': 0,
            'geracao_atual': 0,
            'total_geracoes': payload['quantidade_geracoes'],
            'individuos_avaliados': 0,
            'melhor_score': det_init.get('adesivos_corretos', 0),
            'detalhes_fitness': det_init,
            'melhor_solucao': [],
            'melhor_solucao_str': '',
            'hardware': info_hardware,
            'mensagens': [f"[{time.strftime('%H:%M:%S')}] Sessão iniciada: {info_hardware['cpu_nome']} (16 threads) + GPU: {info_hardware.get('gpu_nome', 'AMD Radeon 780M')} | Score Inicial: {det_init.get('adesivos_corretos', 0)}/54 ({det_init.get('score_total', 0):.1f} pts)"],
            'resultado_final': None,
            'cancelado': False,
            'timestamp_inicio': time.time(),
        }

    # Dispara a thread em segundo plano (daemon thread)
    thread = threading.Thread(target=worker_solucao, args=(session_id, payload), daemon=True)
    thread.start()

    return jsonify({
        'sucesso': True,
        'session_id': session_id,
        'status': 'executando',
        'hardware': info_hardware,
        'mensagem': 'Processamento do Algoritmo Genético iniciado com sucesso.'
    }), 200


@app.route('/info_hardware', methods=['GET'])
def info_hardware_endpoint():
    """
    Endpoint de Telemetria de Hardware (GET /info_hardware).
    
    Retorna as especificações de hardware detectadas em tempo de execução:
    - CPU: Modelo, contagem de núcleos físicos e lógicos (ex: AMD Ryzen 7 PRO 8700GE - 16 threads).
    - GPU: Nome do adaptador Vulkan/Direct3D 12 (ex: AMD Radeon 780M Graphics) e taxa estimada.
    - Modo operacional: Híbrido CPU+GPU, GPU dedicada ou CPU multi-processos.
    
    Retorno (JSON 200):
        dict contendo chaves: 'cpu_nome', 'threads_totais', 'threads_utilizadas',
                              'gpu_nome', 'gpu_disponivel', 'gpu_taxa', 'modo'.
    """
    return jsonify(obter_informacoes_hardware()), 200


@app.route('/status', methods=['GET'])
@app.route('/status/<session_id>', methods=['GET'])
@app.route('/progresso', methods=['GET'])
@app.route('/progresso/<session_id>', methods=['GET'])
def obter_status(session_id=None):
    """
    Endpoint de Polling de Telemetria em Tempo Real (GET /status/<session_id>).
    
    Invocado periodicamente pelo frontend (a cada 1000ms) para obter o snapshot
    atômico do estado de evolução do Algoritmo Genético.
    
    Métricas e Dados Fornecidos:
    - 'status': 'executando', 'concluido', 'cancelado' ou 'erro'.
    - 'geracao_atual' e 'total_geracoes': Progresso das iterações evolucionárias.
    - 'individuos_avaliados': Volume cumulativo de cromossomos simulados.
    - 'melhor_score': Quantidade de adesivos corretos (0 a 54).
    - 'detalhes_fitness': Decomposição detalhada do Score em 6 pilares:
        * pos_cantos, ori_cantos, pos_arestas, ori_arestas, pares_f2l_cruz, penalidade.
    - 'melhor_solucao': Lista de movimentos WCA da melhor sequência encontrada.
    - 'tempo_decorrido': Segundos corridos desde o início.
    - 'tempo_decorrido_formatado': Tempo no padrão canônico HH:MM:SS (ex: "00:02:14").
    - 'mensagens': Últimas 150 linhas de log do terminal em memória.
    
    Parâmetros da URL ou Query:
        session_id (str, opcional): UUID da sessão ativa.
        
    Retorno (JSON):
        200: Snapshot completo do progresso.
        400: Erro se nenhum session_id for informado.
        404: Erro se a sessão expirou ou não existe.
    """
    if not session_id:
        session_id = request.args.get('session_id') or session.get('session_id')

    if not session_id:
        return jsonify({'erro': 'session_id não informado'}), 400

    with LOCK_SESSAO:
        estado = SESSOES_PROGRESSO.get(session_id)
        if not estado:
            return jsonify({
                'status': 'inexistente',
                'session_id': session_id,
                'mensagem': 'Sessão não encontrada.'
            }), 404

        # Retorna cópia com cálculo do tempo decorrido
        snapshot = dict(estado)
        t_decorrido = round(time.time() - snapshot.get('timestamp_inicio', time.time()), 2)
        snapshot['tempo_decorrido'] = t_decorrido
        snapshot['tempo_decorrido_formatado'] = formatar_tempo_hhmmss(t_decorrido)
        return jsonify(snapshot), 200


@app.route('/cancelar_solucao', methods=['POST'])
def cancelar_solucao():
    """
    Endpoint de Cancelamento Assíncrono (POST /cancelar_solucao).
    
    Sinaliza à thread de background e aos processos de ilha que a execução
    deve ser abortada imediatamente. A flag 'cancelado' é lida a cada ciclo
    pelo callback 'is_cancelled()'.
    
    Payload (JSON):
        {"session_id": "<uuid-da-sessao>"}
        
    Retorno (JSON):
        200: Confirmação de encerramento.
        400: Erro se session_id não for fornecido.
        404: Sessão não encontrada.
    """
    dados = request.get_json(silent=True) or {}
    session_id = dados.get('session_id') or session.get('session_id')

    if not session_id:
        return jsonify({'erro': 'session_id não informado'}), 400

    with LOCK_SESSAO:
        if session_id in SESSOES_PROGRESSO:
            SESSOES_PROGRESSO[session_id]['cancelado'] = True
            SESSOES_PROGRESSO[session_id]['status'] = 'cancelado'
            atualizar_sessao(session_id, {'mensagem': 'Solicitação de cancelamento recebida.'})
            return jsonify({'sucesso': True, 'mensagem': 'Execução cancelada com sucesso.'}), 200
        else:
            return jsonify({'erro': 'Sessão não encontrada'}), 404


@app.route('/gerar_embaralhamento_wca', methods=['GET', 'POST'])
def obter_embaralhamento_wca():
    """
    Endpoint Gerador de Scramble Oficial WCA (GET/POST /gerar_embaralhamento_wca).
    
    Gera uma sequência pseudoaleatória conforme o Regulamento Internacional da
    World Cube Association (WCA Artigo 12 / Regulação 4b):
    - Comprimento padrão de 25 movimentos canônicos.
    - Zero redundâncias consecutivas (ex: nunca gera U U').
    - Zero faces paralelas opostas canceladas (ex: nunca gera U D U).
    
    Parâmetros (Query string ou JSON):
        tamanho (int, opcional): Quantidade de movimentos desejada (padrão: 25, mín: 1, máx: 100).
        
    Retorno (JSON 200):
        {
            "sucesso": true,
            "embaralhamento": ["R", "U'", "F2", ...],
            "embaralhamento_str": "R U' F2 ...",
            "tamanho": 25
        }
    """
    from populacao import gerar_embaralhamento_wca
    dados = request.get_json(silent=True) or {}
    tamanho = int(request.args.get('tamanho') or dados.get('tamanho') or 25)
    tamanho = max(1, min(100, tamanho))
    scramble = gerar_embaralhamento_wca(tamanho)
    return jsonify({
        'sucesso': True,
        'embaralhamento': scramble,
        'embaralhamento_str': " ".join(scramble),
        'tamanho': len(scramble)
    }), 200


@app.route('/rodar_algoritmo_genetico', methods=['POST'])
def rodar_ag_legado():
    """Rota legada síncrona mantida para compatibilidade retroativa."""
    return iniciar_solucao()


if __name__ == '__main__':
    # Inicializa o servidor web local na porta 5000 com thread safety e estabilidade no Windows
    app.run(debug=False, use_reloader=False, threaded=True, host='0.0.0.0', port=5000)
