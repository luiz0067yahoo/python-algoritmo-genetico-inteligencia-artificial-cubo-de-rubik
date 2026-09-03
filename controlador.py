# ==============================================================================
# CONTROLADOR.PY - SERVIDOR WEB FLASK, GERENCIAMENTO DE SESSÕES E API REST
# ==============================================================================
# Este módulo implementa o servidor web em Flask que atua como ponte entre a
# interface gráfica 3D (Three.js no navegador) e o Algoritmo Genético em Python.
#
# Principais Responsabilidades:
# 1. Servir a interface web principal ('/').
# 2. Iniciar execuções assíncronas do AG em background ('/iniciar_solucao').
# 3. Fornecer snapshots de métricas e progresso em tempo real ('/status/<session_id>').
# 4. Permitir cancelamento imediato de sessões ativas ('/cancelar_solucao').
# 5. Gerar sequências oficiais de embaralhamento WCA ('/gerar_embaralhamento_wca').
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
        # Executa o resolvedor incremental de alta performance
        resultado = resolver_cubo_incremental(
            embaralhamento=payload.get('embaralhamento', []),
            porcentagem_mutacao=payload.get('porcentagem_mutacao', 0.05),
            porcentagem_cruzamento=payload.get('porcentagem_cruzamento', 0.70),
            porcentagem_selecao=payload.get('porcentagem_selecao', 0.50),
            quantidade_geracoes=payload.get('quantidade_geracoes', 2000),
            quantidade_individuos_inicial=payload.get('quantidade_individuos_inicial', 1000),
            tamanho_minimo=payload.get('tamanho_minimo', 1),
            tamanho_maximo=payload.get('tamanho_maximo', 54),
            intervalo_ciclo=payload.get('intervalo_ciclo', 500),
            callback_progresso=callback,
            is_cancelled=is_cancelled,
        )

        # Atualiza a sessão com os resultados finais
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
    }

    info_hardware = obter_informacoes_hardware()

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
            'melhor_score': 0,
            'melhor_solucao': [],
            'melhor_solucao_str': '',
            'hardware': info_hardware,
            'mensagens': [f"[{time.strftime('%H:%M:%S')}] Sessão iniciada ({info_hardware['cpu_nome']} - {info_hardware['threads_totais']} threads)."],
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
    """Retorna as especificações de hardware da máquina atual (processador, núcleos e threads)."""
    return jsonify(obter_informacoes_hardware()), 200


@app.route('/status', methods=['GET'])
@app.route('/status/<session_id>', methods=['GET'])
@app.route('/progresso', methods=['GET'])
@app.route('/progresso/<session_id>', methods=['GET'])
def obter_status(session_id=None):
    """
    Retorna o snapshot das métricas de progresso da sessão em tempo real:
    - Geração atual e total de gerações
    - Total de indivíduos avaliados
    - Melhor score atingido (0 a 54)
    - Melhor solução encontrada até o momento
    - Logs textuais do terminal
    - Tempo decorrido de execução
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
        snapshot['tempo_decorrido'] = round(time.time() - snapshot.get('timestamp_inicio', time.time()), 2)
        return jsonify(snapshot), 200


@app.route('/cancelar_solucao', methods=['POST'])
def cancelar_solucao():
    """Cancela a execução da sessão ativa solicitada pelo usuário."""
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
    Gera e retorna uma sequência de embaralhamento oficial no padrão da World Cube Association (WCA).
    Por padrão retorna 25 movimentos válidos sem redundâncias ou cancelamentos.
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
    """Rota legada síncrona mantida para compatibilidade com versões anteriores."""
    return iniciar_solucao()


if __name__ == '__main__':
    # Inicializa o servidor web local na porta 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
