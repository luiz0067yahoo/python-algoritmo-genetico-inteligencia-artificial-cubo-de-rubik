import threading
import time
import uuid
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from geracao import resolver_cubo_incremental

app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = 'rubik_secret_ai_optimizer_key_2026'
CORS(app, origins='*')

# Gerenciamento de progresso de sessões em memória com Lock
SESSOES_PROGRESSO = {}
LOCK_SESSAO = threading.Lock()


def atualizar_sessao(session_id, dados):
    """Atualiza o estado de progresso de uma sessão de forma segura."""
    with LOCK_SESSAO:
        if session_id in SESSOES_PROGRESSO:
            estado = SESSOES_PROGRESSO[session_id]
            for chave, valor in dados.items():
                if chave == 'mensagem' and valor:
                    mensagens = estado.get('mensagens', [])
                    timestamp = time.strftime('%H:%M:%S')
                    mensagens.append(f"[{timestamp}] {valor}")
                    # Mantém no máximo 150 mensagens no histórico da sessão
                    if len(mensagens) > 150:
                        mensagens = mensagens[-150:]
                    estado['mensagens'] = mensagens
                else:
                    estado[chave] = valor


def worker_solucao(session_id, payload):
    """Thread executada em segundo plano para o Algoritmo Genético."""
    def callback(info):
        atualizar_sessao(session_id, info)

    def is_cancelled():
        with LOCK_SESSAO:
            sess = SESSOES_PROGRESSO.get(session_id)
            return sess.get('cancelado', False) if sess else True

    try:
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
        with LOCK_SESSAO:
            if session_id in SESSOES_PROGRESSO:
                SESSOES_PROGRESSO[session_id]['status'] = 'erro'
                SESSOES_PROGRESSO[session_id]['erro'] = str(e)
                SESSOES_PROGRESSO[session_id]['mensagem'] = f"Erro na execução: {str(e)}"


@app.route('/')
def index():
    """Renderiza a interface web 3D do Cubo Mágico."""
    return render_template('index.html')


@app.route('/iniciar_solucao', methods=['POST'])
@app.route('/solucionar', methods=['POST'])
def iniciar_solucao():
    """
    Inicia a execução assíncrona do Algoritmo Genético em background,
    registrando a sessão e retornando o session_id para acompanhamento em tempo real via AJAX.
    """
    dados = request.get_json(silent=True) or {}

    session_id = dados.get('session_id') or str(uuid.uuid4())
    session['session_id'] = session_id

    # Normalização de parâmetros
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

    # Inicialização do estado de progresso da sessão
    with LOCK_SESSAO:
        SESSOES_PROGRESSO[session_id] = {
            'status': 'executando',
            'session_id': session_id,
            'etapa': 'Iniciando Algoritmo Genético...',
            'operacao': 'Criando indivíduos',
            'tamanho_atual': payload['tamanho_minimo'],
            'geracao_atual': 0,
            'total_geracoes': payload['quantidade_geracoes'],
            'individuos_avaliados': 0,
            'melhor_score': 0,
            'melhor_solucao': [],
            'melhor_solucao_str': '',
            'mensagens': [f"[{time.strftime('%H:%M:%S')}] Sessão iniciada. Parâmetros configurados."],
            'resultado_final': None,
            'cancelado': False,
            'timestamp_inicio': time.time(),
        }

    # Dispara a thread em segundo plano
    thread = threading.Thread(target=worker_solucao, args=(session_id, payload), daemon=True)
    thread.start()

    return jsonify({
        'sucesso': True,
        'session_id': session_id,
        'status': 'executando',
        'mensagem': 'Processamento do Algoritmo Genético iniciado com sucesso.'
    }), 200


@app.route('/status', methods=['GET'])
@app.route('/status/<session_id>', methods=['GET'])
@app.route('/progresso', methods=['GET'])
@app.route('/progresso/<session_id>', methods=['GET'])
def obter_status(session_id=None):
    """
    Retorna o snapshot das variáveis de sessão em tempo real:
    indivíduos avaliados, etapa atual de avaliação, geração atual, score e mensagens.
    Disponível via rota /status e /status/<session_id> consultada de 1 em 1 segundo pelo JavaScript.
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

        # Retorna cópia rasa do estado da sessão
        snapshot = dict(estado)
        snapshot['tempo_decorrido'] = round(time.time() - snapshot.get('timestamp_inicio', time.time()), 2)
        return jsonify(snapshot), 200



@app.route('/cancelar_solucao', methods=['POST'])
def cancelar_solucao():
    """Cancela a execução da sessão atual."""
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


@app.route('/rodar_algoritmo_genetico', methods=['POST'])
def rodar_ag_legado():
    """Rota legada síncrona mantida para compatibilidade direta."""
    return iniciar_solucao()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
