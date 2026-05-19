from flask import Flask, render_template, jsonify, request
import numpy as np
import secrets
from game import jogoDaVelha, MCTS

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

game = jogoDaVelha()
args = {'C': 1.41, 'num_searches': 500}
mcts = MCTS(game, args)

game_state = {
    'estado': None,
    'player_piece': 1,
    'game_over': False,
    'winner': None,
}


def pega_celulas_vencedoras(estado, acao):
    if acao is None:
        return []
    n = game.contagem_linhas
    linha = acao // game.contagem_colunas
    coluna = acao % game.contagem_colunas
    jogador = estado[linha, coluna]

    if np.sum(estado[linha, :]) == jogador * n:
        return [linha * n + c for c in range(n)]
    if np.sum(estado[:, coluna]) == jogador * n:
        return [r * n + coluna for r in range(n)]
    if np.sum(np.diag(estado)) == jogador * n:
        return [i * n + i for i in range(n)]
    if np.sum(np.diag(np.flip(estado, axis=0))) == jogador * n:
        return [i * n + (n - 1 - i) for i in range(n)]
    return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/new_game', methods=['POST'])
def new_game():
    data = request.json
    player_piece = data.get('player', 1)

    game_state['estado'] = game.pega_inicial()
    game_state['player_piece'] = player_piece
    game_state['game_over'] = False
    game_state['winner'] = None

    winning_cells = []

    # Se o jogador escolheu O (-1), a IA (X, peça=1) joga primeiro
    if player_piece == -1:
        estado = game_state['estado']
        estado_neutro = game.muda_perspectiva(estado, 1)
        prob = mcts.search(estado_neutro)
        acao = int(np.argmax(prob))
        game_state['estado'] = game.proximo(estado, acao, 1)
        valor, eh_terminal = game.pega_valor_e_termina(game_state['estado'], acao)
        if eh_terminal:
            game_state['game_over'] = True
            game_state['winner'] = 1 if valor == 1 else 0
            winning_cells = pega_celulas_vencedoras(game_state['estado'], acao)

    return jsonify({
        'state': game_state['estado'].tolist(),
        'game_over': game_state['game_over'],
        'winner': game_state['winner'],
        'winning_cells': winning_cells,
    })


@app.route('/move', methods=['POST'])
def move():
    data = request.json
    acao = data.get('action')

    if game_state['estado'] is None:
        return jsonify({'error': 'Nenhum jogo em andamento'}), 400
    if game_state['game_over']:
        return jsonify({'error': 'Jogo encerrado'}), 400

    estado = game_state['estado']
    player_piece = game_state['player_piece']

    if game.pega_movimento_valido(estado)[acao] == 0:
        return jsonify({'error': 'Movimento inválido'}), 400

    # Jogada do humano
    estado = game.proximo(estado, acao, player_piece)
    game_state['estado'] = estado
    valor, eh_terminal = game.pega_valor_e_termina(estado, acao)

    if eh_terminal:
        game_state['game_over'] = True
        game_state['winner'] = player_piece if valor == 1 else 0
        winning_cells = pega_celulas_vencedoras(estado, acao) if valor == 1 else []
        return jsonify({
            'state': estado.tolist(),
            'game_over': True,
            'winner': game_state['winner'],
            'winning_cells': winning_cells,
        })

    # Jogada da IA
    ai_piece = game.oponente(player_piece)
    estado_neutro = game.muda_perspectiva(estado, ai_piece)
    prob = mcts.search(estado_neutro)
    ai_acao = int(np.argmax(prob))
    estado = game.proximo(estado, ai_acao, ai_piece)
    game_state['estado'] = estado
    valor, eh_terminal = game.pega_valor_e_termina(estado, ai_acao)

    winning_cells = []
    if eh_terminal:
        game_state['game_over'] = True
        game_state['winner'] = ai_piece if valor == 1 else 0
        winning_cells = pega_celulas_vencedoras(estado, ai_acao) if valor == 1 else []

    return jsonify({
        'state': estado.tolist(),
        'game_over': game_state['game_over'],
        'winner': game_state['winner'],
        'winning_cells': winning_cells,
    })


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
