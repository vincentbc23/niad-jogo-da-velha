import os
import numpy as np
import secrets
from flask import Flask, render_template, jsonify, request, session
from game import jogoDaVelha, MCTS

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

game = jogoDaVelha()
args = {'C': 1.41, 'num_searches': 200}
mcts = MCTS(game, args)


def pega_celulas_vencedoras(estado, acao):
    if acao is None:
        return []
    n = game.contagem_linhas
    linha   = acao // game.contagem_colunas
    coluna  = acao %  game.contagem_colunas
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


def gs_get():
    estado_list = session.get('estado')
    return {
        'estado':      np.array(estado_list) if estado_list else None,
        'player_piece': session.get('player_piece', 1),
        'game_over':   session.get('game_over', False),
        'winner':      session.get('winner', None),
    }


def gs_set(estado, player_piece, game_over, winner):
    session['estado']       = estado.tolist() if estado is not None else None
    session['player_piece'] = player_piece
    session['game_over']    = game_over
    session['winner']       = winner


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/new_game', methods=['POST'])
def new_game():
    data         = request.json
    player_piece = data.get('player', 1)
    estado       = game.pega_inicial()
    gs_set(estado, player_piece, False, None)
    winning_cells = []

    if player_piece == -1:
        estado_neutro = game.muda_perspectiva(estado, 1)
        prob  = mcts.search(estado_neutro)
        acao  = int(np.argmax(prob))
        estado = game.proximo(estado, acao, 1)
        valor, eh_terminal = game.pega_valor_e_termina(estado, acao)
        if eh_terminal:
            gs_set(estado, player_piece, True, 1 if valor == 1 else 0)
            winning_cells = pega_celulas_vencedoras(estado, acao)
        else:
            gs_set(estado, player_piece, False, None)

    gs = gs_get()
    return jsonify({
        'state':        gs['estado'].tolist(),
        'game_over':    gs['game_over'],
        'winner':       gs['winner'],
        'winning_cells': winning_cells,
    })


@app.route('/move', methods=['POST'])
def move():
    gs = gs_get()
    if gs['estado'] is None:
        return jsonify({'error': 'Nenhum jogo em andamento'}), 400
    if gs['game_over']:
        return jsonify({'error': 'Jogo encerrado'}), 400

    acao         = request.json.get('action')
    estado       = gs['estado']
    player_piece = gs['player_piece']

    if game.pega_movimento_valido(estado)[acao] == 0:
        return jsonify({'error': 'Movimento inválido'}), 400

    estado = game.proximo(estado, acao, player_piece)
    valor, eh_terminal = game.pega_valor_e_termina(estado, acao)

    if eh_terminal:
        winner = player_piece if valor == 1 else 0
        gs_set(estado, player_piece, True, winner)
        winning_cells = pega_celulas_vencedoras(estado, acao) if valor == 1 else []
        return jsonify({'state': estado.tolist(), 'game_over': True,
                        'winner': winner, 'winning_cells': winning_cells})

    ai_piece      = game.oponente(player_piece)
    estado_neutro = game.muda_perspectiva(estado, ai_piece)
    prob          = mcts.search(estado_neutro)
    ai_acao       = int(np.argmax(prob))
    estado        = game.proximo(estado, ai_acao, ai_piece)
    valor, eh_terminal = game.pega_valor_e_termina(estado, ai_acao)

    winning_cells = []
    winner = None
    if eh_terminal:
        winner = ai_piece if valor == 1 else 0
        winning_cells = pega_celulas_vencedoras(estado, ai_acao) if valor == 1 else []

    gs_set(estado, player_piece, eh_terminal, winner)
    return jsonify({'state': estado.tolist(), 'game_over': eh_terminal,
                    'winner': winner, 'winning_cells': winning_cells})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
