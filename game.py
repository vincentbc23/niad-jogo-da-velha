import numpy as np
import math


class jogoDaVelha:
    def __init__(self):
        self.contagem_linhas = 3
        self.contagem_colunas = 3
        self.tamanho_acao = self.contagem_linhas * self.contagem_colunas

    def pega_inicial(self):
        return np.zeros((self.contagem_linhas, self.contagem_colunas))

    def proximo(self, estado, acao, jogador):
        linha = acao // self.contagem_colunas
        coluna = acao % self.contagem_colunas
        estado[linha, coluna] = jogador
        return estado

    def pega_movimento_valido(self, estado):
        return (estado.reshape(-1) == 0).astype(np.uint8)

    def verifica_vitoria(self, estado, acao):
        if acao is None:
            return False
        linha = acao // self.contagem_colunas
        coluna = acao % self.contagem_colunas
        jogador = estado[linha, coluna]
        return (
            np.sum(estado[linha, :]) == jogador * self.contagem_colunas
            or np.sum(estado[:, coluna]) == jogador * self.contagem_linhas
            or np.sum(np.diag(estado)) == jogador * self.contagem_linhas
            or np.sum(np.diag(np.flip(estado, axis=0))) == jogador * self.contagem_linhas
        )

    def pega_valor_e_termina(self, estado, acao):
        if self.verifica_vitoria(estado, acao):
            return 1, True
        if np.sum(self.pega_movimento_valido(estado)) == 0:
            return 0, True
        return 0, False

    def oponente(self, jogador):
        return -jogador

    def valor_oponente(self, valor):
        return -valor

    def muda_perspectiva(self, estado, jogador):
        return estado * jogador


class No:
    def __init__(self, game, args, estado, pai=None, acao=None):
        self.game = game
        self.args = args
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.filho = []
        self.movimentos_de_expansao = game.pega_movimento_valido(estado)
        self.count_visitados = 0
        self.valor_soma = 0

    def expandido_ao_maximo(self):
        return np.sum(self.movimentos_de_expansao) == 0 and len(self.filho) > 0

    def select(self):
        melhor_filho = None
        melhor_ucb = -np.inf
        for filho in self.filho:
            ucb = self.pega_ucb(filho)
            if ucb > melhor_ucb:
                melhor_filho = filho
                melhor_ucb = ucb
        return melhor_filho

    def pega_ucb(self, filho):
        valor_q = 1 - ((filho.valor_soma / filho.count_visitados) + 1) / 2
        return valor_q + self.args['C'] * math.sqrt(math.log(self.count_visitados) / filho.count_visitados)

    def expansao(self):
        acao = np.random.choice(np.where(self.movimentos_de_expansao == 1)[0])
        self.movimentos_de_expansao[acao] = 0
        estado_filho = self.estado.copy()
        estado_filho = self.game.proximo(estado_filho, acao, 1)
        estado_filho = self.game.muda_perspectiva(estado_filho, jogador=-1)
        filho = No(self.game, self.args, estado_filho, self, acao)
        self.filho.append(filho)
        return filho

    def simulacao(self):
        valor, eh_terminal = self.game.pega_valor_e_termina(self.estado, self.acao)
        valor = self.game.valor_oponente(valor)
        if eh_terminal:
            return valor
        estado_aleatorio = self.estado.copy()
        jogador_aleatorio = 1
        while True:
            movimento_valido = self.game.pega_movimento_valido(estado_aleatorio)
            acao = np.random.choice(np.where(movimento_valido == 1)[0])
            estado_aleatorio = self.game.proximo(estado_aleatorio, acao, jogador_aleatorio)
            valor, eh_terminal = self.game.pega_valor_e_termina(estado_aleatorio, acao)
            if eh_terminal:
                if jogador_aleatorio == -1:
                    valor = self.game.valor_oponente(valor)
                return valor
            jogador_aleatorio = self.game.oponente(jogador_aleatorio)

    def volta(self, valor):
        self.valor_soma += valor
        self.count_visitados += 1
        valor = self.game.valor_oponente(valor)
        if self.pai is not None:
            self.pai.volta(valor)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, estado):
        raiz = No(self.game, self.args, estado)
        for _ in range(self.args['num_searches']):
            no = raiz
            while no.expandido_ao_maximo():
                no = no.select()
            valor, eh_terminal = self.game.pega_valor_e_termina(no.estado, no.acao)
            valor = self.game.valor_oponente(valor)
            if not eh_terminal:
                no = no.expansao()
                valor = no.simulacao()
            no.volta(valor)
        probabilidade_de_acao = np.zeros(self.game.tamanho_acao)
        for filho in raiz.filho:
            probabilidade_de_acao[filho.acao] = filho.count_visitados
        probabilidade_de_acao /= np.sum(probabilidade_de_acao)
        return probabilidade_de_acao
