# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 07:13:03 2022

@author: A302
"""

import numpy as np 
#and:
#entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#saidas = np.array([0, 0, 0, 1])
#or
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 1])
#XOR  não funciona pq não é linearmente separável
#entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#saidas = np.array([0, 1, 1, 0])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

def stepfunction(soma):
    if (soma >= 1):
        return 1
    else:
        return 0
    
    
def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepfunction(s)

def treinar():
    erroTotal = 1
    while  (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print("Peso atualizado " + str(pesos[j]))
        print('Total de erros ' + str(erroTotal))
treinar()
print("Rede neural trainada")
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
