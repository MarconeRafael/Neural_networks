import numpy as np
#XOR
def sigmoid(soma):
    return 1/(1+ np.exp(-soma))
#a = sigmoind(-1.5)
#b = np.exp(0)

def sigmoidDerivada(sig):
    return sig * (1 - sig) 
#a = sigmoid(0.5)
#B = sigmoidDerivada(a)

entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

saidas = np.array([[0], [1], [1], [0]])

#pesos0 = np.array([[-0.424, -0.740,-0.961],
#                  [0.358, - 0.577, -0.469]])

#pesos1 = np.array([[-0.017], [-0.893], [0.148]])
pesos0 = 2*np.random.random((2,3)) - 1
pesos1 = 2*np.random.random((3,1)) - 1

epocas = 1000000

taxaAprendizagem = 0.4
momento = 1
for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    MediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: ", str(MediaAbsoluta))
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T 
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 *momento) + (pesosNovo0 *taxaAprendizagem)
    
    
    
    
    
    
    #ao atualizar os erros o objetivo é achar o erro minimo global. O fundo da tigela.
    #Gradiente é encontrar a combinação de pesos com menor erro possivel
    #Gradiente é calculado para saber quanto ajustar os pesos
    #calcula primeiro o delta da camada de saida e depois o delta da camada oculta
    #delta = erro * derivadasigmoid
    #backpropagation:
        # pesson+1 = (penon*momento)+(entrada*delta*taxa de aprendizagem)
        #o valor da camada oculta tbm é entrada
