# Machine-Learning

Como ando desenvolvendo diversos modelos de aprendizado de máquina, decidi compartilhar um pouco deles por aqui! 
São dois tipos de machine learning até agora:

## Redes Neurais:
 - rede_neural_breno_cifar10.py
Usando o conceito de rede neural densa, neural network fully connected. Foi carregado o conjunto de imagens do dataset CIFAR-10, convertido para escalas de cinza, normalização das imagens, convertendo os rótulos para o formato one-hot, criando o modelo de uma rede neural densa com uma camada de flatten e duas camadas densas, compila o modelo usando o otimizador Adam, o modelo é treinado por 5 épocas, avalia a precisão do conjunto de testes, faz previsões e plota algumas imagens com rótulos verdadeiros e preditos e calcula e plota a matriz de confusão para visualizar o desempenho do modelo.
 - cifar10.py
Treino de uma rede neural convulacional (CNN), onde classifica imagens do conjunto de dados, dataset, CIFAR-10. É feita toda parte de divisão do dataset, para treino e teste, normalização das imagens, criação da CNN com 3 camadas convolucionais seguidas por camadas de pooling e duas camadas densas, compilação do modelo usando o otimizador Adam, o treinamento feito por 5 épocas, avaliação da precisão do conjunto de testes e por último, a visualização da arquitetura da rede, usando o VisualKeras.
