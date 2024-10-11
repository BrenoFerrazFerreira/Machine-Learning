# Dependencias

!pip install tensorflow visualkeras

# Bibliotecas
import visualkeras
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy

import matplotlib as plt
import numpy as np

# Carregar dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalização do dataset
x_train, x_test = train_images / 255.0, test_images / 255.0

# manter os rótlos sem one-hot
y_train = train_labels.flatten()
y_test = test_labels.flatten()

# Criaçâo do modelo

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Modtra um resumo da rede
model.summary()

# Compilar o modelo
model.compile(
    optimizer='adam',
    loss=sparse_categorical_crossentropy,
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# Avaliar o modelo no conjunto de testes
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}" )

# Visualizar a arquiterura da rede
visualkeras.layered_view(model, legend=True)