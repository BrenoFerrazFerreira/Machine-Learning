import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# def gray_scale(image_array):
#   w_red = 0.2989
#   w_green = 0.5870
#   w_blue = 0.114

#   height, width, _ = image_array.shape

#   image_grayscale = np.zeros((height, width), dtype=np.uint8)
#   for i in range(height):
#     for j in range(width):
#       r, g, b = image_array[i, j]
#       image_grayscale[i, j] = int(r * w_red  + g * w_green + b * w_blue)

#   return image_grayscale

def gray_scale(image_array):
  return np.dot(image_array[...,:3], [0.2989, 0.5870, 0.114])

# qtd, _, _, _ = train_images.shape
# train_images_gray = np.zeros((qtd, 32, 32), dtype=np.uint8)
# for i in range(qtd):
#   train_images_gray[i] = gray_scale(train_images[i])

train_images_gray = gray_scale(train_images)

# qtd, _, _, _ = test_images.shape
# test_images_gray = np.zeros((qtd, 32, 32), dtype=np.uint8)
# for i in range(qtd):
#   test_images_gray[i] = gray_scale(test_images[i])

test_images_gray = gray_scale(test_images)

train_images_gray[2]

# Copiar as imagens em RGB
train_images_rgb = train_images.copy()
test_images_rgb = test_images.copy()

# 2. Normalizar os dados
train_images = train_images_gray / 255.0
test_images = test_images_gray / 255.0

# 3. Transformar as labels em one-hot encoding
train_labels_onehot = to_categorical(train_labels, 10)
test_labels_onehot = to_categorical(test_labels, 10)

# 4. Criar o modelo da rede neural
model = Sequential()

# 5. Adicionar a camada de entrada e de flatten (achatamento)
model.add(Flatten(input_shape=(32, 32))) # 1024 neuronios na cama de entrada

# 6. Adicionar camadas ocultas densas (fully connected)
model.add(Dense(128, activation='relu'))

# 7. Adicionar a camada de saída
model.add(Dense(10, activation='softmax'))

# 8. Compilar o modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Images")
print("Train: ", train_images[0].shape)
print("Test: ", test_images[0].shape)

print("Labels")
print("Train: ", train_labels_onehot.shape)
print("Test: ", test_labels_onehot.shape)
# 0 = [0,0,0,0,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0,0]
# 9 = [0,0,0,0,0,0,0,0,0,1]

# 9. Treinar o modelo
model.fit(train_images, train_labels_onehot, epochs=5, batch_size=32, validation_data=(test_images, test_labels_onehot))

# 10. Avaliar o modelo no conjunto de testes
test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot)
print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

# 11. Fazer previsões no conjunto de teste
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_images, axis=1)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 12. Plotar algumas imagens com as labels reais e preditas
def plot_images(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.title(f"True: {classes[true_labels[i][0]]}\nPred: {classes[predicted_labels[i]]}")
    plt.tight_layout()
    plt.show()

# Plotando as primeiras 10 imagens
plot_images(test_images_rgb, test_labels, y_pred_classes, num_images=10)

# 13. Calcular e plotar a matriz de confusão
conf_matrix = confusion_matrix(test_labels, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation = 'vertical')
plt.title("Matriz de Confusão")
plt.show()