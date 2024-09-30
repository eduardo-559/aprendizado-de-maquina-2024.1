import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import regularizers # type: ignore
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix

# Train the model
def train_model(model, train_images, train_labels, val_images, val_labels, file_path, data_aug=False, epochs=200):
    """
    Trains the given model on the provided training data.
    Args:
        model (keras.Sequential): The model to train.
        train_images (numpy.ndarray): Training images.
        train_labels (numpy.ndarray): Training labels.
        file_path (str): Path to save the best model weights.
        data_aug (bool): Whether to use data augmentation.
        epochs (int): Number of epochs to train the model.
    Returns:
        keras.callbacks.History: The history object containing training metrics.
        keras.Sequential: The model with the best weights loaded.
    """
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path,
                                                    monitor="val_loss", mode="min", 
                                                    save_best_only=True, verbose=1)
    
    if data_aug:
        datagen = ImageDataGenerator(
            rotation_range=15,  # Aumentando um pouco a rotação
            zoom_range=0.2,  # Zoom maior
            width_shift_range=0.2,  # Aumentando o deslocamento horizontal
            height_shift_range=0.2,  # Aumentando o deslocamento vertical
            horizontal_flip=True,  # Incluir flip horizontal
            vertical_flip=False  # Sem flip vertical, pois pode não fazer sentido em Fashion MNIST
        )
        datagen.fit(train_images)
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=32), 
                            callbacks=[early_stop, checkpoint], 
                            validation_data=(val_images, val_labels), 
                            epochs=epochs)
    else:
        history = model.fit(train_images, train_labels, 
                            callbacks=[early_stop, checkpoint], 
                            validation_data=(val_images, val_labels), 
                            epochs=epochs)
    
    # Load the best weights
    model.load_weights(file_path)
    
    return history, model

# Verifica o balanceamento atual do dataset
def verificar_balanceamento(labels):
    """
    Verifica se um dataset está balanceado e exibe a quantidade de amostras para cada classe.
    
    Args:
        labels (array-like): Lista ou array de rótulos (labels) do dataset.
    
    Returns:
        dict: Um dicionário com a quantidade de amostras para cada classe.
    """
    # Conta a quantidade de amostras para cada classe
    contador = Counter(labels)
    
    # Exibe a quantidade de amostras para cada classe
    print("Quantidade de amostras para cada classe:")
    for classe, quantidade in contador.items():
        print(f"Classe {classe}: {quantidade} amostras")
    
    # Identifica a classe com a menor e maior quantidade de amostras
    min_classe = min(contador, key=contador.get)
    max_classe = max(contador, key=contador.get)
    
    print("\nResumo:")
    print(f"Classe com menor quantidade de amostras: Classe {min_classe} com {contador[min_classe]} amostras")
    print(f"Classe com maior quantidade de amostras: Classe {max_classe} com {contador[max_classe]} amostras")
    
    # Verifica se há necessidade de balanceamento
    if len(set(contador.values())) == 1:
        print("\nO dataset está balanceado.")
    else:
        print("\nO dataset NÃO está balanceado. Considere técnicas de balanceamento.")
    
    return dict(contador)
# Exemplo de uso
# labels = [0, 0, 1, 1, 1, 2, 2, 2, 2]  # Substitua por seus próprios rótulos
# verificar_balanceamento(labels)

# Função de preprocessamento de dados
def preprocess_data(images): # Exemplo de uso: prepared_train_images = preprocess_data(train_images)
    """
    Normalizes the image pixel values to the range [0, 1].
    Args:
        images (numpy.ndarray): Array of images to be normalized.
    Returns:
        numpy.ndarray: Normalized images.
    """
    return images / 255.0

# ========================================================================================================

# Funções de plotagem

# Mostrar 1 única figura com o nome da classe
def plot_image(image, label, class_names): 
    """
    Exibe uma imagem com o nome da classe correspondente.
    
    Args:
        image (numpy.ndarray): A imagem a ser exibida.
        label (int): O índice da classe da imagem.
        class_names (list): Lista com os nomes das classes.
    """
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.title(f'Class: {class_names[label]}')
    plt.show()

# Visualize the first 25 images from the training set with their labels
def plot_sample_images(images, labels, class_names, num_samples=25): # Uso: plot_sample_images(train_images, train_labels, class_names) pode ser train ou test images
    plt.figure(figsize=(10,10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

# Plotando o gráfico da perda (loss) com tamanho ajustado para melhor legibilidade
def plot_loss(history): #Uso: plot_loss(history)
    """
    Plots the loss of the model during training with enhanced size for better readability.
    Args:
        history (keras.callbacks.History): The history object returned by model.fit containing training metrics.
    """
    plt.figure(figsize=(12, 8))  # Aumentando o tamanho do gráfico
    plt.plot(history.history['loss'], label='Loss', color='blue', linewidth=2)
    plt.title('Loss during Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    """
    Plots the training and validation loss of the model during training.
    
    Args:
        history (keras.callbacks.History): The history object returned by model.fit containing training metrics.
    """
    plt.figure(figsize=(6, 4))  # Aumentando o tamanho do gráfico
    
    epochs = range(1, len(history.history['loss']) + 1)  # Ajusta as épocas para começar a partir de 1
    
    # Plot do loss (perda durante o treinamento)
    plt.plot(epochs, history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    
    # Plot do val_loss (perda durante a validação)
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    
    # Título e rótulos
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Grid e legibilidade
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ajuste do layout
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_images, test_labels, class_names):
    """
    Gera e plota uma matriz de confusão para as predições do modelo.

    Args:
        model (keras.Sequential): O modelo treinado.
        test_images (numpy.ndarray): Conjunto de imagens de teste.
        test_labels (numpy.ndarray): Conjunto de rótulos verdadeiros.
        class_names (list): Lista de nomes das classes (rótulos).

    Returns:
        None: A função apenas exibe a matriz de confusão.
    """
    # Realiza as predições no conjunto de teste
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calcula a matriz de confusão
    cm = confusion_matrix(test_labels, predicted_labels)

    # Plot da matriz de confusão usando Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Matriz de Confusão')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
