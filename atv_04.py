# Download do dataset

import opendatasets as od
od.download("https://www.kaggle.com/datasets/gpiosenka/sports-classification?datasetId=1209061&sortBy=voteCount&select=train")


# Importa bibliotecas

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Carrega dataset

import os
import tensorflow as tf

dataset_base_dir = '/content/sports-classification'

all_class_names = os.listdir(os.path.join(dataset_base_dir, 'train'))

class_image_counts = {}
for class_name in all_class_names:
    class_dir = os.path.join(dataset_base_dir, 'train', class_name)
    if os.path.isdir(class_dir):
        class_image_counts[class_name] = len(os.listdir(class_dir))

sorted_classes = sorted(class_image_counts.items(), key=lambda item: item[1], reverse=True)
selected_class_names = [class_name for class_name, count in sorted_classes[:10]]

print(f"Classes selecionadas com maior quantidade de imagens: {selected_class_names}")

train_dirs = [os.path.join(dataset_base_dir, 'train', class_name) for class_name in selected_class_names]
val_dirs = [os.path.join(dataset_base_dir, 'valid', class_name) for class_name in selected_class_names]
test_dirs = [os.path.join(dataset_base_dir, 'test', class_name) for class_name in selected_class_names]

def load_subset_dataset(directory_list, labels='inferred', label_mode='int', image_size=(224, 224), batch_size=16, shuffle=True):
    
    image_paths = []
    image_labels = []
    class_to_label = {class_name: i for i, class_name in enumerate(selected_class_names)}

    for class_dir in directory_list:
        class_name = os.path.basename(class_dir)
        label = class_to_label[class_name]
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            image_labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    def load_image(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return img, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, selected_class_names


train_data, class_names = load_subset_dataset(train_dirs, batch_size=16, image_size=(224,224), shuffle=True)
val_data, _ = load_subset_dataset(val_dirs, batch_size=16, image_size=(224,224), shuffle=True)
test_data, _ = load_subset_dataset(test_dirs, batch_size=16, image_size=(224,224), shuffle=False)


# Visualiza os dados

plt.figure(figsize=(20,12))
for x in train_data.take(1):
    for i in range(15):
        plt.subplot(3,5,i+1)
        image = x[0][i] / 255.
        plt.imshow(image)
        plt.title(class_names[x[1][i].numpy()])
        plt.axis('off')

train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache()


# Utiliza o modelo pré-treinado ResNet50

preprocess_input = keras.applications.resnet50.preprocess_input
base_model = keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights="imagenet")

base_model.summary()


# Data augumetation

data_augmentation = keras.Sequential([
    keras.layers.Resizing(height=224, width=224),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomFlip("horizontal",),
    keras.layers.RandomZoom(0.1),
])


# Visualiza data augumentation

for images, labels in train_data.take(1):
    original_image = images[0]
    original_label = labels[0]
    break

augmented_images = [data_augmentation(tf.expand_dims(original_image, 0))[0] for _ in range(4)]

plt.figure(figsize=(12, 6))

plt.subplot(1, 5, 1)
plt.imshow(original_image.numpy().astype("uint8"))
plt.title(f"Original\nClass: {class_names[original_label.numpy()]}")
plt.axis("off")

augmentation_types = ["Random Rotation", "Random Flip (Horizontal)", "Random Zoom", "Random Rotation, Flip, and Zoom"]

for i in range(len(augmented_images)):
    plt.subplot(1, 5, i + 2)
    plt.imshow(augmented_images[i].numpy().astype("uint8"))
    plt.title(f"Augmented {i+1}\n({augmentation_types[i]})")
    plt.axis("off")

plt.tight_layout()
plt.show()


# Cria modelo keras

inputs = keras.Input(shape=[None,None,3])
x = data_augmentation(inputs)
x = tf.keras.layers.Normalization(mean=0.0, variance=1.0)(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x)
x =keras.layers.Dense(512, "relu")(x)
x = keras.layers.BatchNormalization()(x)
outputs = keras.layers.Dense(len(selected_class_names), "softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()

model.layers[4].trainable = False


# Treina o modelo

keras.backend.clear_session()

base_learning_rate = 0.001

model.compile(optimizer=keras.optimizers.Adam(base_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
history_1 = model.fit(train_data, validation_data=val_data, epochs=10)


# Visualiza curva de aprendizado do modelo

def plot_history(history):
    history_dict = history.history

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))

    ax[0].plot(list(range(len(history_dict['loss']))), history_dict["loss"], 'ro-', lw=3, markersize=8, label="training loss")
    ax[0].plot(list(range(len(history_dict['loss']))), history_dict["val_loss"], 'b^-', lw=3, markersize=8, label="validation loss")

    ax[1].plot(list(range(len(history_dict['loss']))), history_dict["accuracy"], 'ro-', lw=3, markersize=8, label="training accuracy")
    ax[1].plot(list(range(len(history_dict['loss']))), history_dict["val_accuracy"], 'b^-', lw=3, markersize=8, label="validation accuracy")

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper left")

    ax[0].set_xlabel("epochs", fontsize=12, weight='bold')
    ax[1].set_xlabel("epochs", fontsize=12, weight='bold')

    ax[0].set_ylabel("loss", fontsize=12, weight='bold')
    ax[1].set_ylabel("accuracy", fontsize=12, weight='bold')

    fig.text(0.515, .93, "Model learning curves", ha="center", va="top", fontsize=18, weight='bold')
    plt.show()

plot_history(history_1)


# Testa o modelo

scores = model.evaluate(test_data, verbose=False)
print(f"Root mean squared error: {scores[0]:.4f}")
print(f"Test accuracy: {scores[1]:.2%}")

pred_1 = model.predict(test_data, verbose=False)
pred_1_cleaned = np.argmax(pred_1, axis=1)

plt.figure(figsize=(30,25))
range_ = 0
for images, labels in test_data.take(5):
    range_ += len(labels)
    prev = range_ - len(labels)
    for i in range(prev, range_):
        plt.subplot(8, 10, i+1)
        x = images[i - prev].numpy() / 255.

        plt.title(f"""  Target: {class_names[labels[i - prev].numpy()]}
        Predicted: {class_names[pred_1_cleaned[i]]}""", fontsize=8)
        plt.imshow(x)
        plt.axis('off')


# Matriz de confusão e classificação

test_labels = []
for images, labels in test_data.take(-1):
    for i in range(len(labels)):
        test_labels.append(labels[i].numpy())

print(classification_report(pred_1_cleaned, test_labels))

conf_matrix_1 = confusion_matrix(test_labels, pred_1_cleaned)
np.fill_diagonal(conf_matrix_1, 0)

labels = [class_names[val] for val in np.unique(test_labels)]

plt.figure(figsize=(40, 30))
g = sns.heatmap(conf_matrix_1, cbar=False, annot=True)
g.set_xticklabels(labels=labels, rotation=75)
g.xaxis.tick_top()
g.set_yticklabels(labels=labels, rotation=0)
plt.show()


# Visualiza os erros de predição

test_images = []
for images, labels in test_data.take(-1):
    for i in range(len(labels)):
        test_images.append(images[i].numpy())

test_images = np.array(test_images)

idx = test_labels != pred_1_cleaned
idx = idx.tolist()

X_wrong = test_images[idx]
y_wrong = pred_1_cleaned[idx]
y_correct = np.array(test_labels)[idx]

plt.figure(figsize=(30,18))
for i in range(len(y_wrong)):
    plt.subplot(4, 8, i+1)
    x = X_wrong[i]/255.

    plt.title(f"""  Target: {class_names[y_correct[i]]}
    Predicted: {class_names[y_wrong[i]]}""", fontsize=8)
    plt.imshow(x)
    plt.axis('off')


# Salva modelo e recarrega

model.save("sports_classification_model.keras")
cloned_model = keras.models.load_model("sports_classification_model.keras")
cloned_model.summary()


# Fine-tuning

len(cloned_model.layers)
cloned_model.layers[8].trainable = True
fine_tune_at = 8

for layer in cloned_model.layers[:fine_tune_at]:
  layer.trainable = False

base_learning_rate = 0.001/10

cloned_model.compile(optimizer=keras.optimizers.Adam(base_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history_2 = cloned_model.fit(train_data, validation_data=val_data, initial_epoch=9, epochs=20)

model.save("sports_classification_model2.keras")


# Visualiza curva de aprendizado do modelo

plot_history(history_2)


# Testa o modelo

scores = cloned_model.evaluate(test_data, verbose=False)

print(f"Root mean squared error: {scores[0]:.4f}")
print(f"Test accuracy: {scores[1]:.2%}")

pred_2 = cloned_model.predict(test_data, verbose=False)
pred_2_cleaned = np.argmax(pred_1, axis=1)

plt.figure(figsize=(30,25))
range_ = 0
for images, labels in test_data.take(5):
    range_ += len(labels)
    prev = range_ - len(labels)
    for i in range(prev, range_):
        plt.subplot(8, 10, i+1)
        x = images[i - prev].numpy() / 255.

        plt.title(f"""  Target: {class_names[labels[i - prev].numpy()]}
        Predicted: {class_names[pred_2_cleaned[i]]}""", fontsize=8)
        plt.imshow(x)
        plt.axis('off')


# Matriz de confusão e classificação

print(classification_report(pred_2_cleaned, test_labels))

conf_matrix_2 = confusion_matrix(test_labels, pred_2_cleaned)
np.fill_diagonal(conf_matrix_2, 0)

labels = [class_names[val] for val in np.unique(test_labels)]

plt.figure(figsize=(40, 30))
g = sns.heatmap(conf_matrix_2, cbar=False, annot=True, annot_kws={"size": 35})
g.set_xticklabels(labels=labels, rotation=75, fontsize=35)
g.xaxis.tick_top()
g.set_yticklabels(labels=labels, rotation=0, fontsize=35)
plt.show()


# Visualiza os erros de predição

idx_2 = test_labels != pred_2_cleaned
idx_2 = idx_2.tolist()

X_wrong_2 = test_images[idx_2]
y_wrong_2 = pred_2_cleaned[idx_2]
y_correct_2 = np.array(test_labels)[idx_2]

plt.figure(figsize=(30,18))
for i in range(len(y_wrong_2)):
    plt.subplot(4, 8, i+1)
    x = X_wrong_2[i]/255.

    plt.title(f"""  Target: {class_names[y_correct_2[i]]}
    Predicted: {class_names[y_wrong_2[i]]}""", fontsize=8)
    plt.imshow(x)
    plt.axis('off')


# Visualiza imagem após a camada de normalização

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

def display_activation(activations, col_size, row_size, layer_name):

    activation = activations[0]

    n_features = activation.shape[-1]
    limit = min(n_features, 64)
    size = int(np.ceil(np.sqrt(limit)))

    fig, ax = plt.subplots(size, size, figsize=(size*1.8, size*1.8))
    fig.suptitle(f'Camada: {layer_name} ({n_features} filtros)', fontsize=14)

    for i in range(limit):
        row = i // size
        col = i % size

        if size > 1:
            current_ax = ax[row][col]
        elif size == 1:
            current_ax = ax
        else:
            continue

        if len(activation.shape) == 3:
            current_ax.imshow(activation[:, :, i], cmap='viridis')
        else:
            current_ax.text(0.1, 0.5, f'Saída não visual', fontsize=10)

        current_ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def get_visual_layers(model):

    visual_layers = []

    ignore_substrings = [
        'input', 'dropout', 'flatten', 'dense', 'add',
        'zero_padding', 'activation', 'shortcut', 
        'conv1_pad', 'conv1_conv', 
    ]

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 0:
            for inner_layer in layer.layers:
                layer_name = inner_layer.name.lower()

                if any(sub in layer_name for sub in ignore_substrings):
                    continue

                if len(inner_layer.output.shape) == 4 and ('conv' in layer_name or 'pool' in layer_name):
                    visual_layers.append(inner_layer)

        elif len(layer.output.shape) == 4 and 'input' not in layer.name.lower():
            visual_layers.append(layer)

    return visual_layers

MODEL_PATH = '/content/sports-classification/sports_classification_model.keras'
IMAGE_TO_ANALYZE_PATH = '/content/sports-classification/test/baseball/1.jpg'

INPUT_SIZE = None
INPUT_SIZE_FALLBACK = 224

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    detected_size = model.input_shape[1]
    INPUT_SIZE = detected_size if detected_size is not None else INPUT_SIZE_FALLBACK

except Exception as e:
    print(f"\n[ERRO CRÍTICO]: Falha ao carregar o modelo em {MODEL_PATH}.")
    print(f"Detalhes do erro: {e}")

    INPUT_SIZE = INPUT_SIZE_FALLBACK
    model = models.Sequential([
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='input_layer'),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv_dummy_1'),
        layers.MaxPooling2D((2, 2), name='pool_dummy_1'),
    ])
    print(f"\nUsando um modelo dummy {INPUT_SIZE}x{INPUT_SIZE} para demonstração.")

try:
    img = tf.keras.preprocessing.image.load_img(
        IMAGE_TO_ANALYZE_PATH,
        target_size=(INPUT_SIZE, INPUT_SIZE)
    )
    x = tf.keras.preprocessing.image.img_to_array(img)
    processed_img = preprocess_input(np.expand_dims(x, axis=0))

except FileNotFoundError:
    print(f"\n[ERRO CRÍTICO]: Não foi possível encontrar a imagem em '{IMAGE_TO_ANALYZE_PATH}'.")
    exit()

visual_layers = get_visual_layers(model)

if not visual_layers:
    print("\n[AVISO] Não foram encontradas camadas convolucionais ou de pooling visuais para plotar.")
else:
    plt.figure()
    plt.title(f"Imagem original de entrada: {IMAGE_TO_ANALYZE_PATH}")
    plt.imshow(x / 255.0)
    plt.axis('off')
    plt.show()

    for layer in visual_layers:
        try:
            extractor_model = models.Model(inputs=model.input, outputs=layer.output)
            activation = extractor_model.predict(processed_img)
            display_activation(activation, 0, 0, layer.name)

        except Exception as e:
            continue

print("Processo de visualização concluído.")
