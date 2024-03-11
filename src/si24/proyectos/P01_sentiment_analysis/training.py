from datasets import load_dataset
import os
import numpy as np

from preprocessing import get_vocab, preprocess_dataset
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(THIS_PATH, 'rotten_tomatoes_dataset.py')

'''
    Instrucciones:
    - En preprocessing.py completa el método get_one_hot_vector
    - En este archivo completa el código para preprocesar los datos, entrenar y evaluar el modelo.
    - cuando finalices corre este archivo para entrenar y evaluar tu modelo.

    En tu blog no olvides incluir:
     - Dos ejemplos de falsos poitivos
     - Dos ejemplos de falsos negativos
     - Dos ejemplos de verdaderos positivos
     - Dos ejemplos de verdaderos negativos
    sobre el conjunto de validación.
'''


def print_samples(dataset, n_samples, random=True):
    if random:
        indices = np.random.randint(0, len(dataset), n_samples)
    else:
        indices = np.arange(n_samples)

    for i in indices:
        idx = i.item()
        datapoint = dataset[idx]
        text = datapoint['text']
        label = datapoint['label']
        is_pos = "positive" if label else "negative"
        print(f"({is_pos}) - Text: {text}")

if __name__ == "__main__":
    # Carga de datos
    dataset = load_dataset(DATASET_PATH)
    training_set = dataset['train']
    validation_set = dataset['validation']
    test_set = dataset['test']

    print_samples(training_set, 5)

    # Preprocesamiento
    vocabulary = get_vocab(training_set)
    X_train, y_train = preprocess_dataset(training_set, vocabulary)
    X_val, y_val = preprocess_dataset(validation_set, vocabulary)

    # Entrenamiento
    # Sklearn
    # TODO entrena el modelo con los datos preprocesados


    # Evaluación
   # TODO: Evalua tu modelo con alguna o varias métricas de clasificación