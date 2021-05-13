from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout

from sklearn.metrics import classification_report

import random
import numpy as np

from options import DATASET_FILEPATH, ACTIVATIONS, LOSSES, OPTIMIZERS, BATCH_SIZES, MAX_LAYERS, MAX_NODES, DEFAULT_EPOCH, DATASET_FILEPATH, LAYER_TYPES
from housekeeping import prepareDataset

x_train, x_test, y_train, y_test = prepareDataset(DATASET_FILEPATH)


def RandomElement(l):
    return l[random.randint(0, len(l) - 1)]


def createRandIndividual():
    bias = [True, False]
    layers = []
    for _ in range(random.randint(0, MAX_LAYERS)):
        name = RandomElement(LAYER_TYPES)
        layer = {}
        if name == "Dense":
            layer = {
                "name": name,
                "options": {
                    "units": random.randint(0, MAX_NODES),
                    "activation": RandomElement(ACTIVATIONS),
                    "dropout": random.random(),
                    "use_bias": RandomElement(bias)
                },
            }
        elif name == "BatchNormalization" or name == "LayerNormalization":
            layer = {
                "name": name,
                "options": {},
            }
        elif name == "Dropout":
            layer = {
                "rate": random.random(),
            }
        layers.append(layer)

    # Final prediction layer
    layers.append({
        "name": "Dense",
        "options": {
            "units": 4,
            "activation": 'softmax',
        },
    })

    layers[0]['input_shape'] = (x_train.shape[1], )

    return {
        "layers": layers,
        "loss_function": LOSSES[random.randint(0,
                                               len(LOSSES) - 1)],
        "optimizer": OPTIMIZERS[random.randint(0,
                                               len(OPTIMIZERS) - 1)],
        "batch_size": BATCH_SIZES[random.randint(0,
                                                 len(BATCH_SIZES) - 1)],
        "learning_rate": random.uniform(0.1, 0.001),
    }


#--------------------------------------------------------------------------
def make_that_model(layers,
                    loss_function,
                    optimizer,
                    learning_rate,
                    exit_activation=None):
    model = keras.Sequential()
    for layer in layers:
        #print(layer)
        if 'dropout' in layer['options'].keys():
            del layer['options']['dropout']
            #layer['options']['Dropout'] = layer['options'].pop('dropout')
        model.add(getattr(keras.layers, layer["name"])(**layer["options"]))

    if exit_activation != None:
        model.add(Activation(exit_activation))

    op = getattr(keras.optimizers, optimizer)(learning_rate=(learning_rate))
    model.compile(loss=loss_function, optimizer=op, metrics=['accuracy'])

    return model


#--------------------------------------------------------------------------


def fitness_score(parameterization, epoch=DEFAULT_EPOCH):

    model = make_that_model(
        parameterization.get('layers'),
        parameterization.get('loss_function'),
        parameterization.get('optimizer'),
        parameterization.get('learning_rate'),
    )

    batch_size = parameterization.get('batch_size')

    _ = model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=0)

    # Predict with it
    pred = model.predict(x_test)

    got = np.argmax(pred, -1)
    want = np.argmax(y_test, -1)

    average = 'weighted avg'

    stat = classification_report(want, got, output_dict=True)
    metrics = dict(
        zip(model.metrics_names,
            model.evaluate(x=x_test, y=y_test, batch_size=batch_size)))

    score = (stat['accuracy'] + stat[average]['precision'] +
             stat[average]['f1-score'] +
             (1 - metrics['loss']) + stat[average]['recall']) / 5

    return score * 100