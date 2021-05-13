from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout

from sklearn.metrics import precision_score, f1_score, recall_score

import random


from options import ACTIVATIONS, LOSSES, OPTIMIZERS, BATCH_SIZES, MAX_LAYERS, MAX_NODES, DEFAULT_EPOCH

from housekeeping import prepareDataset 

x_train, x_test, y_train, y_test = prepareDataset("./dataset.csv")

def createRandIndividual():
    layers = []
    for i in range(random.randint(0, MAX_LAYERS)):
        layer = {
            "count": random.randint(0, MAX_NODES),
            "activation": ACTIVATIONS[random.randint(0, len(ACTIVATIONS)-1)],
            "dropout": random.random(),
        }
        layers.append(layer)

    return {
        "layers": layers,
        "loss_function": LOSSES[random.randint(0, len(LOSSES)-1)],
        "optimizer": OPTIMIZERS[random.randint(0, len(OPTIMIZERS)-1)],
        "batch_size": BATCH_SIZES[random.randint(0, len(BATCH_SIZES)-1)],
    }

def make_that_model(layers, loss_function, optimizer, exit_activation=None):
    model = keras.Sequential()
    for i, layer in enumerate(layers):
        if i==0:
            model.add(Dense(layer["count"], input_dim=x_train.shape[1]))
            model.add(Activation(layer["activation"]))
            model.add(Dropout(layer["dropout"]))
        else:
            model.add(Dense(layer["count"]))
            model.add(Activation(layer["activation"]))
            model.add(Dropout(layer["dropout"]))

    model.add(Dense(1))

    if exit_activation!=None:
        model.add(Activation(exit_activation))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model


def fitness_score(parameterization, epoch=DEFAULT_EPOCH):
    
    model = make_that_model(parameterization.get('layers'),
                            parameterization.get('loss_function'),
                            parameterization.get('optimizer'),
                             )
    
    batch_size = parameterization.get('batch_size')
    
    _ = model.fit(x_train, y_train, batch_size=batch_size,  epochs=epoch, verbose=0)

    # Predict with it
    pred_y = model.predict(x_test)

    y_pred = [(0.5 < n) for n in pred_y]
    y_actual = [(0.5 < n) for n in y_test]

    precision = precision_score(y_actual, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_actual, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_actual, y_pred, average='binary', zero_division=0)

        # Collect Metrics
    metrics = dict(zip(model.metrics_names, model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=batch_size
    )))

    score = (metrics['accuracy']+precision+f1+(1-metrics['loss'])+recall) / 5

    return score * 100