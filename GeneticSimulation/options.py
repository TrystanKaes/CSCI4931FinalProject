MAX_THREADS = 10000
MAX_LAYERS = 25
MAX_NODES = 200
DEFAULT_EPOCH = 100

ACTIVATIONS = [
    'relu',
    'softmax',
    'sigmoid',
    # 'leakyrelu',
    # 'prelu',
    'elu',
    # 'thresholdedrelu'
]

LOSSES = [
    'categorical_crossentropy',
    'binary_crossentropy',
    # 'sparse_categorical_crossentropy',
    # 'poisson',
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'mean_squared_logarithmic_error',
    'log_cosh',
    'squared_hinge',
    'categorical_hinge',
]

OPTIMIZERS = [
    'sgd',
    'rmsprop',
    'adam',
    'adadelta',
    'adagrad',
    'adamax',
    'nadam',
    'ftrl',
]

BATCH_SIZES = [16, 32, 64, 128, 256, 512]

POPULATION_STORAGE = "/Population/"