DATASET_FILEPATH = "./CORE_SLE_RA_Control_blood_panels.csv"
POPULATION_STORAGE = "Population"

MAX_THREADS = 10000
MAX_LAYERS = 25
MAX_NODES = 200
DEFAULT_EPOCH = 500

ACTIVATIONS = [
    'relu',
    'softmax',
    'sigmoid',
    # 'leakyrelu',
    # 'prelu',
    'elu',
    # 'thresholdedrelu'
]

LAYER_TYPES = [
    'Dense',
    # 'Dropout',
    'BatchNormalization',
    # 'LayerNormalization',
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
    'SGD',
    'RMSprop',
    'Adam',
    'Adadelta',
    'Adagrad',
    'Adamax',
    'Nadam',
    'Ftrl',
]

BATCH_SIZES = [16, 32, 64, 128, 256, 512]
