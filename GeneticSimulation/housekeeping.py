import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

LABELER = None


def prepareDataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(data.columns[0], axis=1)

    global LABELER
    LABELER = LabelEncoder()

    int_data = LABELER.fit_transform(data['Condition'])
    int_data = int_data.reshape(len(int_data), 1)

    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(int_data)

    y = onehot_data

    x = data.drop(['Name', 'Condition'], axis=1)

    # Prepare and scale the data.
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=25)

    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)

    return x_train, x_test, y_train, y_test
