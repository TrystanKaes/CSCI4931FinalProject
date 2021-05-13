import pandas as pd
from sklearn.model_selection import train_test_split  #Train-test splitting library

from sklearn.preprocessing import StandardScaler


def prepareDataset(filepath):
    data = pd.read_csv(filepath)
    x = data.drop('Exited', axis=1)
    x = x.drop('CustomerId', axis=1)
    x.Surname = x.Surname.astype('category').values.codes
    x.Geography = x.Geography.astype('category').values.codes
    x.Gender = x.Gender.astype('category').values.codes
    y = data['Exited']

    x = StandardScaler().fit_transform(x)

    return train_test_split(x, y, test_size=0.2, random_state=25)
