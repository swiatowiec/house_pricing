from sklearn.datasets import load_boston
import pandas as pd

class DatasetRepository:
    def read_data(self):
        dataset = load_boston()
        features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        target = pd.DataFrame(dataset.target, columns=['MEDV'])
        df = pd.concat([features, target], axis=1)
        return df, features, target