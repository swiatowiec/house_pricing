from sklearn.datasets import load_boston
import pandas as pd
import json
import os


class DatasetRepository:
    def read_data(self):
        dataset = load_boston()
        features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        target = pd.DataFrame(dataset.target, columns=['MEDV'])
        df = pd.concat([features, target], axis=1)
        return df, features, target

    def save_metrics(self, rmse, r2, path):
        json_content = {'rmse': rmse, "r2": r2}
        metrics_path =  os.path.join(path, "metrics.json")
        with open(metrics_path, 'w') as file:
            json.dump(json_content, file)