from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

class ModelService:
    def generate_plots(self, dataset, path):
        #TODO add path as params
        corr_matrix = dataset.corr().round(2)
        sns.heatmap(data=corr_matrix, annot=True)
        plt.savefig(os.path.join('correlation_matrix.png'))

        sns.set(rc={'figure.figsize':(10,10)})
        sns.distplot(dataset['MEDV'], bins=20)
        plt.savefig(os.path.join(path, 'prices_(MEDV).png'))

    def train_predict(self, features, target):
        selected_features = self._select_features(features=features)
        #TODO test size as params
        X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state = 42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        rmse, r2 = self._calculate_metrics(y_test, pred)
        return rmse, r2

    def _select_features(self, features):
        #feature selection based on corr matrix
        selected_features = pd.DataFrame(np.c_[features['RM'], features['INDUS'], features['NOX'], features['TAX'], features['PTRATIO'], features['LSTAT']])
        return selected_features

    def _calculate_metrics(self, y_test, pred):
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        return rmse, r2