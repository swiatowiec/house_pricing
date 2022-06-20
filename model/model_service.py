from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class ModelService:
    def train_predict(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = 42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)