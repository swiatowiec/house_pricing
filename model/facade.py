from dataset_repository import DatasetRepository

class PredictHousePrices:
    def __init__(self,
                dataset_repository: DatasetRepository):
        self._dataset_repository = dataset_repository

    def train_and_predict(self):         
        dataset, features, target = self._dataset_repository.read_data()