from dataset_repository import DatasetRepository
from model_service import ModelService

class PredictHousePrices:
    def __init__(self,
                dataset_repository: DatasetRepository,
                model_service: ModelService):
        self._dataset_repository = dataset_repository
        self._model_service = model_service
        

    def train_and_predict(self):         
        dataset, features, target = self._dataset_repository.read_data()
        self._model_service.generate_plots(dataset=dataset)
        rmse, r2 = self._model_service.train_predict(features=features, target=target)
