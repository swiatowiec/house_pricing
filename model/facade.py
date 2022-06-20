from dataset_repository import DatasetRepository
from model_service import ModelService

class PredictHousePrices:
    def __init__(self,
                dataset_repository: DatasetRepository,
                model_service: ModelService):
        self._dataset_repository = dataset_repository
        self._model_service = model_service
        
    def train_and_predict(self, artifacts_path, test_size):         
        dataset, features, target = self._dataset_repository.read_data()
        self._model_service.generate_plots(dataset=dataset, path=artifacts_path)
        rmse, r2 = self._model_service.train_predict(features=features, target=target, test_size=test_size)
        self._dataset_repository.save_metrics(rmse=rmse, r2=r2, path=artifacts_path)