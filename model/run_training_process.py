from dataset_repository import DatasetRepository
from model_service import ModelService
from facade import PredictHousePrices

if __name__ == '__main__':
    #TODO add parser and args
    dataset_repository = DatasetRepository()
    model_service = ModelService()
    train_predict = PredictHousePrices(dataset_repository=dataset_repository,
    model_service=model_service,
    )
    train_predict.train_and_predict()