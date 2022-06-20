import unittest
from dataset_repository import DatasetRepository
from model_service import ModelService
from facade import PredictHousePrices


class ModelTests(unittest.TestCase):
    def setUp(self):
        dataset_repository = DatasetRepository()
        model_service = ModelService()
        train_predict = PredictHousePrices(dataset_repository=dataset_repository,
        model_service=model_service,
        )
        return train_predict

    def test_if_reading_function_works(self):
        train_predict = self.setUp()
        dataset, features, target = train_predict._dataset_repository.read_data()
        self.assertEqual(dataset.shape, (506, 14))