import unittest
from dataset_repository import DatasetRepository
from model_service import ModelService
from facade import PredictHousePrices
import tempfile
import uuid
import os


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

    def test_if_plots_are_created(self):
        PATH = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.mkdir(PATH)
        train_predict = self.setUp()
        dataset, features, target = train_predict._dataset_repository.read_data()
        train_predict._model_service.generate_plots(dataset, PATH)
        num_of_creates_plots = len(os.listdir(PATH))
        self.assertEqual(num_of_creates_plots, 2)