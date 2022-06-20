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

    def test_train_predict(self):
        train_predict = self.setUp()
        dataset, features, target = train_predict._dataset_repository.read_data()
        rmse, r2 = train_predict._model_service.train_predict(features=features, target=target, test_size=0.2)
        rmse_rounded = round(rmse, 4)
        r2_rounded = round(r2, 4)
        self.assertEqual(rmse_rounded, 5.2723)
        self.assertEqual(r2_rounded, 0.6210)

    def test_if_metrics_are_saved(self):
        PATH = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.mkdir(PATH)
        train_predict = self.setUp()
        train_predict._dataset_repository.save_metrics(rmse=5.272, r2=0.620, path=PATH)
        num_of_creates_files = len(os.listdir(PATH))
        self.assertEqual(num_of_creates_files, 1)

    if __name__ == '__main__':
        unittest.main()
