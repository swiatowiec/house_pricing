from dataset_repository import DatasetRepository
from model_service import ModelService
from facade import PredictHousePrices
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts_path', type=str, default='artifacts')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    dataset_repository = DatasetRepository()
    model_service = ModelService()
    train_predict = PredictHousePrices(dataset_repository=dataset_repository,
                                        model_service=model_service,
                                        )
    train_predict.train_and_predict(artifacts_path=args.artifacts_path, 
                                    test_size=args.test_size,
                                    )
    print("Success")