from source.datasets.baseline.prediction import BaselineDatasetPrediction
from source.datasets.baseline.regression import BaselineDatasetRegression
from source.datasets.baseline.generation import BaselineDatasetGeneration
from source.datasets.molx.prediction import MolXDatasetPrediction
from source.datasets.molx.regression import MolXDatasetRegression
from source.datasets.molx.generation import MolXDatasetGeneration

load_dataset = {
    'baseline_prediction': BaselineDatasetPrediction,
    'baseline_regression': BaselineDatasetRegression,
    'baseline_generation': BaselineDatasetGeneration,
    'molx_prediction': MolXDatasetPrediction,
    'molx_regression': MolXDatasetRegression,
    'molx_generation': MolXDatasetGeneration,
}