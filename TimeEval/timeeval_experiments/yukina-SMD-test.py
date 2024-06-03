import os.path
from pathlib import Path
from typing import Dict, Any

import numpy as np

from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.algorithms import subsequence_if
from timeeval.params import FixedParameters

from timeeval.constants import HPI_CLUSTER

from timeeval.adapters import DockerAdapter

from timeeval import DefaultMetrics

path_project = '/home/yukina/Missile_Fault_Detection/project'

# Load dataset metadata
dm = DatasetManager(Path(os.path.join(path_project, 'TimeEval/data')), create_if_missing=False)


# Define algorithm
def my_algorithm(data: np.ndarray, args: Dict[str, Any]) -> np.ndarray:
    score_value = args.get("score_value", 0)
    return np.full_like(data, fill_value=score_value)


# Select datasets and algorithms
datasets = dm.select(collection="SMD")

# Add algorithms to evaluate...
algorithms = [
    Algorithm(
        name="autoencoder",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/autoencoder",
            tag="1.0",  # please use a specific tag instead of "latest" for reproducibility
            skip_pull=True  # set to True because the image is already present from the previous section
        ),
        # param_config=FixedParameters({"score_value": 1.}),
        # required by DockerAdapter
        data_as_file = True,
        # You must specify the algorithm metadata here. The categories for all TimeEval algorithms can
        # be found in their README or their manifest.json-File.
        # UNSUPERVISED --> no training, SEMI_SUPERVISED --> training on normal data, SUPERVISED --> training on anomalies
        # if SEMI_SUPERVISED or SUPERVISED, the datasets must have a corresponding training time series
        training_type = TrainingType.SEMI_SUPERVISED,
        # MULTIVARIATE (multidimensional TS) or UNIVARIATE (just a single dimension is supported)
        input_dimensionality = InputDimensionality.MULTIVARIATE
    )
]
timeeval = TimeEval(dm, datasets, algorithms,
    # you can choose from different metrics:
    metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC],
)

# execute evaluation
timeeval.run()
# retrieve results
print(timeeval.get_results())