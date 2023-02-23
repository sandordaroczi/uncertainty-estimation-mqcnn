from enum import Enum

import numpy as np

EPSILON = np.finfo(np.float64).eps


class PredEnum(Enum):
    POINT_ESTIMATES = 'point_estimates'
    SAMPLES = 'samples'
    QUANTILES = 'quantiles'
    DISTRIBUTION_PARAMS = 'distribution_params'


valid_prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES, PredEnum.SAMPLES, PredEnum.DISTRIBUTION_PARAMS]
