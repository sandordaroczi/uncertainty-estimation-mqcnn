import copy
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset as Gluon_PandasDataset
from gluonts.dataset.split import DateSplitter as Gluon_DateSplitter
from gluonts.mx import MQCNNEstimator
from scipy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from .constants import PredEnum, valid_prediction_types, EPSILON

target_transformer_map = {
    "id": (lambda x: x),
    "log": np.log,
    "log1p": np.log1p
}

inverse_target_transformer_map = {
    "id": (lambda x: x),
    "log": np.exp,
    "log1p": np.expm1
}

valid_target_transformers = list(target_transformer_map.values())


class Model(ABC):

    @abstractmethod
    def __init__(self, vectorizer, target_transformer: str = "log1p"):
        self.target_transformer_string = target_transformer
        try:
            self.target_transformer = target_transformer_map[target_transformer]
            self.inverse_target_transformer = inverse_target_transformer_map[target_transformer]
        except KeyError as e:
            raise KeyError(f"Target transformer should be one of {list(target_transformer_map.keys())}") from e

        self.vectorizer = vectorizer
        self.quantiles = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, target: str, verbose: bool, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame, verbose: bool, **kwargs: object) -> Dict:
        pass

    @abstractmethod
    def metrics(self, y_test, predictions, prediction_types: Optional[List],
                verbose: bool):
        """

        Args:
            y_test: array-like of shape (n_samples, 1)
                array containing the test target values
            predictions: Dict
                a dictionary with values containing point and uncertainty predictions
            prediction_types: List
                a list of prediction types
            verbose: bool
                currently not used

        Returns:
            metrics values as a dict
        """
        if prediction_types is None:
            prediction_types = list(predictions.keys())
        prediction_types = list(set(prediction_types) & set(valid_prediction_types))

        metrics = {}
        if PredEnum.POINT_ESTIMATES in prediction_types:
            point_predictions = predictions[PredEnum.POINT_ESTIMATES]
            metrics["mse"] = self.mse(y_test, point_predictions)
            metrics["mae"] = self.mae(y_test, point_predictions)
            metrics["rmse"] = self.rmse(y_test, point_predictions)
            metrics["mape"] = self.mape(y_test, point_predictions)
            metrics["rmspe"] = self.rmspe(y_test, point_predictions)

        if PredEnum.QUANTILES in prediction_types:
            quantiles = predictions[PredEnum.QUANTILES]
            if quantiles.shape[1] != 2:
                raise ValueError(f"Exactly 2 quantiles have to be supplied"
                                 f"to calculate avg_interval_length (sharpness) and coverage")
            metrics["avg_interval_length"] = self.avg_interval_length(quantiles)
            metrics["sharpness"] = metrics["avg_interval_length"]
            metrics["coverage"] = self.coverage(y_test, quantiles)

        if PredEnum.SAMPLES in prediction_types:
            metrics["crps"] = self.crps(y_test, predictions[PredEnum.SAMPLES])
            try:
                metrics["nll_from_samples"] = self.neg_log_likelihood_with_kde(y_test, predictions[PredEnum.SAMPLES],
                                                                               parallel=True, verbose=verbose)
            except LinAlgError as e:
                print(f"Exception encountered while trying to calculate NLL from samples using KDE: {e}")
                print("Setting NLL value as infinity.")
                metrics["nll_from_samples"] = np.inf

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            pass
        if len(metrics.keys()) == 0:
            raise ValueError(f"Cannot compute metrics."
                             f"Please provide via the prediction_types parameter which metrics to compute."
                             f"Valid options: {valid_prediction_types}")

        return metrics

    @staticmethod
    def get_predictions_for_ci_quantiles(predictions, confidence_interval_quantiles, quantiles):
        if confidence_interval_quantiles is None:
            confidence_interval_quantiles = [min(quantiles), max(quantiles)]
        if not all(quantile in quantiles for quantile in confidence_interval_quantiles):
            raise ValueError('Please specify confidence_interval_quantiles which are computed by the model')

        if len(confidence_interval_quantiles) != 2:
            raise ValueError('Please specify exactly 2 quantiles for confidence_interval_quantiles')

        if confidence_interval_quantiles[0] > confidence_interval_quantiles[1]:
            raise ValueError('Lower quantile has to be a smaller than upper quantile')

        lower_limit = np.reshape(predictions[PredEnum.QUANTILES][confidence_interval_quantiles[0]],
                                 newshape=(-1, 1))
        upper_limit = np.reshape(predictions[PredEnum.QUANTILES][confidence_interval_quantiles[1]],
                                 newshape=(-1, 1))

        return np.concatenate((lower_limit, upper_limit), axis=1)

    @staticmethod
    def train_val_split(X, y, prob, group_after: Optional[str] = None):
        if group_after is None:
            bit = np.random.choice([0, 1], size=(X.shape[0],), p=[prob, 1 - prob]).astype(bool)
            X_train = X[bit]
            X_calib = X[np.logical_not(bit)]
            y_train = y[bit]
            y_calib = y[np.logical_not(bit)]
        else:
            groups = X[group_after].unique()
            bit = np.random.choice([0, 1], size=(groups.shape[0],), p=[prob, 1 - prob]).astype(bool)
            chosen_groups = groups[bit]
            X_train = X[X[group_after].isin(chosen_groups)]
            X_calib = X[~X[group_after].isin(chosen_groups)]
            y_train = y[X[group_after].isin(chosen_groups)]
            y_calib = y[~X[group_after].isin(chosen_groups)]

        return X_train, y_train, X_calib, y_calib

    @staticmethod
    def mse(y_test, predictions):
        """mean squared error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_squared_error(y_test, predictions)

    @staticmethod
    def mae(y_test, predictions):
        """mean absolute error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_absolute_error(y_test, predictions)

    @staticmethod
    def rmse(y_test, predictions):
        """residual mean squared error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return np.sqrt(mean_squared_error(y_test, predictions))

    @staticmethod
    def mape(y_test, predictions):
        """mean absolute percentage error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_absolute_percentage_error(y_test, predictions)

    @staticmethod
    def rmspe(y_test, predictions):
        """residual mean squared percentage error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return np.sqrt(np.mean(
            np.square((y_test - predictions) / (y_test + EPSILON))))  # Avoid numerical issues of division by zero

    @staticmethod
    def mase(y_test, predictions, y_train, n_timeseries, seasonal_periodicity=1):
        """mean absolut scaled error: measure of the accuracy of forecasts

        Args:
            y_test: array-like of shape (n_timeseries, forecast_horizon, 1)
                observed value
            predictions: array-like of shape (n_timeseries, forecast_horizon, 1)
            y_train: array-like of shape (n_timeseries, lookback+forecast_horizon, 1)
                target variables which have been used during model training
            n_timeseries: int describing the total number of time series
            seasonal_periodicity: Seasonal periodicity of training data.
        """
        mase = []
        for ts_index in range(n_timeseries):
            y_pred_naive = y_train[ts_index, :-seasonal_periodicity]
            mae_naive = mean_absolute_error(y_train[ts_index, seasonal_periodicity:], y_pred_naive)
            mae_pred = mean_absolute_error(y_test, predictions)
            single_mase = mae_pred / np.maximum(mae_naive, EPSILON)
            mase.append(single_mase)

        return np.mean(mase)

    @staticmethod
    def avg_interval_length(predictions):
        """ average interval length

        Args:
            predictions:
                lower and upperbounds of confidence interval for each datapoint in a np.array(N,2)
        """
        return np.mean(predictions[:, 1] - predictions[:, 0])

    @staticmethod
    def coverage(y_test, predictions):
        """coverage

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,2)
                point estimates for the real values
        """
        return np.mean((y_test >= predictions[:, 0]) & (y_test <= predictions[:, 1]))


class MQCNN(Model):
    """
    Implementation of MQCNN for use and benchmarking with different vectorizers for confidence intervals
    """

    def __init__(self, vectorizer=None, target_transformer: str = "log1p",
                 freq: str = "D", lookback: Optional[int] = None, forecast_horizon: int = 1,
                 item_id: str = None, timestamp: Optional[str] = None,
                 feat_static_cat: Optional[List[str]] = None,
                 feat_dynamic_real: Optional[List[str]] = None,
                 past_feat_dynamic_real: Optional[List[str]] = None,
                 cardinality_static_cat: Optional[List[int]] = None,
                 dynamic_feature_scaler=StandardScaler(),
                 quantiles: Optional[List[float]] = None):

        """
        Args:
            vectorizer: not used in this model
            target_transformer: specify if model should be trained on log scale --> metrics are evaluated on real scale
            freq: Frequency of observations in the time series. Must be a valid pandas frequency.
            lookback: Number of time units that condition the predictions
            forecast_horizon: Length of the prediction
            item_id: Name of the column that, when grouped by, gives the different time series.
            timestamp: Name of the column that contains the timestamp information.
            feat_static_cat: List of column names that contain static real features.
            feat_dynamic_real: List of column names that contain dynamic real features.
            past_feat_dynamic_real: List of column names that contain dynamic real features only for the history
            cardinality_static_cat: List of amount of categories for feat_static_cat
            dynamic_feature_scaler: Perform scaling of dynamic features via e.g. scikit-learn StandardScaler
            quantiles: Optional[List[float]]: List of quantiles you want to perform Quantile Regression on.
                        Must contain 0.5 quantile
            and be at least of length 3
        """

        if item_id is None:
            raise ValueError(
                'group_cols must be specified, otherwise MQCNN for multiple time series does not make sense')

        if lookback is None:
            lookback = 4 * forecast_horizon

        if feat_static_cat is not None and cardinality_static_cat is not None:
            use_feat_static_cat = True
        elif feat_static_cat is not None and cardinality_static_cat is None:
            raise ValueError('if feat_static_cat shall be included cardinality of these must be specified too')
        else:
            use_feat_static_cat = False
            feat_static_cat = []

        if feat_dynamic_real is None:
            feat_dynamic_real = []
            use_feat_dynamic_real = False
        else:
            use_feat_dynamic_real = True

        if past_feat_dynamic_real is None:
            past_feat_dynamic_real = []
            use_past_feat_dynamic_real = False
        else:
            use_past_feat_dynamic_real = True

        if quantiles is None:
            quantiles = [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]

        if len(quantiles) < 3:
            raise ValueError('Specify at least one median and two upper and lower bound quantiles')

        if 0.5 not in quantiles:
            raise ValueError('Median quantile is required to get meaningful point predictions')

        add_time_feature = True if timestamp is None else False

        super().__init__(vectorizer, target_transformer)
        self.model = None
        self.freq = freq
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.item_id = item_id
        self.timestamp = timestamp
        self.feat_static_cat = feat_static_cat
        self.cardinality_static_cat = cardinality_static_cat
        self.feat_dynamic_real = feat_dynamic_real
        self.past_feat_dynamic_real = past_feat_dynamic_real
        self.dynamic_feature_scaler = dynamic_feature_scaler
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_past_feat_dynamic_real = use_past_feat_dynamic_real
        self.add_time_feature = add_time_feature
        self.quantiles = quantiles

        self.predictor = None
        self.target = None

    def fit(self, X: pd.DataFrame, target: str, verbose: bool = False, params_mqcnn: Optional[Dict] = None):

        """
        fit method trains model with possibility for early stopping
        train data: pd.DataFrame
            consists of all data of dataframe X from least recent data up until the beginning of the last
            forecast_horizon time steps
        validation data: srt
            consists of all data of dataframe X

        Args:
            X: pd.Dataframe
                dataframe with features --> Given up until start of holdout set
            target: str
                value to estimate, target value
            params_mqcnn: dict
                additional parameters to specify MQCNNEstimator further

        Returns:
            self.predictor
        """
        if params_mqcnn is None:
            params_mqcnn = {}

        start_time = time.perf_counter()
        self.target = target
        X_copy = X.copy()
        X_copy[self.target] = self.target_transformer(X_copy[self.target])
        most_recent_date = X_copy[self.timestamp].max()
        end_train_date = most_recent_date - pd.Timedelta(days=self.forecast_horizon)

        # Define MQCNNEstimator
        self.model = MQCNNEstimator(freq=self.freq, prediction_length=self.forecast_horizon,
                                    context_length=self.lookback, add_time_feature=self.add_time_feature,
                                    use_feat_static_cat=self.use_feat_static_cat,
                                    cardinality=self.cardinality_static_cat,
                                    use_feat_dynamic_real=self.use_feat_dynamic_real,
                                    use_past_feat_dynamic_real=self.use_past_feat_dynamic_real,
                                    quantiles=self.quantiles,
                                    **params_mqcnn)

        # Scale Dynamic Features
        if self.dynamic_feature_scaler is not None:
            train_df = X_copy[X_copy[self.timestamp] <= end_train_date].sort_values([self.item_id])
            self.dynamic_feature_scaler.fit(train_df[self.feat_dynamic_real + self.past_feat_dynamic_real])
            X_copy[self.feat_dynamic_real + self.past_feat_dynamic_real] = \
                self.dynamic_feature_scaler.transform(X_copy[self.feat_dynamic_real + self.past_feat_dynamic_real])

        dataset = Gluon_PandasDataset.from_long_dataframe(X_copy, timestamp=self.timestamp, freq=self.freq,
                                                          target=self.target,
                                                          item_id=self.item_id,
                                                          feat_static_cat=self.feat_static_cat,
                                                          feat_dynamic_real=self.feat_dynamic_real,
                                                          past_feat_dynamic_real=self.past_feat_dynamic_real)

        splitter = Gluon_DateSplitter(date=pd.Period(end_train_date, freq="1D"))
        train_dataset, val_template = splitter.split(dataset)
        validation = val_template.generate_instances(prediction_length=self.forecast_horizon)
        val_dataset = [entry[0] for entry in validation]

        self.predictor = self.model.train(training_data=train_dataset, validation_data=val_dataset)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.predictor

    def predict(self, X_test: pd.DataFrame, point_pred_mean_instead_median: bool = False,
                prediction_types: Optional[List] = None, verbose: bool = False,
                **kwargs):

        """
        predict method computes confidence intervals and points predictions with MQCNN model
        prediction is done for last forecast_horizon timesteps of input X_test (until fct = Forecast Creation Time)

        Args:
            X_test: pd.Dataframe
                dataframe with features
            point_pred_mean_instead_median: bool
                if set to True the point prediction will be
            prediction_types: list
                predictions to return. Possible values: pointestimates, quantiles
            **kwargs:
                additional arguments for predictor.predict()

        Returns: dict
            under key PredEnum.POINT_ESTIMATES is np.array of shape (#timeseries, #forecast_horizon, 1)
            under key PredEnum.QUANTILES is dictionary, which contains under every quantile key predictions of shape
            (#timeseries, #forecast_horizon)

        """

        start_time = time.perf_counter()
        if prediction_types is None:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES]

        X_test_copy = X_test.copy()
        X_test_copy[self.target] = self.target_transformer(X_test_copy[self.target])
        most_recent_date = X_test_copy[self.timestamp].max()
        end_train_date = most_recent_date - pd.Timedelta(days=self.forecast_horizon)

        # Scale Dynamic Features
        if self.dynamic_feature_scaler is not None:
            X_test_copy[self.feat_dynamic_real + self.past_feat_dynamic_real] = \
                self.dynamic_feature_scaler.transform(X_test_copy[self.feat_dynamic_real + self.past_feat_dynamic_real])

        dataset = Gluon_PandasDataset.from_long_dataframe(X_test_copy, timestamp=self.timestamp, freq=self.freq,
                                                          target=self.target, item_id=self.item_id,
                                                          feat_static_cat=self.feat_static_cat,
                                                          feat_dynamic_real=self.feat_dynamic_real,
                                                          past_feat_dynamic_real=self.past_feat_dynamic_real)

        splitter = Gluon_DateSplitter(date=pd.Period(end_train_date, freq="1D"))
        train_dataset, _ = splitter.split(dataset)

        forecast_it = self.predictor.predict(dataset=train_dataset)
        forecast_list = list(forecast_it)

        predictions = {}
        if PredEnum.POINT_ESTIMATES in prediction_types:
            if point_pred_mean_instead_median:
                point_pred = np.array([forecast.mean for forecast in forecast_list])
            else:
                point_pred = np.array([forecast.quantile(inference_quantile=0.5) for forecast in forecast_list])
            point_pred = point_pred[:, :, np.newaxis]
            predictions[PredEnum.POINT_ESTIMATES] = self.inverse_target_transformer(point_pred)

        if PredEnum.QUANTILES in prediction_types:
            quantile_fc = {}
            for quantile in self.quantiles:
                single_quantile_fc = np.array([np.transpose(np.array(forecast.quantile(inference_quantile=quantile)))
                                               for forecast in forecast_list])
                quantile_fc[quantile] = self.inverse_target_transformer(single_quantile_fc)
            predictions[PredEnum.QUANTILES] = quantile_fc

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test: List[np.array], predictions, prediction_types: Optional[List] = None,
                verbose: bool = False, confidence_interval_quantiles: Optional[List[float]] = None):
        """
        metrics method computes all possible metrics(defined in Abstract Model class)
        given the predictions obtained in predict method
        Args:
            y_test: np.array
                contains ground truth of shape (#timeseries, #forecast_horizon, 1)
            predictions: dict
                every key contains predictions
                POINT_ESIMATES: shape (#timeseries, #forecast_horizon, 1)
                QUANTILES: dictionary where every key is one quantile of shape (#timeseries, #forecast_horizon, 1)
            prediction_types: list
                specify for which prediction types we want to calculate metrics
            confidence_interval_quantiles: List[float]
                quantiles for which confidence intervals shall be computed. Must be contained in computed quantiles.
        Returns: dict
                contains metrics
        """

        if prediction_types is None:
            prediction_types = list(predictions.keys())

        prediction_types_super = list(set(prediction_types) & {PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES})

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.POINT_ESTIMATES in prediction_types_super:
            predictions_reshaped[PredEnum.POINT_ESTIMATES] = np.reshape(predictions[PredEnum.POINT_ESTIMATES],
                                                                        newshape=(-1, 1))

        if PredEnum.QUANTILES in prediction_types_super:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(
                predictions,
                confidence_interval_quantiles,
                self.quantiles
            )

        y_test_reshaped = np.reshape(y_test, newshape=(-1, 1))

        return super().metrics(y_test_reshaped, predictions_reshaped, prediction_types_super)

    @staticmethod
    def obtain_y_test_out_of_X_test(X_test, forecast_horizon, timestamp, target, item_id):
        """
        Helper method to obtain y_test for metrics method directly out of X_test. Input has to be the same dataframe
        as for predict method
        Args:
            X_test: pd.Dataframe
                dataframe with features
            forecast_horizon: int
                Length of the prediction
            timestamp: str
                column with time information
            target: str
                Name of the target column
            item_id: str
                Name of the column that, when grouped by, gives the different time series.

        Returns:
            y_test: np.array
                array with ground truth for forecast horizon for every time series
                shape (number of timeseries, forecast_horizon)
        """
        y_test = []
        for _, sliced in X_test.groupby(item_id, observed=True):
            split_date = X_test[timestamp].max() - pd.Timedelta(days=forecast_horizon)
            single_y = sliced[target][sliced[timestamp] > split_date].to_numpy()
            y_test.append(single_y)

        return np.array(y_test)
