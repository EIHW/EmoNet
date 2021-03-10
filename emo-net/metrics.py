def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.stats import shapiro, pearsonr
from sklearn.metrics import recall_score, make_scorer, accuracy_score, f1_score, mean_squared_error, classification_report, confusion_matrix, multilabel_confusion_matrix, precision_score, roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics.scorer import _BaseScorer
from statistics import pstdev, mean
from typing import Dict, List, ClassVar, Set
from math import sqrt
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


def mask_metric(func):
    def mask_metric_function(*args, **kwargs):
        mask = np.not_equal(kwargs['y_true'], -1).astype(float)
        kwargs['y_true'] = (kwargs['y_true'] * mask)
        kwargs['y_pred'] = (kwargs['y_pred'] * mask)
        return func(*args, **kwargs)

    return mask_metric_function


def optimal_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def compute_binary_cutoffs(y_true, y_pred):
    if y_true.shape == y_pred.shape and len(y_true.shape) == 1:  # 2 classes
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return [optimal_threshold(fpr, tpr, thresholds)]
    elif y_true.shape == y_pred.shape and len(y_true.shape) == 2:  # multilabel
        fpr_tpr_thresholds = [
            roc_curve(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
        return [optimal_threshold(*x) for x in fpr_tpr_thresholds]


class ClassificationMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 labels: List = None,
                 validation_generator=None,
                 validation_data=None,
                 multi_label=False,
                 partition='validation',
                 period=1,
                 dataset_name='default'):
        super().__init__()
        if labels is not None:
            self.labels = {name: index for index, name in enumerate(labels)}
            self.binary = (len(labels) == 2)

        elif validation_generator is not None:
            self.labels = validation_generator.class_indices
            self.binary = len(self.labels) == 2
        self.validation_generator = validation_generator
        self.validation_data = validation_data
        self.multi_label = multi_label
        self.partition = partition
        self.keras_metric_quantities = KERAS_METRIC_QUANTITIES
        self.dataset_name = dataset_name

        self._binary_cutoffs = []
        self._data = []
        self.period = period

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            if self.validation_generator is None:
                X_val, y_val = self.validation_data[0], self.validation_data[1]
                y_pred = np.asarray(self.model.predict(X_val))
            else:
                y_pred = np.squeeze(self.model.predict(self.validation_generator))
                y_val = np.squeeze(self.validation_generator.categorical_classes)
            logs = self.compute_metrics(y_val,
                                        y_pred,
                                        multi_label=self.multi_label,
                                        binary=self.binary,
                                        labels=sorted(
                                            self.labels.values()),
                                        prefix=f'{self.partition}',
                                        logs=logs,
                                        target_names=sorted(self.labels.keys()))

            return

    def get_data(self):
        return self._data

    def compute_metrics(self,
                        y_val,
                        y_pred,
                        multi_label=False,
                        binary=False,
                        labels=None,
                        prefix='',
                        logs={},
                        target_names=None):
        eval_string = f'\nEvaluation results for partition {self.partition} of dataset {self.dataset_name}:\n'
        all_classes_present = np.all(np.any(y_val > 0, axis=0))
        if multi_label:
            binary_cutoffs = compute_binary_cutoffs(y_true=y_val,
                                                    y_pred=y_pred)
            self._binary_cutoffs.append(binary_cutoffs)
            logger.info(f'Optimal cutoffs: {binary_cutoffs}')
        else:
            binary_cutoffs = None
        y_val_t, y_pred_t = ClassificationMetric._transform_arrays(
            y_true=y_val,
            y_pred=y_pred,
            multi_label=multi_label,
            binary=binary,
            binary_cutoffs=binary_cutoffs)
        eval_string += classification_report(y_val_t,
                                            y_pred_t,
                                            target_names=target_names)
        if self.multi_label:
            eval_string += '\n'+ str(multilabel_confusion_matrix(y_true=y_val_t,
                                            y_pred=y_pred_t,
                                            labels=labels))
        else:
            conf_matrix = confusion_matrix(y_true=np.argmax(y_val_t, axis=1) if
                                    len(y_val_t.shape) > 1 else y_val_t,
                                    y_pred=np.argmax(y_pred_t, axis=1) if
                                    len(y_pred_t.shape) > 1 else y_pred_t,
                                    labels=labels)
            eval_string += '\n'+ str(conf_matrix)
            logs[f'{prefix}confusion_matrix'] = conf_matrix
        for i, cm in enumerate(CLASSIFICATION_METRICS):
            if all_classes_present or not (cm == ROC_AUC or cm == PR_AUC):
                if cm.needs_categorical:
                    metric = cm.compute(y_true=y_val_t,
                                        y_pred=y_pred_t,
                                        labels=labels,
                                        binary=binary,
                                        multi_label=multi_label,
                                        binary_cutoffs=binary_cutoffs)
                else:
                    metric = cm.compute(y_true=y_val,
                                        y_pred=y_pred,
                                        labels=labels,
                                        binary=binary,
                                        multi_label=multi_label,
                                        binary_cutoffs=binary_cutoffs)
                metric_value = metric.value
                eval_string += f'\n{prefix} {cm.description}: {metric_value}'
                if not self._data:  # first recorded value
                    self._data.append({
                        f'{self.keras_metric_quantities[cm]}/{prefix}':
                        metric_value,
                    })
                elif i == 0 and self._data and f'{self.keras_metric_quantities[cm]}/{prefix}' in self._data[
                        -1].keys():
                    self._data.append({
                        f'{self.keras_metric_quantities[cm]}/{prefix}':
                        metric_value,
                    })
                else:
                    self._data[-1][
                        f'{self.keras_metric_quantities[cm]}/{prefix}'] = metric_value
                if len(
                        self._data
                ) > 1:  # this is the second epoch and metrics have been recorded for the first epoch
                    cur_best = self._data[-2][
                        f'{self.keras_metric_quantities[cm]}_best/{prefix}']
                else:  # this is the first epoch
                    cur_best = metric_value

                new_best = metric_value if metric > cm(
                    value=cur_best) else cur_best

                self._data[-1][
                    f'{self.keras_metric_quantities[cm]}_best/{prefix}'] = new_best

                logs[f'{self.keras_metric_quantities[cm]}/{prefix}'] = metric_value
                logs[f'{self.keras_metric_quantities[cm]}_best/{prefix}'] = new_best
            else:
                logger.info(
                    f'Not all classes occur in the validation data, skipping ROC AUC and PR AUC.'
                )
        logger.info(eval_string)
        return logs


class RegressionMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))

        for metric in REGRESSION_METRICS:
            metric_value = metric.compute(y_true=y_val, y_pred=y_predict).value
            self._data.append({f'val_{metric.__name__.lower()}': metric_value})
            logs[f'val_{metric.__name__.lower()}'] = metric_value
        return

    def get_data(self):
        return self._data


@dataclass(order=True)
class Metric(ABC):
    sort_index: float = field(init=False, repr=False)
    description: ClassVar[str] = 'Metric'
    key: ClassVar[str] = 'M'
    value: float
    scikit_scorer: ClassVar[_BaseScorer] = field(init=False, repr=False)
    greater_is_better: ClassVar[bool] = True

    def __post_init__(self):
        self.sort_index = self.value if self.greater_is_better else -self.value


@dataclass(order=True)
class ClassificationMetric(Metric, ABC):
    multi_label: bool = False
    binary: bool = False
    average: ClassVar[str] = None
    needs_categorical: ClassVar[bool] = True

    @classmethod
    @mask_metric
    @abstractmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> Metric:
        pass

    @staticmethod
    def _transform_arrays(y_true: np.array,
                          y_pred: np.array,
                          multi_label: bool,
                          binary: bool,
                          binary_cutoffs: List[float] = None
                          ) -> (np.array, np.array):
        if binary:
            if len(y_pred.shape) > 1:
                y_pred = np.reshape(y_pred, -1)
            if len(y_true.shape) > 1:
                y_true = np.reshape(y_true, -1)
            assert (
                y_true.shape == y_pred.shape and len(y_true.shape) == 1
            ), f'Shapes of predictions and labels for binary classification should conform to (n_samples,) but received {y_pred.shape} and {y_true.shape}.'
            #if binary_cutoffs is None:
            binary_cutoffs = 0.5
            #y_pred_transformed = np.zeros_like(y_pred, dtype=int)
            #y_pred_transformed[y_pred > binary_cutoffs[0]] = 1
            y_pred_transformed = np.where(y_pred > binary_cutoffs, 1, 0)
            y_true_transformed = y_true
            
        elif multi_label:
            assert (
                y_true.shape == y_pred.shape
            ), f'Shapes of predictions and labels for multilabel classification should conform to (n_samples, n_classes) but received {y_pred.shape} and {y_true.shape}.'
            if binary_cutoffs is None:
                binary_cutoffs = compute_binary_cutoffs(y_true, y_pred)
            # y_pred_transformed = np.zeros_like(y_pred, dtype=int)
            # y_pred_transformed[y_pred > 0.5] = 1
            y_pred_transformed = np.where(y_pred > binary_cutoffs, 1, 0)
            y_true_transformed = y_true
        else:
            if y_true.shape[1] > 1:
                y_true_transformed = np.zeros_like(y_true)
                y_true_transformed[range(len(y_true)), y_true.argmax(1)] = 1
            if y_pred.shape[1] > 1:
                y_pred_transformed = np.zeros_like(y_pred)
                y_pred_transformed[range(len(y_pred)), y_pred.argmax(1)] = 1
            assert (
                y_true.shape == y_pred.shape
            ), f'Shapes of predictions and labels for multiclass classification should conform to (n_samples,n_classes) but received {y_pred.shape} and {y_true.shape}.'
        return y_true_transformed, y_pred_transformed


@dataclass(order=True)
class RegressionMetric(Metric, ABC):
    @staticmethod
    @abstractmethod
    def compute(y_true: np.array, y_pred: np.array) -> Metric:
        pass


@dataclass(order=True)
class MicroRecall(ClassificationMetric):
    description: ClassVar[str] = 'Micro Average Recall'
    average: ClassVar[str] = 'micro'
    key: ClassVar[str] = 'Recall/Micro'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(recall_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        score = recall_score(y_true=y_true,
                             y_pred=y_pred,
                             labels=labels,
                             average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class UAR(MicroRecall):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Unweighted Average Recall'
    key: ClassVar[str] = 'Recall/Macro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(recall_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class Accuracy(ClassificationMetric):
    description: ClassVar[str] = 'Accuracy'
    key: ClassVar[str] = 'acc'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(accuracy_score)
    greater_is_better: ClassVar[bool] = True

    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        # y_true, y_pred = ClassificationMetric._transform_arrays(
        #     y_true=y_true,
        #     y_pred=y_pred,
        #     multi_label=multi_label,
        #     binary=binary,
        #     binary_cutoffs=binary_cutoffs)
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MacroF1(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Macro Average F1'
    key: ClassVar[str] = 'F1/Macro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(f1_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        # y_true, y_pred = ClassificationMetric._transform_arrays(
        #     y_true=y_true,
        #     y_pred=y_pred,
        #     multi_label=multi_label,
        #     binary=binary,
        #     binary_cutoffs=binary_cutoffs)
        score = f1_score(y_true=y_true,
                         y_pred=y_pred,
                         labels=labels,
                         average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MicroF1(MacroF1):
    average: ClassVar[str] = 'micro'
    description: ClassVar[str] = 'Micro Average F1'
    key: ClassVar[str] = 'F1/Micro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(f1_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class MacroPrecision(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Macro Average Precision'
    key: ClassVar[str] = 'Prec/Macro'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(precision_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        # y_true, y_pred = ClassificationMetric._transform_arrays(
        #     y_true=y_true,
        #     y_pred=y_pred,
        #     multi_label=multi_label,
        #     binary=binary,
        #     binary_cutoffs=binary_cutoffs)
        score = precision_score(y_true=y_true,
                                y_pred=y_pred,
                                labels=labels,
                                average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MicroPrecision(MacroPrecision):
    average: ClassVar[str] = 'micro'
    description: ClassVar[str] = 'Micro Average Prec'
    key: ClassVar[str] = 'Prec/Micro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(precision_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class ROC_AUC(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[
        str] = 'Area Under the Receiver Operating Characteristic Curve'
    key: ClassVar[str] = 'ROC AUC'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(roc_auc_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True
    needs_categorical: ClassVar[bool] = False

    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        score = roc_auc_score(y_true=y_true,
                              y_score=y_pred,
                              average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class PR_AUC(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Area Under the Precision Recall Curve'
    key: ClassVar[str] = 'PR AUC'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(average_precision_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True
    needs_categorical: ClassVar[bool] = False


    @classmethod
    @mask_metric
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        score = average_precision_score(y_true=y_true,
                                        y_score=y_pred,
                                        average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MSE(RegressionMetric):
    description: ClassVar[str] = 'Mean Squared Error'
    key: ClassVar[str] = 'mse'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(mean_squared_error,
                                                       greater_is_better=False)
    greater_is_better: ClassVar[bool] = False

    @staticmethod
    def compute(y_true: np.array, y_pred: np.array) -> RegressionMetric:
        score = mean_squared_error(y_true=y_true, y_pred=y_pred)
        return MSE(value=score)


def pearson_correlation_coefficient(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


@dataclass(order=True)
class PCC(RegressionMetric):
    description: ClassVar[str] = 'Pearson\'s Correlation Coeffiecient'
    key: ClassVar[str] = 'pcc'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(
        pearson_correlation_coefficient, greater_is_better=True)
    greater_is_better: ClassVar[bool] = True

    @staticmethod
    def compute(y_true: np.array, y_pred: np.array) -> RegressionMetric:
        score = pearson_correlation_coefficient(y_true=y_true, y_pred=y_pred)
        return PCC(value=score)


def concordance_correlation_coefficient(y_true, y_pred):
    ccc = 2 * pearson_correlation_coefficient(y_true=y_true, y_pred=y_pred) / (
        np.var(y_true) + np.var(y_pred) +
        (np.mean(y_true) - np.mean(y_pred))**2)
    return ccc


@dataclass(order=True)
class CCC(RegressionMetric):
    description: ClassVar[str] = 'Concordance Correlation Coeffiecient'
    key: ClassVar[str] = 'ccc'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(
        pearson_correlation_coefficient, greater_is_better=True)
    greater_is_better: ClassVar[bool] = True

    @staticmethod
    def compute(y_true: np.array, y_pred: np.array) -> RegressionMetric:
        score = concordance_correlation_coefficient(y_true=y_true,
                                                    y_pred=y_pred)
        return CCC(value=score)


def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


@dataclass(order=True)
class RMSE(RegressionMetric):
    description: ClassVar[str] = 'Root Mean Squared Error'
    key: ClassVar[str] = 'rmse'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(root_mean_squared_error,
                                                       greater_is_better=False)
    greater_is_better: ClassVar[bool] = False

    @staticmethod
    def compute(y_true: np.array, y_pred: np.array) -> RegressionMetric:
        score = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        return RMSE(value=score)


@dataclass
class MetricStats():
    mean: float
    standard_deviation: float
    normality_tests: Dict[str, tuple]


def compute_metric_stats(metrics: List[Metric]) -> MetricStats:
    metric_values = [metric.value for metric in metrics]
    normality_tests = dict()
    if len(metric_values) > 2:
        normality_tests['Shapiro-Wilk'] = shapiro(metric_values)
    return MetricStats(mean=mean(metric_values),
                       standard_deviation=pstdev(metric_values),
                       normality_tests=normality_tests)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


CLASSIFICATION_METRICS = all_subclasses(ClassificationMetric)
REGRESSION_METRICS = all_subclasses(RegressionMetric)
ALL_METRICS = all_subclasses(Metric)

# scorers for use in scikit-learn classifiers
# SCIKIT_CLASSIFICATION_SCORERS = {
#     MicroRecall.__name__: MicroRecall.scikit_scorer
#     UAR.__name__: UAR.scikit_scorer,
#     Accuracy.__name__: Accuracy.scikit_scorer,
#     MicroF1.__name__: MicroF1.scikit_scorer,
#     MacroF1.__name__: MacroF1.scikit_scorer,
#     MicroPrecision.__name__: MicroPrecision.scikit_scorer,
#     MacroPrecision.__name__: MacroPrecision.scikit_scorer,
#     ROC_AUC.__name__: ROC_AUC.scikit_scorer,
#     PR_AUC.__name__: PR_AUC.scikit_scorer

# }
SCIKIT_CLASSIFICATION_SCORERS = {
    M.__name__: M.scikit_scorer
    for M in CLASSIFICATION_METRICS if M != ROC_AUC and M != PR_AUC
}

SCIKIT_CLASSIFICATION_SCORERS_EXTENDED = {
    M.__name__: M.scikit_scorer
    for M in CLASSIFICATION_METRICS
}

# scorers for use in scikit-learn regressors
# SCIKIT_REGRESSION_SCORERS = {
#     MSE.__name__: MSE.scikit_scorer,
#     RMSE.__name__: RMSE.scikit_scorer,
#     PCC.__name__: PCC.scikit_scorer,
#     CCC.__name__: CCC.scikit_scorer
# }

SCIKIT_REGRESSION_SCORERS = {
    M.__name__: M.scikit_scorer
    for M in REGRESSION_METRICS
}

# KERAS_METRIC_QUANTITIES = {
#     F1: 'val_fmeasure_acc',
#     Accuracy: 'val_acc',
#     UAR: 'val_uar',
#     MSE: 'val_mean_squared_error',
#     Precision: 'val_precision',
#     ROC_AUC: 'val_roc_auc',
#     PR_AUC: 'val_pr_auc'
# }

KERAS_METRIC_QUANTITIES = {
    M: f'val_{"_".join(M.key.lower().split(" "))}'
    for M in ALL_METRICS
}

KERAS_METRIC_MODES = {
    M: 'max' if M.greater_is_better else 'min'
    for M in ALL_METRICS
}

# CLASSIFICATION_METRICS = [
#     MicroRecall, UAR, Accuracy, MicroF1, MacroF1, MicroPrecision,
#     MacroPrecision, ROC_AUC, PR_AUC
# ]
# REGRESSION_METRICS = [MSE, RMSE, PCC, CCC]

KEY_TO_METRIC = {metric.__name__: metric for metric in ALL_METRICS}
