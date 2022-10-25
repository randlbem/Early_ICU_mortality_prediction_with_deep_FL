from os import truncate
from helpers import windowed_range
import tensorflow as tf
import numpy as np

#--------------------------------------------------------------------------------#
# Functions                                                                      #
#--------------------------------------------------------------------------------#

def _map_labels(y_true, n, t0, t1):
    # find class of true label in one-hot-encoding:
    l = tf.zeros_like(y_true, dtype=tf.int32)
    for i in range(n):
        lower = tf.gather(t0, [i])
        upper = tf.gather(t1, [i])
        l = tf.where(tf.math.logical_and(y_true>lower, y_true<=upper), tf.cast(i, dtype=tf.int32), tf.cast(l, dtype=tf.int32))
    l = tf.keras.backend.one_hot(l, num_classes=n)
    l = tf.reshape(l, (-1, n))
    l = tf.slice(l, [0,1], [-1,-1])

    return l

def _map_predictions(y_pred, n):
    # calculate probability of predictions belonging to a class:
    e = tf.math.abs(tf.range(n, dtype=tf.float32) - (tf.cast(n-1, dtype=tf.float32) * y_pred))
    e = tf.keras.backend.clip(e, 0., 1.)
    e = tf.reshape(e, (-1, n))
    e = tf.slice(e, [0,1], [-1,-1])

    return 1-e

def _init_graphs(num_labels):
    # create empty numpy arrays for bounds:
    t0 = np.empty(num_labels)
    t1 = np.empty(num_labels)

    # fill numpy arrays with class bounds:
    for start, stop, i in windowed_range(num_labels):
        t0[i] = start
        t1[i] = stop

    # create graph fuctions:
    map_labels = tf.function(
        func=lambda y_true: _map_labels(y_true, num_labels, tf.constant(t0, dtype=tf.float32), tf.constant(t1, dtype=tf.float32)),
        #input_signature=None,
        #autograph=True,
        jit_compile=truncate
    )
    map_predictions = tf.function(
        func=lambda y_pred: _map_predictions(y_pred, num_labels),
        #input_signature=None,
        #autograph=True,
        jit_compile=True
    )

    # return functions:
    return map_labels, map_predictions

#--------------------------------------------------------------------------------#
# Classes                                                                        #
#--------------------------------------------------------------------------------#

class ContinuousAUC(tf.keras.metrics.AUC):
    def __init__(self, num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, num_labels=2, from_logits=False):
        # init parent:
        super().__init__(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            #thresholds=thresholds,
            #multi_label=True,
            #num_labels=num_labels,
            #label_weights=None,
            from_logits=from_logits
        )

        # init tf graphs:
        self.map_labels, self.map_predictions = _init_graphs(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            self.map_labels(y_true),
            self.map_predictions(y_pred),
            sample_weight=sample_weight
        )


class ContinuousPrecision(tf.keras.metrics.Precision):
    def __init__(self, top_k=None, label_id=None, name=None, dtype=None, num_labels=2):
        # init parent:
        super().__init__(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name,
            dtype=dtype
        )

        # init tf graphs:
        self.map_labels, self.map_predictions = _init_graphs(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            self.map_labels(y_true),
            self.map_predictions(y_pred),
            sample_weight=sample_weight
        )


class ContinuousRecall(tf.keras.metrics.Recall):
    def __init__(self, top_k=None, label_id=None, name=None, dtype=None, num_labels=2):
        # init parent:
        super().__init__(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name,
            dtype=dtype
        )

        # init tf graphs:
        self.map_labels, self.map_predictions = _init_graphs(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            self.map_labels(y_true),
            self.map_predictions(y_pred),
            sample_weight=sample_weight
        )

class ContinuousF1(tf.keras.metrics.Metric):
    def __init__(self, top_k=None, label_id=None, name=None, dtype=None, num_labels=2):
        # init parent:
        super().__init__(
            name=name,
            dtype=dtype
        )
        # init precision metric:
        self.precision = tf.keras.metrics.Precision(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name+'_precision',
            dtype=dtype
        )
        # init recall metric:
        self.recall = tf.keras.metrics.Recall(
            #thresholds=thresholds,
            top_k=top_k,
            class_id=label_id,
            name=name+'_recall',
            dtype=dtype
        )

        # init tf graphs:
        self.map_labels, self.map_predictions = _init_graphs(num_labels)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # create one-hot-representation:
        true = self.map_labels(y_true)
        pred = self.map_predictions(y_pred)

        self.precision.update_state(true, pred, sample_weight=sample_weight)
        self.recall.update_state(true, pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        if precision > 0. and recall > 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return -1.