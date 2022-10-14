from helpers import windowed_range
import tensorflow as tf
import numpy as np

#--------------------------------------------------------------------------------#
# Functions                                                                      #
#--------------------------------------------------------------------------------#

def _init_bounds(num_labels):
    # create empty numpy arrays for bounds:
    t0 = np.empty(num_labels)
    t1 = np.empty(num_labels)

    # fill numpy arrays with class bounds:
    for start, stop, i in windowed_range(num_labels):
        t0[i] = start
        t1[i] = stop

    # convert arays to tensorflow:
    return tf.constant(num_labels, dtype=tf.int32), tf.constant(t0, dtype=tf.float32), tf.constant(t1, dtype=tf.float32)

def _to_one_hot(y_true, y_pred, t0, t1, n_labels):
    # find class of true label in one-hot-encoding:
    l = tf.zeros_like(y_true, dtype=tf.int32)
    for i in range(n_labels):
        lower = tf.gather(t0, [i])
        upper = tf.gather(t1, [i])
        l = tf.where(tf.math.logical_and(y_true>lower, y_true<=upper), tf.cast(i, dtype=tf.int32), tf.cast(l, dtype=tf.int32))
    l = tf.keras.backend.one_hot(l, num_classes=n_labels)
    l = tf.reshape(l, (-1, n_labels))
    l = tf.slice(l, [0,1], [-1,-1])

    # calculate error of prediction for each class:
    e = tf.math.abs(tf.range(n_labels, dtype=tf.float32) - (tf.cast(n_labels-1, dtype=tf.float32) * y_pred))
    e = tf.keras.backend.clip(e, 0., 1.)
    e = tf.reshape(e, (-1, n_labels))
    e = tf.slice(e, [0,1], [-1,-1])

    return l, 1-e

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

        # init labels and boundaries:
        self.n_labels, self.t0, self.t1 = _init_bounds(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # create one-hot-representation:
        true, pred = _to_one_hot(y_true, y_pred, self.t0, self.t1, self.n_labels)

        return super().update_state(
            true,
            pred,
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

        # init labels and boundaries:
        self.n_labels, self.t0, self.t1 = _init_bounds(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # create one-hot-representation:
        true, pred = _to_one_hot(y_true, y_pred, self.t0, self.t1, self.n_labels)

        return super().update_state(
            true,
            pred,
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

        # init labels and boundaries:
        self.n_labels, self.t0, self.t1 = _init_bounds(num_labels)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # create one-hot-representation:
        true, pred = _to_one_hot(y_true, y_pred, self.t0, self.t1, self.n_labels)

        return super().update_state(
            true,
            pred,
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

        # init labels and boundaries:
        self.n_labels, self.t0, self.t1 = _init_bounds(num_labels)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # create one-hot-representation:
        true, pred = _to_one_hot(y_true, y_pred, self.t0, self.t1, self.n_labels)

        self.precision.update_state(true, pred, sample_weight=sample_weight)
        self.recall.update_state(true, pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        if precision > 0. and recall > 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return -1.