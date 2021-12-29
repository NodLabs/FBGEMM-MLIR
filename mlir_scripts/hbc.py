from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import sys
from absl import app
import time

import numpy as np
import os
import tempfile
import tensorflow as tf

class HistogramBinningCalibrationByFeature(tf.Module):
    def __init__(self, num_segments, num_bins):
        super(HistogramBinningCalibrationByFeature, self).__init__()
        # HistogramBinningCalibrationByFeature
        self._num_segments = num_segments
        self._num_bins = num_bins
        _num_interval = (self._num_segments + 1) * self._num_bins
        _lower_bound = 0
        _upper_bound = 1
        l, u = _lower_bound, _upper_bound
        w = (u - l) / self._num_bins
        self.step = w
        self._boundaries = tf.range(l + w, u - w / 2, w)
        self._bin_num_examples = tf.zeros([_num_interval], dtype=tf.float64)
        self._bin_num_positives = tf.zeros([_num_interval], dtype=tf.float64)
        self._bin_ids = tf.range(_num_interval)
        self._iteration = 0

    @tf.function(input_signature=[
        tf.TensorSpec([5000,], tf.int64),
        tf.TensorSpec([5000, 1], tf.int64),
        tf.TensorSpec([5000, 1], tf.float32)
    ])
    def forward(self, segment_value, offsets, logit):
        origin_prediction = tf.sigmoid(logit - 0.9162907600402832)
        # HistogramBinningCalibrationByFeature
        _3251 = tf.reshape(origin_prediction, (-1,))  # Reshape
        dense_segment_value = tf.zeros(tf.size(logit), dtype=segment_value.dtype)
        tmp = tf.gather_nd(segment_value, offsets) + 1
        _3257 = tf.scatter_nd(offsets, tmp, segment_value.shape)
        _3253 = tf.reshape(offsets, (-1,))
        _3258 = tf.reshape(_3257, (-1,))  # Reshape
        _3259 = tf.cast(_3258, tf.int64)  # Cast
        _3260 = tf.zeros(_3253.shape, dtype=tf.int64)  # ConstantFill
        _3261 = tf.ones(_3253.shape, dtype=tf.int64)  # ConstantFill
        _3262 = tf.greater(_3259, self._num_segments)  # GT
        _3263 = tf.greater(_3260, _3259)  # GT
        _3264 = _3253 == _3261  # EQ
        _3265 = tf.where(_3262, _3260, _3259)  # Conditional
        _3266 = tf.where(_3263, _3260, _3265)  # Conditional
        _3267 = tf.where(_3264, _3266, _3260)  # Conditional
        _3268 = tf.math.ceil(_3251 / self.step) - 1
        _3269 = tf.cast(_3268, tf.int64)  # Cast
        _3270 = _3267 * self._num_bins  # Mul
        _3271 = _3269 + _3270  # Add
        _3272 = tf.reshape(tf.cast(_3271, tf.int32), (-1,))  # Cast
        _3273 = tf.gather(self._bin_num_positives, tf.cast(_3272, tf.int64))  # Gather
        _3274 = tf.gather(self._bin_num_examples, tf.cast(_3272, tf.int64))  # Gather
        _3275 = _3273 / _3274  # Div
        _3276 = tf.cast(_3275, tf.float32)  # Cast
        _3277 = _3276 * 0.9995 + _3251 * 0.0005  # WeightedSum
        _3278 = tf.greater(_3274, 10000.0)  # GT
        _3279 = tf.where(_3278, _3277, tf.cast(_3251, tf.float32))  # Conditional
        prediction = tf.reshape(_3279, (-1, 1))  # Reshape
        return prediction

if __name__ == "__main__":
    # Compile the model using IREE
    data_type = tf.float32
    num_logits=5000
    num_segments = 42
    logit = tf.random.uniform(shape=[num_logits], dtype=data_type)
    segment_values = tf.random.uniform(shape=[num_logits], minval=0, maxval=num_segments, dtype=tf.int64)
    lengths = tf.ones(5000, dtype=tf.int64)
    num_bins = 5000
    lengths = tf.expand_dims(lengths, 1)
    logit = tf.expand_dims(logit, 1)

    offsets = np.cumsum(lengths)-lengths[0]

    compiler_module = tfc.compile_module(HistogramBinningCalibrationByFeature(num_segments, num_bins), exported_names = ["forward"], import_only=True)
    flatbuffer_blob = compile_str(compiler_module, target_backends=["dylib-llvm-aot"])
    # Save module as MLIR file in a directory
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config("dylib",tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    HbcCompiled = ctx.modules.module

    print(tf.squeeze(HbcCompiled.forward(segment_values, tf.expand_dims(tf.convert_to_tensor(offsets),1), logit)))

