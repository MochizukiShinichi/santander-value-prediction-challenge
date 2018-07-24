# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Define a Wide + Deep model for classification on structured data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import six
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import numpy as np
# Get unique data entries per column to calcualte hash size
with file_io.FileIO('bucketSize.csv', 'r') as file:
    _UNIQUE_COUNTS = file.read().split(',')
_HASH_BUCKET_SIZE = [3*int(i) for i in _UNIQUE_COUNTS[2:]]

# get feature types: int or float
with file_io.FileIO('features.csv', 'r') as file:
    features = file.read().split(',')
IS_WIDE = [True if _ == 'int64' else False for _ in features[2:]]

# Define the format of your input data including unused columns
FEATURE_COLUMN_NUM = 4991
FEATURE_COLUMNS = ['feature_{}'.format(i) for i in range(FEATURE_COLUMN_NUM)]
LABEL_COLUMN = 'target'
UNUSED_COLUMNS = ['ID']
CSV_COLUMNS = ['ID', 'target'] + FEATURE_COLUMNS
CSV_COLUMN_DEFAULTS = [tf.constant([],dtype=tf.string)] + [tf.constant([0.0], dtype=tf.float64)] + [tf.constant([0], dtype=tf.int64) if _ else tf.constant([0.0], dtype=tf.float64) for _ in IS_WIDE]

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
wide_columns = [tf.feature_column.categorical_column_with_hash_bucket(c, hash_bucket_size=_HASH_BUCKET_SIZE[i], dtype = tf.int64) for i, c in enumerate(FEATURE_COLUMNS) if IS_WIDE[i]]
deep_columns = [tf.feature_column.numeric_column(c) for i, c in enumerate(FEATURE_COLUMNS) if not IS_WIDE[i]]
INPUT_COLUMNS = wide_columns + deep_columns


def build_estimator(config, embedding_size=8, hidden_units=None):
  """Build a wide and deep model for predicting income category.

  Wide and deep models use deep neural nets to learn high level abstractions
  about complex features or interactions between such features.
  These models then combined the outputs from the DNN with a linear regression
  performed on simpler features. This provides a balance between power and
  speed that is effective on many structured data problems.

  You can read more about wide and deep models here:
  https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

  To define model we can use the prebuilt DNNCombinedLinearClassifier class,
  and need only define the data transformations particular to our dataset, and
  then
  assign these (potentially) transformed features to either the DNN, or linear
  regression portion of the model.

  Args:
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    embedding_size: int, the number of dimensions used to represent categorical
      features when providing them as inputs to the DNN.
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    learning_rate: float, the learning rate for the optimizer.
  Returns:
    A DNNCombinedLinearClassifier
  """

  return tf.estimator.DNNLinearCombinedRegressor(
      config=config,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      dnn_optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001,
          l2_regularization_strength=0.001)
  )


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})

SERVING_FUNCTIONS = {
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=1,
                      batch_size=200):
  """Generates features and labels for training or evaluation.
  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

  Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
  Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle:
    # Process the files in a random order.
    filename_dataset = filename_dataset.shuffle(len(filenames))
    
  # For each filename, parse it into one element per line, and skip the header
  # if necessary.
  dataset = filename_dataset.flat_map(
      lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines))
  
  dataset = dataset.map(parse_csv)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features, features.pop(LABEL_COLUMN)
