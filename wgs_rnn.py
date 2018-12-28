"""
Recurrent Neural Network for bacteria classification based on whole genome
sequences. Genomic data retrieved from GeneBank

[1] https://www.tensorflow.org/tutorials/
[2] http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
[3] https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
[4] https://www.tensorflow.org/tutorials/seq2seq#background_on_the_attention_mechanism
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import wgs_dataset

import numpy as np
import tensorflow as tf

import os, gc, time, shutil

# Random seed and TF logging
tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(43)

# dataset
wgs = wgs_dataset.wgs()

# Constants
MODEL_DIR        = './rnn_model'
ITERATIONS       = 3600

BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3

HIDDEN_UNITS     = 128
NUM_CLASSES      = None

ATTENTION        = True
LSTM             = False

# Model function for the RNN
def rnn_model_fn(features, labels, mode):
  # Input layer. Shape (batch, num inputs, sequence length)
  input_layer = tf.unstack(features["data"], axis=2)
  
  # Hidden layer
  if LSTM:
    forward_cell = tf.contrib.rnn.LSTMCell(
      num_units=HIDDEN_UNITS,
      use_peepholes = True,
      initializer=tf.contrib.layers.xavier_initializer(),
      activation=tf.nn.tanh
      )
    backward_cell = tf.contrib.rnn.LSTMCell(
      num_units=HIDDEN_UNITS,
      use_peepholes = True,
      initializer=tf.contrib.layers.xavier_initializer(),
      activation=tf.nn.tanh
      )
  else:
    forward_cell = tf.contrib.rnn.GRUCell(
      num_units=HIDDEN_UNITS,
      activation=tf.nn.tanh,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )
    backward_cell = tf.contrib.rnn.GRUCell(
      num_units=HIDDEN_UNITS,
      activation=tf.nn.tanh,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )
  
  rnn_outputs, output_state_fw, output_state_bw = \
    tf.nn.static_bidirectional_rnn(
    cell_fw=forward_cell,
    cell_bw=backward_cell,
    inputs=input_layer,
    initial_state_fw=None,
    initial_state_bw=None,
    dtype=tf.float32,
    sequence_length=None,
    scope=None
    )
  
  if ATTENTION:
    last_output = rnn_outputs[-1]
    attention_num_units = last_output.shape[-1]
    attention_state = tf.concat([output_state_fw, output_state_bw], axis=1)
    attention_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
      num_units=attention_num_units, memory=attention_outputs)
    alignments, _ = attention_mechanism(
      query=last_output, state=attention_state)
    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])
    
    if LSTM: 
      # LSTM state is a tuple (c, h) by default
      output_attention = tf.concat(
        [context, output_state_fw[0], output_state_bw[0]], 1)
    else:
      output_attention = tf.concat(
        [context, output_state_fw, output_state_bw], 1)
    
    output_state = tf.layers.dense(
      inputs=output_attention, 
      units=4*HIDDEN_UNITS, 
      activation=tf.nn.tanh,
      use_bias=True, 
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      )
    
  else:
    output_state = tf.concat([output_state_fw, output_state_bw], axis=1)
    
  # Classification layer
  output_state = tf.layers.dropout(
    inputs=output_state, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
  output = tf.layers.dense(
    inputs=output_state, 
    units=NUM_CLASSES, 
    activation=None,
    use_bias=True,
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    )
  
  # Generate predictions
  predictions = {
    "labels": tf.argmax(input=output, axis=1),
    "alignments": alignments,
    "probabilities": tf.contrib.layers.softmax(output)
    }
  
  # Prediction mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  # Loss function (TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=labels, depth=NUM_CLASSES)
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=output)
  
  # Print number of parameters
  #print('\nNumber of trainable variables: ')
  #print(np.sum(
  #  [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
  #quit()
  
  # Training operation
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer( 
      learning_rate=LEARNING_RATE,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-08
      )
    train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # Evaluation metrics
  accuracy  = tf.metrics.accuracy(
    labels=labels, predictions=predictions["labels"])
  precision  = tf.metrics.precision(
    labels=labels, predictions=predictions["labels"])
  recall  = tf.metrics.recall(
    labels=labels, predictions=predictions["labels"])
  mean_iou = tf.metrics.mean_iou(
    labels=labels, predictions=predictions["labels"], num_classes=NUM_CLASSES)
  
  # Retrieve confusion matrix calculated in mean_iou and append mean_iou update
  # operation
  total_confusion_matrix = (tf.to_int32(
    tf.get_default_graph().get_tensor_by_name(
    'mean_iou/total_confusion_matrix:0')), mean_iou[1])
  
  eval_metric_ops = {
    "accuracy"            : accuracy,
    "precision"           : precision,
    "recall"              : recall,
    "total_conf_matrix\n" : total_confusion_matrix
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Train and eval function
steps = []
accuracies = []
def continuous_eval_predicate_fn(eval_results):
  # Insert delay
  time.sleep(60)
  
  # None argument for the first evaluation
  if not eval_results:
    return True
    
  steps.append(eval_results["global_step"])
  accuracies.append(eval_results["accuracy"])
  
  if eval_results["global_step"] == ITERATIONS:
    max_accuracy = max(accuracies)
    step = steps[accuracies.index(max_accuracy)]
    print('\n--- Experiment result: ', end='')
    print('Accuracy %.4f at %d steps' % (max_accuracy, step), flush=True)
    return False
    
  return True

# Neural network train and test
def train_test(description):
  print(description, flush=True)
  
  # Dataset
  wgs.load()
  global NUM_CLASSES
  NUM_CLASSES = len(wgs.class_to_label)
  train_data    = wgs.train_data        
  train_labels  = wgs.train_labels 
  val_data      = wgs.val_data   
  val_labels    = wgs.val_labels 
  test_data     = wgs.test_data         
  test_labels   = wgs.test_labels 
  
  # Clean model directory
  shutil.rmtree(MODEL_DIR)
  os.mkdir(MODEL_DIR)
  
  # Configure GPU memory usage
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  # Estimator
  classifier = tf.estimator.Estimator(
    model_fn=rnn_model_fn, 
    model_dir=MODEL_DIR, 
    config=tf.contrib.learn.RunConfig(tf_random_seed=43, session_config=config))
  
  # Train the model using a validation set
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"data": train_data},
    y=train_labels,
    batch_size=BATCH_SIZE,
    num_epochs=None, # Continue until training steps are finished
    shuffle=True
    )
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"data": val_data},
    y=val_labels,
    batch_size=BATCH_SIZE,
    num_epochs=1, 
    shuffle=False
    )
  experiment = tf.contrib.learn.Experiment(
    estimator=classifier,
    train_input_fn=train_input_fn,
    eval_input_fn=eval_input_fn,
    train_steps=ITERATIONS,
    eval_steps=None, # evaluate runs until input is exhausted
    eval_delay_secs=120, 
    train_steps_per_iteration=100
    )
  experiment.continuous_train_and_eval(
    continuous_eval_predicate_fn=continuous_eval_predicate_fn)  
  
  # Test the model and print results
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"data": test_data},
    y=test_labels,
    batch_size=BATCH_SIZE,
    num_epochs=1,
    shuffle=False)
  test_results = classifier.evaluate(input_fn=test_input_fn)
  print("\nModel evaluation with test dataset:")
  print(test_results)
  
# Main function
def main(unused_argv):
  train_test(
    '\n\n--- Experiment with 128 hidden units')
  
if __name__ == "__main__":
  tf.app.run()
