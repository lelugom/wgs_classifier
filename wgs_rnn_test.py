"""
Load a small test dataset and generate model metrics for the bacterial 
classification system

"""

import wgs_dataset, wgs_rnn

import numpy as np
import tensorflow as tf
from sklearn import metrics
import os, csv, sys, copy

TEST_CSV = 'datasets/test_dataset.csv'
MODEL_DIR = 'app/rnn_model'

# Load test dataset from file
def load_dataset():
  data = []
  labels = []
  wgs = wgs_dataset.wgs()

  with open(TEST_CSV, mode='r') as test_dataset:
      reader = csv.reader(test_dataset, delimiter=',')
      for row in reader:
        if reader.line_num == 1: # Ignore header
          continue
        labels.append(int(row[0]))
        kmers = [float(k) for k in row[1:]]
        data.append(wgs.convert_histograms(kmers))

  data   = np.asarray(data  , dtype=np.float32)
  labels = np.asarray(labels, dtype=np.int32)
  return labels, data

# Load train model and compute metrics
def test_model():
  labels, data = load_dataset()

  # Estimator
  wgs_rnn.NUM_CLASSES = max(labels) + 1
  classifier = tf.estimator.Estimator(
    model_fn=wgs_rnn.rnn_model_fn, 
    model_dir=MODEL_DIR, 
    config=tf.contrib.learn.RunConfig(
      tf_random_seed=43, session_config=tf.ConfigProto())
    )
    
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"data": data},
    y=labels,
    batch_size=wgs_rnn.BATCH_SIZE,
    num_epochs=1,
    shuffle=False)
  
  # Compute model metrics with Scikit learn
  predict_labels = []
  predict_results = classifier.predict(input_fn=test_input_fn)
  for prediction in predict_results:
    predict_labels.append(prediction['labels'])
  predict_labels = np.asarray(predict_labels)
  
  accuracy  = metrics.accuracy_score(labels,predict_labels)
  precision = metrics.precision_score(labels,predict_labels, average='weighted')
  recall    = metrics.recall_score(labels,predict_labels, average='weighted')
  fscore    = metrics.f1_score(labels,predict_labels, average='weighted')
  
  print("\n\nTest results\n")
  print('\taccuracy  :  %.5f   ' % (accuracy )) 
  print('\tprecision :  %.5f   ' % (precision)) 
  print('\trecall    :  %.5f   ' % (recall   )) 
  print('\tfscore    :  %.5f   ' % (fscore   ))

if __name__ == "__main__":
  test_model()