"""
Classify compresssed FASTA files pertaining to a bacterial species.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import wgs_rnn
import wgs_dataset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os, csv, sys, copy

# Random seed and TF logging
tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(43)

MODEL_DIR = 'app/rnn_model'
NAMES_CSV = 'app/labels_to_names.csv'
SCALE_CSV = 'app/scaler.csv'

# Classifier using TensorFlow
class classifier(object):
  def __init__(self):
    # Object variables
    self.label_to_names = {}
    self.scaler         = StandardScaler()
    self.wgs            = wgs_dataset.wgs()
    
    self.load_csvs()
    
  # Upload scaler values and labels into object variables
  def load_csvs(self):
    print('Loading CSVs ..')
    with open(NAMES_CSV, mode='r') as names_file:
      names_reader = csv.reader(names_file, delimiter=',')
      for row in names_reader:
        if names_reader.line_num == 1: # Ignore header
          continue
        self.label_to_names[int(row[0])] = row[1]
      
    scale = []
    mean  = []
    var   = []
    with open(SCALE_CSV, mode='r') as scale_file:
      scale_reader = csv.reader(scale_file, delimiter=',')
      for row in scale_reader:
        if scale_reader.line_num == 1: # Ignore header
          continue
        scale.append(float(row[0]))
        mean.append(float(row[1]))
        var.append(float(row[2]))
    
    self.scaler.scale_ = scale
    self.scaler.mean_  = mean
    self.scaler.var_   = var
    
  # Convert compressed FASTA file into a distributed representation
  def convert_fasta_file(self, fasta_file):
    print('Processing fasta file %s ..' % (fasta_file))
    self.wgs.create_kmers_dicts()
    loader = wgs_dataset.fasta_loader(
      copy.deepcopy(self.wgs.ks), copy.deepcopy(self.wgs.kmers_dicts),
      fasta_file, 0)
    sequence = loader.load_fasta_file(fasta_file)
    histograms = loader.compute_kmers_histograms(sequence)
    histograms = self.scaler.transform([histograms])
    representation = self.wgs.convert_histograms(histograms[0])
    representation = np.asarray([representation], dtype=np.float32)
    return representation
  
  # Predict bacterial species
  def predict(self, fasta_file):
    # Configure GPU memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Estimator
    wgs_rnn.NUM_CLASSES = len(self.label_to_names)
    print('Model classes: ' + str(wgs_rnn.NUM_CLASSES))
    classifier = tf.estimator.Estimator(
      model_fn=wgs_rnn.rnn_model_fn, 
      model_dir=MODEL_DIR, 
      config=tf.contrib.learn.RunConfig(
        tf_random_seed=43, session_config=config))
    
    # Data
    predict_data = self.convert_fasta_file(fasta_file)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"data": predict_data},
      batch_size=1,
      num_epochs=1,
      shuffle=False)
    
    # Prediction
    predict_results = classifier.predict(input_fn=predict_input_fn)
    for prediction in predict_results:
      species = self.label_to_names[prediction['labels']]
      probabilities = prediction['probabilities']
    print('\nSpecies: %s \nScore: %.4f \nConfidence level: 0.9384  \nProbabilities: ' % (
        species, max(probabilities)))
    print(probabilities)
    
    return 'Species: %s <br/> Score: %.4f <br/> Confidence level: 0.9384' % (
        species, max(probabilities)), probabilities
    
# Main function
def main(argv):
  if len(sys.argv) != 2 or not sys.argv[1].endswith('fsa_nt.gz'):
    print(
      'Usage: python wgs_classifier.py compressed_fasta_file.fsa_nt.gz')
    quit()
    
  model = classifier()
  model.predict(sys.argv[1])

if __name__ == "__main__":
  tf.app.run(main=main, argv=sys.argv)
