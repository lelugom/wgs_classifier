"""
Test suite for WGS python modules

$ python3 -m pytest -svv # To print console output and increase verbosity
"""

import os
import copy
import pytest
import numpy as np

# Download
import wgs_download
ncbi = wgs_download.ncbi()

def test_download_tax():
  ncbi.download_tax()
  assert os.path.exists(wgs_download.TAX_FILE)
  
def test_load_taxonomy():
  ncbi.load_taxonomy()
  assert ncbi.tax_parent['53'] == '224463'
  assert ncbi.tax_rank['53'] == 'genus'
  assert ncbi.name_tax['Bacteria'] == '2'
  
def test_load_prj_urls():
  if os.path.exists(wgs_download.URL_CSV):
    expected = [
      'ftp://ftp.ncbi.nlm.nih.gov/sra/wgs_aux/AT/BJ/ATBJ01/ATBJ01.1.fsa_nt.gz',
      'ATBJ01.1.fsa_nt.gz']
    ncbi.load_prj_urls()
    assert expected == ncbi.prj_urls['ATBJ01']
    
def test_get_fasta_url():
  expected = (        
    'ftp://ftp.ncbi.nlm.nih.gov/sra/wgs_aux/AA/KB/AAKB02/AAKB02.1.fsa_nt.gz', 
    'AAKB02.1.fsa_nt.gz')
  assert expected == ncbi.get_fasta_url('AAKB02')
  
def test_download_fasta():
  ncbi.download_fasta()
  assert os.path.exists(wgs_download.OUT_DIR)

# Dataset
import wgs_dataset
wgs = wgs_dataset.wgs()

def test_load():
  # load function performs clean up. So, use dataset object instead of 
  # global wgs object
  dataset = wgs_dataset.wgs()
  dataset.concatenated_representation = True
  dataset.load()
  assert len(dataset.train_data[0]) == 1344
  
def test_load_subfolders():
  wgs.ks = [3, 4, 5]
  wgs.create_kmers_dicts()
  wgs.load_subfolders()
  assert len(wgs.labels) > 20000
  
  wgs.data = []
  wgs.labels = []
  wgs.seqs_per_class = 2000
  wgs.min_seqs_per_class = 2000
  wgs.load_subfolders()
  assert len(wgs.labels) == 17991
  
  wgs.data = []
  wgs.labels = []
  wgs.seqs_per_class = 100
  wgs.min_seqs_per_class = 1000
  wgs.load_subfolders()
  assert len(wgs.labels) == 1400
  
  label = wgs.class_to_label.get('36809', None)
  assert label != None
  
def test_get_kmers_count():
  count = wgs_dataset.get_kmers_count(5)
  assert count == 1024
  
def test_create_kmers_dicts():
  wgs.ks = [3, 4, 5]
  wgs.create_kmers_dicts()
  kmer_dict = wgs.kmers_dicts[3]
  
  assert 'TTT' in kmer_dict
  assert 'AGC' in kmer_dict
  assert 'AAA' in kmer_dict
  
  kmer_dict = wgs.kmers_dicts[4]
  assert 'CGAC' in kmer_dict
  
  kmer_dict = wgs.kmers_dicts[5]
  assert 'AGTTT' in kmer_dict

def test_convert_histograms():
  wgs.ks = [3, 4, 5]
  histograms = np.ones(1344)
  representation = wgs.convert_histograms(histograms)
  assert representation[0][64]   == 0
  assert representation[1][256]  == 0
  assert representation[0][1023] == 0
  assert representation[1][1023] == 0
  assert representation.shape == (3, 1024)
  
  wgs.ks = [3, 4]
  representation = wgs.convert_histograms(histograms)
  assert representation[0][64] == 0
  assert representation.shape == (2, 256)
  
  wgs.ks = [4, 5]
  representation = wgs.convert_histograms(histograms)
  assert representation[0][256] == 0
  assert representation.shape == (2, 1024)
  
  wgs.ks = [5]
  representation = wgs.convert_histograms(histograms)
  assert representation[0][1023] == 1
  assert representation.shape == (1, 1024)
  
def test_print_stats():
  wgs.print_stats([1,5,3,6,9,8,4,5,2,1,5,3])

# FASTA loader
def test_load_fasta_file():
  wgs.ks = [3, 4, 5]
  wgs.create_kmers_dicts()
  fasta_file = 'datasets/bacteria/197/AANJ01.1.fsa_nt.gz'
  loader = wgs_dataset.fasta_loader(copy.deepcopy(wgs.ks),
          copy.deepcopy(wgs.kmers_dicts), fasta_file, 0)
  sequence = loader.load_fasta_file(fasta_file)
  
  assert sequence.startswith(
    'TATAGTATTTAATCCATAAATTAATAAATCTCTATCAGTATTATCTTCCTTATCTTCGTTATTTTCATTT')
  assert sequence.endswith(
    'AGCATTATCTGGTATAATATATGCCTGAGGCTGTT')

def test_compute_kmers_histograms():
  wgs.ks = [3, 4, 5]
  wgs.create_kmers_dicts()
  loader = wgs_dataset.fasta_loader(copy.deepcopy(wgs.ks),
          copy.deepcopy(wgs.kmers_dicts), '', 0)
  sequence = 'TAGACTGTCAAAA'
  kmer_dict = loader.kmers_dicts[3]
  
  histograms = loader.compute_kmers_histograms(sequence)
  assert np.sum(histograms) == 30
  assert np.size(histograms) == 1344
  # Indexes only work in the first histogram, which represents 3-mers
  assert histograms[kmer_dict['TAG']] == 1
  assert histograms[kmer_dict['AAA']] == 2

# crossval
def test_crossval():
  crossval = wgs_dataset.crossval()
  crossval.load()
  crossval.next_fold()
  
  train_data    = crossval.train_data        
  train_labels  = crossval.train_labels 
  test_data     = crossval.test_data         
  test_labels   = crossval.test_labels  
  
  assert np.round(len(train_data) / len(test_data)) == 4.0
  
  crossval.next_fold()
  assert not np.array_equal(test_data, crossval.test_data)
  assert not np.array_equal(test_labels, crossval.test_labels)
  
# RNN
import wgs_rnn

def test_continuous_eval_predicate_fn():
  wgs_rnn.steps = [1, 200, 500, 700]
  wgs_rnn.accuracies = [0.8, 0.99, 0.5, 0.4]
  
  eval_results = {}
  eval_results['global_step'] = wgs_rnn.ITERATIONS
  eval_results['accuracy'] = 0.3
  
  assert not wgs_rnn.continuous_eval_predicate_fn(eval_results) 
