"""
Process FASTA files for automatic labelling of sequences. Load training,
validation, and test datasets.

[1] http://scikit-learn.org/stable/modules/preprocessing.html
[2] https://pymotw.com/2/multiprocessing/communication.html
[3] https://stackoverflow.com/questions/10415028/
how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce
[4] https://docs.python.org/3/library/multiprocessing.html
[5]  http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

"""

import wgs_download

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

import os, re, gc, csv, sys, copy, gzip, glob, math, time, pprint, random, \
  tarfile, multiprocessing, matplotlib


# Increase font size in plots
matplotlib.rcParams.update({'font.size': 11})

# Random seed
random.seed(43)
np.random.seed(43)

# Constants
MAX_THREADS  = 16
MIN_SEQ_LEN  = 200000
MAX_SEQ_LEN  = 30000000

ALPHABET = ['A', 'G', 'C', 'T']
REP_CSV  = '/'.join([wgs_download.OUT_DIR, 'wgs_representations_345.csv'])

NAMES_CSV = 'labels_to_names.csv'
SCALE_CSV = 'scaler.csv'
 
# Calculate the number of k-mers for a given k
def get_kmers_count(k):
  return int(math.pow(len(ALPHABET), k))
    
# Hold datset for WGS
class wgs(object):
  def __init__(self, ks=[3, 4, 5]):
    # Object variables
    self.ks              = ks
    self.class_to_label  = {}
    self.seq_lens        = []
    self.kmers_dicts     = {}
    self.representations = {}
    self.ncbi            = wgs_download.ncbi()
    
    self.data            = []
    self.labels          = []
    self.train_data      = []
    self.train_labels    = []
    self.val_data        = []
    self.val_labels      = []
    self.test_data       = []
    self.test_labels     = []
    
    self.scaler          = None
    self.process_pool    = []
    
    self.seqs_per_class     = 2000
    self.min_seqs_per_class = 100
    
    self.concatenated_representation = False
    
    self.zero_center = False
    
  # Load dataset from local disk
  def load(self):
    self.ncbi.download()
    self.create_kmers_dicts()
    self.load_subfolders()
    self.split_data()
    self.preprocess_data()
    self.dump_csvs()
    self.convert_fields()
    
    # Clean up
    del self.kmers_dicts, self.seq_lens, self.representations, \
      self.data, self.labels, self.scaler, self.ncbi
    gc.collect()
    
  # Calculate weights using the inverted frequency model. Class Weights are 
  # calculated over the training set labels. Weight arrays in numpy format.
  def compute_weights(self):
    weights = {}
    total_labels = len(self.train_labels)
    
    for i in range(0, len(self.class_to_label)):
      class_labels = len(self.train_labels[np.where(self.train_labels == i)])
      weights[i] = total_labels / class_labels
      
    print("Updated weights: ")
    pprint.pprint(weights)
    
    self.train_weights = np.ones(self.train_labels.shape, dtype=np.float32)
    for i in range(len(self.train_labels)):
      self.train_weights[i] = weights[self.train_labels[i]]
      
    self.test_weights = np.ones(self.test_labels.shape, dtype=np.float32)
    for i in range(len(self.test_labels)):
      self.test_weights[i] = weights[self.test_labels[i]]
      
    if self.val_data != []:
      self.val_weights = np.ones(self.val_labels.shape, dtype=np.float32)
      for i in range(len(self.val_labels)):
        self.val_weights[i] = weights[self.val_labels[i]]
    else:
      self.val_weights = np.asarray(self.val_weights, dtype=np.float32)
      
  # Split data into training 60%, validation 20%, and testing 20%
  def split_data(self):
    data, self.test_data, labels, self.test_labels = \
      train_test_split(self.data, self.labels, test_size=0.2)
    self.train_data, self.val_data, self.train_labels, self.val_labels = \
      train_test_split(data, labels, test_size=0.25)
  
  # Normalize kmers hitograms and save parameters in scaler. Use training data 
  # for mean and stdev computation. Then, normalize all sets with the training
  # means and stdevs
  def preprocess_data(self):
    self.scaler = StandardScaler().fit(self.train_data)
    if self.zero_center:
      train_mean = np.mean(self.train_data)
      self.train_data -= train_mean
      self.test_data  -= train_mean
      if self.val_data != []:
        self.validate_data -= train_mean
    else:
      self.train_data = self.scaler.transform(self.train_data)
      self.test_data  = self.scaler.transform(self.test_data)
      if self.val_data != []:
        self.val_data = self.scaler.transform(self.val_data)
   
  # Store label to species names into a CSV file. Store means and stdevs from
  # scaler into a CSV file. Both files are to be used by the classifier
  # names format: label,species_name
  # scaler format: scale,mean,var
  def dump_csvs(self):
    print('Saving CSVs ..')
    with open(NAMES_CSV, mode='w') as names_file:
      names_writer = csv.writer(names_file, delimiter=',')
      # header
      names_writer.writerow(['label', 'species_name'])
      # data
      for species, label in self.class_to_label.items():
        for name, taxid in self.ncbi.name_tax.items():
          if species == taxid:
            names_writer.writerow([str(label), name])
  
    with open(SCALE_CSV, mode='w') as scale_file:
      scale_writer = csv.writer(scale_file, delimiter=',')
      # header
      scale_writer.writerow(['scale', 'mean', 'var'])
      # data
      for i in range(0, len(self.scaler.scale_)):
        scale_writer.writerow([
          '%.8f' % (self.scaler.scale_[i]),
          '%.8f' % (self.scaler.mean_[i]),
          '%.8f' % (self.scaler.var_[i]),
          ])
  
  # Transform concatenated histograms in the sets to matrix form if 
  # concatenated_representation object variable is set to False. 
  # Otherwise, flatten the representation by reshaping data variables.
  # Convert lists to float np.arrays for TensorFlow further processing. 
  # Delete not needed object fields
  def convert_fields(self):
    if self.concatenated_representation:
      length  = len(self.train_data[0])
      self.train_data = self.train_data.reshape([-1, 1, length])
      self.test_data  = self.test_data .reshape([-1, 1, length])
      if self.val_data:
        self.val_data   = self.val_data  .reshape([-1, 1, length])
    else:
      sets = [self.train_data, self.test_data, self.val_data]
      for s in range(len(sets)):
        converted_set = []
        for i in range(len(sets[s])):
          converted_set.append(self.convert_histograms(sets[s][i]))
        if s == 0:
          self.train_data = converted_set
        elif s == 1:
          self.test_data = converted_set
        else:
          self.val_data = converted_set
        
    self.train_data    = np.asarray(self.train_data   , dtype=np.float32)
    self.train_labels  = np.asarray(self.train_labels , dtype=np.int32)
    self.test_data     = np.asarray(self.test_data    , dtype=np.float32)
    self.test_labels   = np.asarray(self.test_labels  , dtype=np.int32)
    self.val_data      = np.asarray(self.val_data     , dtype=np.float32)
    self.val_labels    = np.asarray(self.val_labels   , dtype=np.int32)
   
    print("Sequences: training %d, validation %d, testing %d" % 
      (len(self.train_data), len(self.val_data), len(self.test_data)))
    print('Entry shape: ', end='')
    print(self.train_data[0].shape)
    print('Number of species: ', end='')
    print(len(self.class_to_label))
    print("Class to labels:")
    pprint.pprint(self.class_to_label)
    
    # Flush stdout to update log
    sys.stdout.flush() 
    
  # Load genomic data from subfolders. Each subfolder is named after
  # the bacteria taxid
  def load_subfolders(self):
    subfolders = glob.glob('/'.join([wgs_download.OUT_DIR, '*', '']))
    
    self.load_representations()
    
    for subfolder in subfolders:
      taxid = subfolder.split('/')[-2]
      self.class_to_label[taxid] = len(self.class_to_label)
      self.load_subfolder(subfolder, float(self.class_to_label[taxid]))
    
    self.print_stats(self.seq_lens)
  
  # Load compressed FASTA files from subfolder 
  def load_subfolder(self, subfolder, label):
    fasta_files = glob.glob('/'.join([subfolder, '*fsa_nt.gz']))
    
    # Ignore species with less samples than self.min_seqs_per_class.
    # Update self.class_to_label dictionary accordingly
    if len(fasta_files) < self.min_seqs_per_class:
      print('\tIgnoring FASTA files from %s. Subfolder only has %d samples' % (
        subfolder, len(fasta_files)))
      taxid = None
      for key, value in self.class_to_label.items():
        if value == label:
          taxid = key
      del self.class_to_label[taxid] 
      return
    
    # Trim fasta files to load only self.seqs_per_class sequences 
    while len(fasta_files) > self.seqs_per_class:
      idx = random.randint(0, len(fasta_files) - 1)
      fasta_files.pop(idx)
    
    # CSV file to store distributed representations. The directory was already
    # created in the wgs_download module
    rep_file = open(REP_CSV, 'a')
    
    print('\tLoading and processing %d FASTA files from %s ..' % (
      len(fasta_files), subfolder))
    for fasta_file in fasta_files:
      representation = self.representations.get(fasta_file, None)
      if representation:
        # Ignore representation[1] and use label instead
        self.labels.append(label)
        self.seq_lens.append(representation[0])
        self.data.append(np.array(representation[2:], dtype=np.float32))
      else:
        print('\tadding process for %s ...' % fasta_file)
        loader = fasta_loader(copy.deepcopy(self.ks),
          copy.deepcopy(self.kmers_dicts), fasta_file, label + 0.0)
        self.process_pool.append(loader)
        if len(self.process_pool) == MAX_THREADS:
          self.clean_process_pool(rep_file)
    
    self.clean_process_pool(rep_file)
    rep_file.close()
    
  # Run all the processes and clean process pool. Write results to rep_file
  # Use a queue for storing process results cause they are not accessible
  # after join() returns. get() process in Queue blocks 
  def clean_process_pool(self, rep_file):
    print('\trunning processes ..')
    results = []
    queue = multiprocessing.Queue()
    
    for process in self.process_pool:
      process.queue = queue
      process.start()
    
    for process in self.process_pool:
      results.append(queue.get()) 
      
    for process in self.process_pool:
      process.join()
      
    for result in results:
      if result['seq_len']:
        self.seq_lens.append(result['seq_len'])
        self.data.append(result['representation'])
        self.labels.append(result['label'])
        if rep_file != None:
          rep_file.write(result['csv_row'])
        
    self.process_pool = []
    queue = None
    gc.collect()
  
  # Loas WGS distributed representations from the CSV file if it exists
  # format: fasta_file, seq_len, label, histograms
  def load_representations(self):
    if not os.path.exists(REP_CSV):
      return
      
    with open(REP_CSV,  mode='r') as rep_file:
      for line in rep_file:
        line = line.strip()
        fields = line.split(',')
        representation = fields[1:]
        representation = [float(r) for r in representation]
        self.representations[fields[0]] = representation
  
  # Create dictionaries for k-mers
  def create_kmers_dicts(self):
    for k in self.ks:
      kmers = {}
      
      for i in range(0, get_kmers_count(k)):
        kmer = ''
        mask = 3
        for j in range(0, k):
          kmer += ALPHABET[(i & mask) >> (2 * j)]
          mask = mask << 2
        kmers[kmer] = i
        
      self.kmers_dicts[k] = kmers
    
  # Convert the vector holding histograms into a matrix, one row per k-mer
  def convert_histograms(self, histograms):
    max_kmer_count = get_kmers_count(max(self.ks))
    representation = np.zeros((len(self.ks), max_kmer_count))
    
    # k=3 is the minimum considered
    hists_ptr = 0
    for k in range(3, min(self.ks)):
      hists_ptr += get_kmers_count(k)
    
    for i in range(0, len(self.ks)):
      kmers_count = int(math.pow(len(ALPHABET), self.ks[i]))
      for j in range(0, kmers_count):
        representation[i][j] = histograms[hists_ptr]
        hists_ptr += 1
    
    return representation
    
  # Print statistics for sequence lengths
  def print_stats(self, lens, plot=False):
    print("\nSequence lengths stats: ")
    iq_range = scipy.stats.iqr(lens)
    print("min=%d max=%d mean=%.3f median=%d std=%.3f var=%.3f, IQR=%d\n" % (
      np.amin(lens), np.amax(lens), np.mean(lens), np.median(lens), 
      np.std(lens), np.var(lens), iq_range))
    
    if not plot:
      return
    
    #  Freedman-Diaconis rule of thumb for bins count
    h = 2.0 * iq_range / np.cbrt(len(lens))
    bins = int((max(lens) - min(lens)) / h)
    
    # Express in Mbps
    lens = [l / 1000000 for l in lens]
    
    # Generate plots 
    #plt.subplot(212)
    plt.hist(lens, bins)
    plt.xlabel('Length (Mbps)')
    #plt.ylabel('Frequency')
    #plt.subplot(211)
    #plt.title('Sequence lengths')
    #plt.boxplot(lens, 0, 'g')
    #plt.ylabel('Length (Mbps)')
    plt.tight_layout()
    plt.show()
  
# FASTA file loader. Pass a copy of kmers_dicts, ks, and label to avoid 
# conflicts. threading.Thread is affected by Python's Global Interpreter Lock. 
# GIL does not allow threads to run concurrently. Instead, use
# multiprocessing.Process
class fasta_loader(multiprocessing.Process):
  def __init__(self, ks, kmers_dicts, fasta_file, label):
    multiprocessing.Process.__init__(self)
    # Object variables
    self.ks              = ks
    self.kmers_dicts     = kmers_dicts
    self.fasta_file      = fasta_file
    self.label           = label
    
    self.queue           = None
    self.representation  = ''
    self.csv_row         = ''
    
    self.seq_len         = 0
    self.seq_size        = 1
    self.min_seq_len     = MIN_SEQ_LEN
    self.max_seq_len     = MAX_SEQ_LEN
    
  def run(self):
    result = {}
    sequence = self.load_fasta_file(self.fasta_file)
    self.seq_len = len(sequence)
    
    # Ignore length outliers
    if self.seq_len < self.min_seq_len or self.seq_len > self.max_seq_len:
      print("\tIgnoring %s with %d bps" % (
        self.fasta_file, self.seq_len), flush=True)
      result['seq_len'] = 0
      self.queue.put(result)
      return
    
    # Trim sequence is self.seq_size is less than 1. 
    if self.seq_size < 1:
      fragment_length = int(math.floor(self.seq_size * len(sequence)))
      start = random.randint(0, len(sequence) - fragment_length - 1)
      sequence = sequence[start : start + fragment_length]
    
    self.representation = self.compute_kmers_histograms(sequence)
    str_representation = [str(r) for r in self.representation]
    self.csv_row = ','.join(
      [self.fasta_file, str(self.seq_len), str(self.label)] +
      str_representation) + '\n'
    
    result['label'] = self.label
    result['seq_len'] = self.seq_len 
    result['csv_row'] = self.csv_row 
    result['representation'] = self.representation
    self.queue.put(result)
    
  # Load nucleotide sequence from a compressed FASTA file. Use regex instead
  # of line by line processing, which is slower
  def load_fasta_file(self, file):
    sequence = ''
    
    with gzip.open(file, 'rt') as fasta_file:
      try:
        sequence = fasta_file.read()
        sequence = re.sub('\>.+?\n', '', sequence)
        sequence = sequence.replace('\n', '')
      except:
        raise("Error while reading fasta sequence from %s" % file)
        
    return sequence
    
  # Compute kmers distributed representations. Return a vector holding all the
  # concatenated histograms, starting from the histogram representing the 
  # first k in self.ks array
  def compute_kmers_histograms(self, sequence):
    histograms = np.array([], dtype=np.float32)
    k_histograms = {}
    for k in self.ks:
      k_histograms[k] = np.zeros(get_kmers_count(k), dtype=np.float32)
    
    for w in range(0, len(sequence) - min(self.ks) + 1):
      for k in self.ks:
        kmers_dict = self.kmers_dicts[k]
        histogram = k_histograms[k]
        substring = sequence[w : w + k]
        index = kmers_dict.get(substring, None)
        if index != None:
          histogram[index] += 1
        
    for k in sorted(self.ks):
      histograms = np.append(histograms, k_histograms[k])
    
    return histograms
  
  # Plot k_histograms dictionary for distributed sequence representation
  def plot_kmers(self, k_histograms, plot=False):
    if not plot:
      return
    
    for i in range(len(self.ks)):
      histogram = k_histograms[self.ks[i]]
      plt.subplot(311 + i)
      if i == 0:
        plt.title('k-mers representation')
      plt.bar(np.arange(1, len(histogram) + 1), histogram)
      plt.xlabel('%d-mers' % (self.ks[i]))
      plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Subclass of wgs dataset to perform k fold cross-validation
class crossval(wgs):
  def __init__(self, k=10):
    wgs.__init__(self)
    # Object variables
    self.k         = k
    self.splits    = None
    self.test_size = 0.2
    self.kfold   = ShuffleSplit(n_splits=self.k, test_size=self.test_size)
    
  # Override. Load dataset from disk
  def load(self):
    self.ncbi.download()
    self.create_kmers_dicts()
    self.load_subfolders()
    self.data   = np.asarray(self.data   , dtype=np.float32)
    self.labels = np.asarray(self.labels , dtype=np.int32)
    self.splits = self.kfold.split(X=self.data, y=self.labels)
  
  # Get next fold indices. Update train and test sets. Preprocess and convert 
  # them for neural network training and testing
  def next_fold(self):
    train_indices, test_indices = next(self.splits)
    self.train_data    = self.data[train_indices]
    self.train_labels  = self.labels[train_indices]
    self.test_data     = self.data[test_indices]
    self.test_labels   = self.labels[test_indices]
    
    self.preprocess_data()
    self.convert_fields()

# Subclass of wgs dataset to produce only a test set with ten samples 
# per species
class test(wgs):
  def __init__(self, seq_size=1):
    wgs.__init__(self)
    # Object variables
    self.seqs_per_class = 10
    
    self.k         = 10
    self.seq_size  = seq_size
    self.scaler    = StandardScaler()
    
  # Override. Load dataset from disk
  def load(self):
    self.ncbi.download()
    self.create_kmers_dicts()
    self.load_scaler()
  
  # Override. Load genomic data from subfolders. Each subfolder is named after
  # the bacteria taxid
  def load_subfolders(self):
    subfolders = glob.glob('/'.join([wgs_download.OUT_DIR, '*', '']))  
    for subfolder in subfolders:
      taxid = subfolder.split('/')[-2]
      self.class_to_label[taxid] = len(self.class_to_label)
      self.load_subfolder(subfolder, float(self.class_to_label[taxid]))
  
  # Override. Load compressed FASTA files from subfolder 
  def load_subfolder(self, subfolder, label):
    fasta_files = glob.glob('/'.join([subfolder, '*fsa_nt.gz']))
    
    # Ignore species with less samples than self.min_seqs_per_class.
    # Update self.class_to_label dictionary accordingly
    if len(fasta_files) < self.min_seqs_per_class:
      print('\tIgnoring FASTA files from %s. Subfolder only has %d samples' % (
        subfolder, len(fasta_files)))
      taxid = None
      for key, value in self.class_to_label.items():
        if value == label:
          taxid = key
      del self.class_to_label[taxid] 
      return
      
    # Trim fasta files to load only self.seqs_per_class sequences 
    while len(fasta_files) > self.seqs_per_class:
      idx = random.randint(0, len(fasta_files) - 1)
      fasta_files.pop(idx)
      
    print('\tLoading and processing %d FASTA files from %s ..' % (
      len(fasta_files), subfolder))
    for fasta_file in fasta_files:
      print('\tadding process for %s ...' % fasta_file)
      loader = fasta_loader(copy.deepcopy(self.ks),
          copy.deepcopy(self.kmers_dicts), fasta_file, label + 0.0)
      loader.seq_size = self.seq_size
      loader.min_seq_len = 0
      self.process_pool.append(loader)
      if len(self.process_pool) == MAX_THREADS:
          self.clean_process_pool(None)
    
    self.clean_process_pool(None)
      
  # Upload scaler values
  def load_scaler(self):
    print('Loading CSV ..')
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
  
  # Get next fold indices. Update test set. Preprocess and convert 
  # it for neural network testing
  def next_fold(self):
    self.seq_lens        = []
    self.data            = []
    self.labels          = []
    self.class_to_label  = {}
    
    self.load_subfolders()
    self.data   = np.asarray(self.data   , dtype=np.float32)
    self.labels = np.asarray(self.labels , dtype=np.int32)
    self.data   = self.scaler.transform(self.data)
    
    converted_set = []
    for i in range(len(self.data)):
      converted_set.append(self.convert_histograms(self.data[i]))
    self.data = np.asarray(converted_set, dtype=np.float32)
    
    print("Sequences: testing %d" % (len(self.data)))
    print('Entry shape: ', end='')
    print(self.data[0].shape)
    print('Data shape: ', end='')
    print(self.data.shape)
    print('Labels shape: ', end='')
    print(self.labels.shape, flush=True)

# Print k-mers dicts in C++ format
def print_kmers():
  dataset = wgs(ks=[3,4,5,6])
  dataset.create_kmers_dicts()
  for k in dataset.ks:
    kmers_dict = dataset.kmers_dicts[k]
    string = '{'
    for kmer in kmers_dict:
      string += "{\"%s\"," % (kmer)
      string += str(kmers_dict[kmer]) + "},"
    string = string[:len(string) - 1]
    print(string + '};')

if __name__ == "__main__":
  wgs().load()
  
