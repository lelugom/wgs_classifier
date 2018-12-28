"""
Download FASTA files and taxonomy files from NCBI FTP site [1]. Recordseet file
in CSV format should be downloaded from [2] and stored as wgs_bct_recordset.csv.
Consider only files with extension .fsa_nt.gz. Example list for bacteroides fragilis in [3]

[1] ftp://ftp.ncbi.nlm.nih.gov/
[2] https://www.ncbi.nlm.nih.gov/Traces/wgs/?page=2&view=wgs&search=BCT
[3] https://www.ncbi.nlm.nih.gov/genome/genomes/414

"""

import os, re, sys, gzip, pprint, tarfile, urllib.request

# Constants
OUT_DIR   = "datasets/bacteria"
TAX_FILE  = "datasets/taxdump.tar.gz"
WGS_CSV   = "wgs_bct_recordset.csv"
TAX_URL   = "ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
PRJ_URL   = 'https://www.ncbi.nlm.nih.gov/Traces/wgs'
URL_CSV   = '/'.join([OUT_DIR, 'wgs_prj_urls.csv'])

THRESHOLD = 100
MAX_COUNT = 2000

VERIFY_OUT_DIR = True

# Hold tax datasets and downloaded data
class ncbi(object):
  # Create object
  def __init__(self):
    # Object variables
    self.tax_rank       = {}
    self.tax_parent     = {}
    self.name_tax       = {}
    self.species_count  = {}
    
    self.prj_urls       = {}
    
    self.regex_species  = re.compile('^(\w+\s+\w+\.*).*')
  
  # Download both taxonomic and FASTA files from NCBI to local disk
  def download(self):
    self.download_tax()
    self.load_taxonomy()
    self.download_fasta()
    
  # Download taxonomic dump files from NCBI if they are not in the 
  # local disk
  def download_tax(self):
    if os.path.exists(TAX_FILE):
      print("File %s already exist" % (TAX_FILE))
      return
   
    try:
      print("Retrieving: %s" % TAX_URL)
      urllib.request.urlretrieve(TAX_URL, TAX_FILE)
    except:
      raise Exception("Unable to download taxonomy files")
      
  # Load taxonomy dictionaries from NCBI dump
  def load_taxonomy(self):
    tax_sep   = "\t|\t"
    tax_term  = "\t|\n"
    
    print("Loading tax id info ..")
    with tarfile.open(TAX_FILE, mode='r:*') as dump:
      try: 
        nodes = dump.extractfile('nodes.dmp')
        names = dump.extractfile('names.dmp')
        
        # Load tax id information
        for line in nodes:
          line = line.decode('utf-8').replace(tax_term, '')
          fields = line.split(tax_sep)
          self.tax_parent[fields[0]] = fields[1]
          self.tax_rank[fields[0]] = fields[2]
          
        # Load names
        for line in names:
          line = line.decode('utf-8').replace(tax_term, '')
          fields = line.split(tax_sep)
          # Unique name to avoid repeated entries
          if fields[3] == 'scientific name': 
            self.name_tax[fields[1]] = fields[0]
            
      except:
        raise Exception("Error while reading file from %s " % TAX_FILE)
        
  # Download WGS Fasta files to local disk
  def download_fasta(self):
    if os.path.exists(OUT_DIR) and VERIFY_OUT_DIR:
      print("FASTA files already downloaded in %s" % OUT_DIR)
      return
      
    if not os.path.exists(WGS_CSV):
      raise Exception("Unable to find %s" % WGS_CSV)
    
    # Recordset file
    print("Loading WGS data from %s .." % WGS_CSV)
    with open(WGS_CSV, mode='r') as recordset:
      records = recordset.readlines()
      recordset.close()
      
    # Get header indices
    header = records[0].strip().split(',')
    try: 
      prefix_idx = header.index("prefix_s")
      project_idx = header.index("project_s")
      div_idx = header.index("div_s")
      organism_idx = header.index("organism_an")
      records.pop(0)
    except:
      raise Exception("Error while readig recordset file ")
      
    self.compute_species_count(
      records, prefix_idx, div_idx, project_idx, organism_idx)
    self.get_fasta_files(records, prefix_idx, organism_idx)
  
  # Get URLs and download FASTA files from NCBI FTP site
  def get_fasta_files(self, records, prefix_idx, organism_idx):
    url_count = 0
    no_url_count = 0
    taxid_count = {}
    
    self.load_prj_urls()
    
    # CSV file to store URLs
    url_file_dir = os.path.dirname(URL_CSV)
    if not os.path.exists(url_file_dir):
      os.makedirs(url_file_dir)
    url_file = open(URL_CSV, 'a')
    
    for record in records:
      record_data = record.strip().split(',')
      organism = record_data[organism_idx]
      project = record_data[prefix_idx]
      
      # NZ_* projects has no NCBI page
      if project.startswith('NZ_'):
        continue
        
      match = self.regex_species.match(organism)
      if match:
        species = match.group(1)
        
        if self.species_count.get(species, None):
          # Download only MAX_COUNT number of fasta files
          taxid = self.name_tax[species]
          taxid_count[taxid] = taxid_count.get(taxid, 0) 
          if taxid_count[taxid] >= MAX_COUNT:
            continue
          
          url, fasta_file = self.get_fasta_url(project)
          if not url:
            no_url_count += 1
            continue
            
          filename = '/'.join([OUT_DIR, taxid, fasta_file])
          self.download_file(url, filename)
          taxid_count[taxid] += 1
          if not self.prj_urls.get(project, None):
            url_file.write(','.join([project, url, fasta_file]) + '\n')
          url_count += 1
          print('\tDownloaded urls %d. Invalid urls %d' % \
            (url_count, no_url_count))
      
    url_file.close()
    print("Downloaded files: Invalid url count %d" % no_url_count)
    pprint.pprint(taxid_count)
  
  # Load WGS project URLs and filenames from the CSV file if it exists
  def load_prj_urls(self):
    if not os.path.exists(URL_CSV):
      return
      
    with open(URL_CSV, mode='r') as url_file:
      for line in url_file:
        line = line.strip()
        fields = line.split(',')
        self.prj_urls[fields[0]] = [fields[1], fields[2]]
  
  # Download Project NCBI page and extract fasta file url and name
  def get_fasta_url(self, project):
    url = ''
    fasta_file = ''
      
    # Look for data from object dict if it exists
    prj_url = self.prj_urls.get(project, None)
    if prj_url:
      print('\tURL for %s already loaded, skipping ..' % project)
      url = prj_url[0]
      fasta_file = prj_url[1]
      return url, fasta_file
      
    try:
      html_path = '/'.join([PRJ_URL, project])
      print('\tRetrieving: %s' % html_path)
      html_file = urllib.request.urlopen(html_path)
      html_lines = html_file.readlines()
      html_file.close()
    except:
      raise Exception("Error while downloading %s " % html_path)
    
    fasta_regex = re.compile('.+em.FASTA.+?\"(.+fsa_nt\.gz)\">(.+?)<.+')
    for line in html_lines:
      match = fasta_regex.match(line.strip().decode('utf-8'))
      if match:
        url = match.group(1)
        fasta_file = match.group(2)
    
    return url, fasta_file
  
  # Connect to FTP server and download fasta file to local disk
  def download_file(self, url, filename):
    download_dir = os.path.dirname(filename)
    if not os.path.exists(download_dir):
      os.makedirs(download_dir)
      
    try:
      if not os.path.exists(filename):
        print('\tRetrieving: %s' % url)
        urllib.request.urlretrieve(url, filename)
      else:
        print('\tFile %s already exists, skipping ..' % filename)
    except Exception as exc:
      print('There was a problem downloading %s.\n'\
        'Check input arguments and try again.' % url)
      
  # Explore records and calculate the number of rows per species. Filter species
  # revising tax rank for species and THRESHOLD value
  def compute_species_count(
    self, records, prefix_idx, div_idx, project_idx, organism_idx):
    count = {}
    
    for record in records:
      record_data = record.strip().split(',')
      organism = record_data[organism_idx]
      
      # NZ_* projects has no NCBI page
      match = self.regex_species.match(organism)
      if match and record_data[div_idx] == 'BCT' \
        and record_data[project_idx] == 'WGS'    \
        and not record_data[prefix_idx].startswith('NZ_'):
          
        species = match.group(1)
        name = self.name_tax.get(species, None)
        if name and self.tax_rank[name] == 'species':
          count[species] = count.get(species, 0) + 1
        else:
          print("organism:%s species:%s has tax %s" % (
            organism, species, name))
     
    for key in count:
      if count[key] > THRESHOLD:
        self.species_count[key] = count[key]
    
    pprint.pprint(self.species_count)
    self.print_species_count()  
    
  # Print species count in LaTex format
  def print_species_count(self):
    print('\n\\hline')
    print('\t\\textbf{Species} & \\textbf{Tax ID} & \\textbf{Projects} \\\\')
    print('\\hline')
    for key in self.species_count:
      print('\t%s  &  %s  &  %s  \\\\  \\hline' % (
        key, self.name_tax[key], self.species_count[key])) 
    print('\n')
      
if __name__ == "__main__":
  VERIFY_OUT_DIR = False
  ncbi().download()   
