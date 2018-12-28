// Supported species

export var SPECIES = [
  {name: 'Lactobacillus rhamnosus'			, taxid:	47715},
	{name: 'Campylobacter jejuni'			, taxid:	197},
	{name: 'Microbacterium sp.'			, taxid:	51671},
	{name: 'Streptococcus sp.'			, taxid:	1306},
	{name: 'Streptomyces sp.'			, taxid:	1931},
	{name: 'Enterobacter cloacae'			, taxid:	550},
	{name: 'Staphylococcus epidermidis'			, taxid:	1282},
	{name: 'Pseudomonas syringae'			, taxid:	317},
	{name: 'Mycobacterium tuberculosis'			, taxid:	1773},
	{name: 'Salmonella enterica'			, taxid:	28901},
	{name: 'Vibrio cholerae'			, taxid:	666},
	{name: 'Escherichia coli'			, taxid:	562},
	{name: 'Klebsiella pneumoniae'			, taxid:	573},
	{name: 'Mycobacterium abscessus'			, taxid:	36809},
	{name: 'Acinetobacter baumannii'			, taxid:	470},
	{name: 'Acinetobacter sp.'			, taxid:	472},
	{name: 'Brucella abortus'			, taxid:	235},
	{name: 'Bacillus cereus'			, taxid:	1396},
	{name: 'Enterococcus faecium'			, taxid:	1352},
	{name: 'Enterococcus faecalis'			, taxid:	1351},
	{name: 'Pseudomonas aeruginosa'			, taxid:	287},
	{name: 'Serratia marcescens'			, taxid:	615},
	{name: 'Burkholderia pseudomallei'			, taxid:	28450},
	{name: 'Neisseria gonorrhoeae'			, taxid:	485},
	{name: 'Pseudomonas sp.'			, taxid:	306},
	{name: 'Staphylococcus aureus'			, taxid:	1280},
	{name: 'Streptococcus agalactiae'			, taxid:	1311},
	{name: 'Rhizobium sp.'			, taxid:	391},
	{name: 'Sphingomonas sp.'			, taxid:	28214},
	{name: 'Lachnospiraceae bacterium'			, taxid:	1898203},
	{name: 'Clostridiales bacterium'			, taxid:	1898207},
	{name: 'Yersinia pestis'			, taxid:	632},
	{name: 'Streptococcus pyogenes'			, taxid:	1314},
	{name: 'Burkholderia sp.'			, taxid:	36773},
	{name: 'Mesorhizobium sp.'			, taxid:	1871066},
	{name: 'Proteobacteria bacterium'			, taxid:	1977087},
	{name: 'Verrucomicrobia bacterium'			, taxid:	2026799},
	{name: 'Firmicutes bacterium'			, taxid:	1879010},
	{name: 'Mycobacterium sp.'			, taxid:	1785},
	{name: 'Prevotella sp.'			, taxid:	59823},
	{name: 'Xanthomonas oryzae'			, taxid:	347},
	{name: 'Bacillus subtilis'			, taxid:	1423},
	{name: 'Lactobacillus plantarum'			, taxid:	1590},
	{name: 'Helicobacter pylori'			, taxid:	210},
	{name: 'Bordetella pertussis'			, taxid:	520},
	{name: 'Vibrio parahaemolyticus'			, taxid:	670},
	{name: 'Bacillus thuringiensis'			, taxid:	1428},
	{name: 'Stenotrophomonas maltophilia'			, taxid:	40324},
	{name: 'Oenococcus oeni'			, taxid:	1247},
	{name: 'Gammaproteobacteria bacterium'			, taxid:	1913989},
	{name: 'Listeria monocytogenes'			, taxid:	1639},
	{name: 'Staphylococcus sp.'			, taxid:	29387},
	{name: 'Shigella flexneri'			, taxid:	623},
	{name: 'Bacillus anthracis'			, taxid:	1392},
	{name: 'Lactococcus lactis'			, taxid:	1358},
	{name: 'Clostridium botulinum'			, taxid:	1491},
	{name: 'Clostridioides difficile'			, taxid:	1496},
	{name: 'Clostridium sp.'			, taxid:	1506},
	{name: 'Bacteroides fragilis'			, taxid:	817},
	{name: 'Legionella pneumophila'			, taxid:	446},
	{name: 'Rhodococcus sp.'			, taxid:	1831},
	{name: 'Paenibacillus sp.'			, taxid:	58172},
	{name: 'Cronobacter sakazakii'			, taxid:	28141},
	{name: 'Vibrio sp.'			, taxid:	678},
	{name: 'Pseudomonas stutzeri'			, taxid:	316},
	{name: 'Acinetobacter pittii'			, taxid:	48296},
	{name: 'Acidobacteria bacterium'			, taxid:	1978231},
	{name: 'Shigella sonnei'			, taxid:	624},
	{name: 'Bacteroidetes bacterium'			, taxid:	1898104},
	{name: 'Streptococcus pneumoniae'			, taxid:	1313},
	{name: 'Streptococcus equi'			, taxid:	1336},
	{name: 'Francisella tularensis'			, taxid:	263},
	{name: 'Corynebacterium diphtheriae'			, taxid:	1717},
	{name: 'Mycobacterium avium'			, taxid:	1764},
	{name: 'Neisseria meningitidis'			, taxid:	487},
	{name: 'Bacteroidales bacterium'			, taxid:	2030927},
	{name: 'Actinobacteria bacterium'			, taxid:	1883427},
	{name: 'Vibrio vulnificus'			, taxid:	672},
	{name: 'Klebsiella oxytoca'			, taxid:	571},
	{name: 'Yersinia enterocolitica'			, taxid:	630},
	{name: 'Porphyromonadaceae bacterium'			, taxid:	2049046},
	{name: 'Streptococcus suis'			, taxid:	1307},
	{name: 'Streptococcus mutans'			, taxid:	1309},
	{name: 'Klebsiella aerogenes'			, taxid:	548},
	{name: 'Campylobacter coli'			, taxid:	195},
	{name: 'Klebsiella quasipneumoniae'			, taxid:	1463165},
	{name: 'Burkholderia cenocepacia'			, taxid:	95486},
	{name: 'Pasteurella multocida'			, taxid:	747},
	{name: 'Corynebacterium sp.'			, taxid:	1720},
	{name: 'Deltaproteobacteria bacterium'			, taxid:	2026735},
	{name: 'Euryarchaeota archaeon'			, taxid:	2026739},
	{name: 'Chloroflexi bacterium'			, taxid:	2026724},
	{name: 'Enterobacter hormaechei'			, taxid:	158836},
	{name: 'Bacillus pseudomycoides'			, taxid:	64104},
	{name: 'Haemophilus influenzae'			, taxid:	727},
	{name: 'Ruminococcaceae bacterium'			, taxid:	1898205},
	{name: 'Leptospira interrogans'			, taxid:	173},
	{name: 'Burkholderia ubonensis'			, taxid:	101571},
	{name: 'Campylobacter concisus'			, taxid:	199},
	{name: 'Flavobacteriaceae bacterium'			, taxid:	1871037},
	{name: 'Alphaproteobacteria bacterium'			, taxid:	1913988},
	{name: 'Flavobacteriales bacterium'			, taxid:	2021391},
	{name: 'Bacillales bacterium'			, taxid:	1904864},
	{name: 'Dehalococcoidia bacterium'			, taxid:	2026734},
	{name: 'Prochlorococcus sp.'			, taxid:	1220},
	{name: 'Staphylococcus haemolyticus'			, taxid:	1283},
	{name: 'Staphylococcus argenteus'			, taxid:	985002},
	{name: 'Elusimicrobia bacterium'			, taxid:	2030800},
	{name: 'Rhodospirillaceae bacterium'			, taxid:	1898112},
	{name: 'Bacillus toyonensis'			, taxid:	155322},
	{name: 'Bacillus wiedmannii'			, taxid:	1890302},
];

export var CLI = '/wgs_classifier_cli.zip'

export var DOWNLOAD_INSTRUCTIONS = {__html: `
  <div class="post">
    <h2>Command line interface</h2>
    <p>&nbsp;</p>
    <p>A command line interface is available for download at the following <a href="` + CLI + `">link</a>. Once the file is in the local disk, it is necessary to install all the libraries and prepare the environment. First,  create a Python virtual environment and install the <a href="https://www.tensorflow.org/install/install_linux">TensorFlow</a> library</p>

  <pre class="console">
  sudo apt-get install python3-pip python3-dev python-virtualenv
  virtualenv --system-site-packages -p python3 ~/tensorflow
  source ~/tensorflow/bin/activate
  easy_install -U pip
  pip3 install --upgrade tensorflow==1.6.0</pre>

    <p>Now, install the remaining libraries</p>
      
  <pre class="console">
  pip3 install --upgrade numpy scipy scikit-learn matplotlib </pre>
  
    <p>Unzip the CLI file and execute the classifier. The supported file format is the Nucleotide FASTA fsa_nt.gz</a> </p>
      
  <pre class="console">
  mkdir wgs_classifier
  mv wgs_classifier_cli.zip  wgs_classifier
  cd wgs_classifier
  unzip wgs_classifier_cli.zip
  python3 wgs_classifier.py path_to_file/file.fsa_nt.gz</pre>
  
    <p>Results include the species scientific name and the score generated by the recurrent neural network. Softmax results from the classification layer are also included. </p>
    
    <h2>Classification system update</h2>
    <p>&nbsp;</p>
    <p>To update the recurrent neural network, download a CSV file from the NCBI <a href="https://www.ncbi.nlm.nih.gov/Traces/wgs/?page=2&view=wgs&search=BCT">site</a>. The link filters the results to include WGS projects for Bacteria only. Save the CSV file in <var>wgs_classifier/wgs_bct_recordset.csv</var>. Now, run the download script</p>
    
  <pre class="console">
  python3 wgs_download.py</pre>
    
    <p>Rerun the command if any network error appears in the console. The script will skip all FASTA files already stored in the local disk. Once the script returns no error, we are ready to retrain the model. Should you have a collection of at least 100 FASTA whole genome sequences per species, you can add new species to the model. Save all the *.fsa_nt.gz sequences in the directory <var>wgs_classifier/datasets/bacteria/taxid</var>, where taxid is a valid NCBI taxonomy <a href="https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi">identifier</a></p>
    
  <pre class="console">
  python3 wgs_rnn.py</pre>
    
    <p>To update the recurrent model, the <var>wgs_rnn.py</var> script cleans the model directory <var>rnn_model</var> and run the training operation for 3600 steps. Every 100 steps, TensorFlow prints evaluation results. Thus, we can observe the evolution of loss and accuracy values. After the training process finishes, the model directory will contain the updated recurrent model for subsequent classifications. </p> 
  </div>
    
    
`}

