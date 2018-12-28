"""
Backend for the React.js app with the WGS bacterial classifier

[1] https://testdriven.io/part-one-getting-started
[2] http://blog.teamtreehouse.com/uploading-files-ajax
[3] http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
"""
import os
import wgs_classifier

from flask import Flask, jsonify, send_from_directory, request, redirect
from werkzeug.utils import secure_filename

STATIC_DIR = 'app/'
UPLOAD_DIR = 'tmp/'

# Instantiate the app
app = Flask(__name__)

# Serve HTML file 
@app.route('/', methods=['GET'])
def main():
  return send_from_directory(STATIC_DIR, 'index.html')

# Serve JavaScript app
@app.route('/index.js', methods=['GET'])
def send_js():
  return send_from_directory(STATIC_DIR, 'index.js')
  
# Serve WGS classifier CLI tool
@app.route('/wgs_classifier_cli.zip', methods=['GET'])
def send_cli():
  return send_from_directory(STATIC_DIR, 'wgs_classifier_cli.zip')

# API functions
@app.route('/classify', methods=['POST'])
def classify():
  answer = {'error': ''}
  if 'file' not in request.files:
    answer['error'] = 'No file in request'
    return jsonify(answer)
  
  file = request.files['file']
  if file.filename == '':
    answer['error'] = 'Error while uploading file'
    return jsonify(answer)
  
  if not file.filename.endswith('fsa_nt.gz'):
    answer['error'] = file.filename + ': unsupported file type'
    return jsonify(answer)
    
  fasta_file = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
  answer['status'] = file.filename + ' uploaded'
  file.save(fasta_file)
  
  try: 
    model = wgs_classifier.classifier()
    answer['result'], answer['probabilities'] = model.predict(fasta_file)
    answer['probabilities'] = ['%.4f' % (p) for p in answer['probabilities']]
  except Exception as e:
    answer['error'] = 'Failed to process file. ' + str(e)
    print('Error: ' + str(e))
    return jsonify(answer)
  
  print(answer)
  os.remove(fasta_file)
  return jsonify(answer)
    
# Run the app
if __name__ == '__main__':
  app.run(host = '0.0.0.0', threaded=False, processes=4) 
