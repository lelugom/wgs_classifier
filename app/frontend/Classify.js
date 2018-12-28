import React from 'react';
import NavBar from './NavBar';
import ErrorAlert from './ErrorAlert';
import axios from 'axios';
var constants = require('./constants.js');
require("./index.css");

// Classify form
function ClassForm(props) {
  return(
    <div>
      <form className="col-md-6 col-md-offset-3 centerBox"
        enctype="multipart/form-data" onSubmit={props.handleSubmit}>
        <h1 className="text-info">Bacteria classification</h1>
        <hr/>
        <div className="form-group">
          <label htmlFor="fileInput">Nucleotide FASTA file (.fsa_nt.gz):</label>
          <input type="file" className="form-control-file" id="fileInput"
              onChange={props.handleFile}/>
        </div>
        <button type="submit" className="btn btn-primary btn-lg">
          Process sequence
        </button>
      </form>
    </div>  
  );
}

// Progress bar
function ProgressBar(props) {
  return (
    <div className='progress'>
      <div className='progress-bar' role='progressbar'
         aria-valuenow= {props.progress}
         aria-valuemin='0'
         aria-valuemax='100'
         style={{width: props.progress + '%'}}>
        <span className='sr-only'>Upload file</span>
      </div>
     </div>
  );
}
              
// Classify container
class Classify extends React.Component {
  constructor(props){
    super(props);

    this.state = {
      fastaFile     : '',
      fastaName     : '',
      error         : '',
      status        : '',
      result        : '',
      progress      : '',
      probabilities : []
    }
    
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleFile   = this.handleFile.bind(this);
    this.cleanState  = this.cleanState.bind(this);
  }
  
  // event handling
  handleSubmit(event){
    event.preventDefault();
    if(!this.state.fastaFile){
      this.setState({error: 'Please upload a file'})
      return;
    }
    if(!this.state.fastaName.endsWith('fsa_nt.gz')){
      this.setState((prevState, props) => ({
        error: prevState.fastaName + ': unsupported file type'
      }));
      return;
    }
    
    this.setState({status: 'Uploading file ..'});
    const data = new FormData();
    data.append('file', this.state.fastaFile);
    data.append('name', this.state.fastaName);
    var config = {
      onUploadProgress: (progressEvent) => {
        var percentCompleted = Math.round( 
          (progressEvent.loaded * 100) / progressEvent.total);
        this.setState({progress: String(percentCompleted)});
        if(percentCompleted == 100){
          this.setState({progress: ''});
          this.setState((prevState, props) => ({
            status: prevState.status + 
              '\nProcessing sequence .. this can take a few minutes'
          }));
        }
      }
    };
    
    axios.post('/classify', data, config)
      .then((response) => {
        console.log(response.data); 
        this.setState((prevState, props) => ({
            status: prevState.status + '\nDone'
          }));
        this.setState({result: {__html: response.data.result}});
        this.setState({probabilities: response.data.probabilities});
      })
      .catch((error) => {
        this.setState({error: 'Error processing file'});
        this.cleanState();
        console.log(error); 
      })
  }
  
  handleFile(event){
    event.preventDefault();       
    this.setState({error: ''});
    this.cleanState();
    this.setState({fastaName: event.target.files[0].name});
    this.setState({fastaFile: event.target.files[0]});
  }
  
  cleanState(){
    this.setState({status:    ''});
    this.setState({result:    ''});
    this.setState({progress:  ''});
    this.setState({probabilities : []})
  }
  
  // Display container
  render() {
    return (
      <div>
        <NavBar/>
        { this.state.error && <ErrorAlert error = {this.state.error}/> }
        <div className="container mainPanel">
        <ClassForm
          handleSubmit = {this.handleSubmit}
          handleFile   = {this.handleFile}/>
        { ! this.state.error && this.state.status &&  
          <div className="col-md-6 col-md-offset-3 leftBox">
            <hr/>
            <pre className="console">{this.state.status}</pre>
            { this.state.progress &&
              <ProgressBar progress = {this.state.progress} />
            }
            <hr/>
            { this.state.result &&
              <p className="lead" 
                dangerouslySetInnerHTML= {this.state.result} />
            }
          </div>
        }
        </div>
      </div>
    );
  }
}

export default Classify;