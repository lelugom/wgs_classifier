import React from 'react';
import NavBar from './NavBar';
var constants = require('./constants.js');
require("./index.css");

// Download instructions
function Download(props) {
  return (
    <div>
    <NavBar/>
    <div className="container mainPane">
      <div className="col-md-10 col-md-offset-1" 
        dangerouslySetInnerHTML={constants.DOWNLOAD_INSTRUCTIONS} /> 
    </div>
    </div>
  );
}

export default Download;