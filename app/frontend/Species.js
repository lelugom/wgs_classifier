import React from 'react';
import NavBar from './NavBar';
var constants = require('./constants.js');
require("./index.css");

// Table to display supported species
function Species(props) {
  return (
    <div>
    <NavBar/>
    <div className="container mainPane">
    <div className="col-md-10 col-md-offset-1 leftBox">
      <div className="panel panel-primary ">
        <div className="panel-heading text-center"><strong>Bacteria</strong></div>
        <div className="table-responsive">
        <table className="table">
          <thead>
            <tr><th>Scientific Name</th><th>Tax ID</th></tr>
          </thead>
          <tbody>
            {constants.SPECIES.map( (species) =>
              <tr key={species.name}>
                <td>{species.name}</td><td>{species.taxid}</td>
              </tr>
            )}
          </tbody>
        </table>
        </div>
      </div>
    </div>
    </div>
    </div>
  );
}

export default Species;