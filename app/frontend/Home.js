import React from 'react';
import NavBar from './NavBar';
import { Link } from 'react-router';
require("./index.css");

// Jumbotron Welcome functional component
function Welcome(props) {  
  return(
    <div className="container mainPanel">
      <div className="Jumbotron col-md-4 col-md-offset-4 centerBox" id="home_jumbotron">
        <h1 className="text-info">Welcome</h1>
        <p className="lead" >
          Bacteria classifier based on whole genome sequence information. Genomic data from <a href="https://www.ncbi.nlm.nih.gov/genbank/">GenBank</a>
        </p>
        <hr/>
        <p>Press the button to start</p>
        <p className="lead">
          <Link to='/classify'>
            <button className="btn btn-success btn-lg" role="button">Enter</button>
          </Link>
        </p>
      </div>
    </div>
  );
}

// Home page Component
class Home extends React.Component {
  static contextTypes = {
    router: React.PropTypes.object,
  };
  
  render () {
    return (
      <div>
        <NavBar/>
        <Welcome/>
      </div>
    );
  };
}

export default Home;