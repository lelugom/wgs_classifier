import React from 'react';
import { Link } from 'react-router';
require("./index.css");

class NavBar extends React.Component {
  render() {
    return (
      <div>
        <nav className="navbar navbar-inverse navbar-fixed-top">
          <div className="container-fluid">
            <div className="navbar-header">
              <button type="button" className="navbar-toggle collapsed" data-toggle="collapse" data-target="#index-collapse" aria-expanded="false">
                <span className="sr-only">Toggle navigation</span>
                <span className="icon-bar"></span>
                <span className="icon-bar"></span>
                <span className="icon-bar"></span>
              </button>
              <Link className="navbar-brand" to="/"><span className="glyphicon glyphicon-home"></span> Home</Link>
            </div>
            <div className="collapse navbar-collapse" id="index-collapse">
              <ul className="nav navbar-nav navbar-left">
                <li><Link to="/classify"><span className="glyphicon glyphicon-blackboard"></span> Classifier</Link></li>
                <li><Link to="/species"><span className="glyphicon glyphicon-list"></span> Species</Link></li>
                <li><Link to="/download"><span className="glyphicon glyphicon-download-alt"></span> Download</Link></li>
              </ul>
            </div>
          </div>
        </nav>
      </div>
    );
  }
}

export default NavBar;
