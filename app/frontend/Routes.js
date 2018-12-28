import React from 'react';
import { Router, Route, IndexRoute, hashHistory } from 'react-router'
import Main from './Main';
import Home from './Home';
import Classify from './Classify';
import Download from './Download';
import Species from './Species';
require("./index.css");

function Routes(props) {
  return (
    <Router history={hashHistory}>
      <Route path='/' component={Main}>
        <IndexRoute component={Home}/>
        <Route path="classify" component={Classify}/>
        <Route path="download" component={Download}/>
        <Route path="species"  component={Species}/>
      </Route>
    </Router>
  );
}

export default Routes;