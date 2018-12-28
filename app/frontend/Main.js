import React from 'react';
require("./index.css");

function Main(props) {
  return (
    <div>
      {props.children}  
    </div>
  );
}

export default Main;