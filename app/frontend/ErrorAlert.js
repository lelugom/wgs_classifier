import React from 'react'
require("./index.css")

export default function ErrorAlert (props) {
  return (
    <div className="col-md-4 col-md-offset-4 alertBox">
      <div className="alert alert-danger alert-dismissible fade in" role="alert">
        <button type="button" className="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true"></span></button>
        <strong>Error!</strong> {props.error}
      </div>
    </div>
  )
}