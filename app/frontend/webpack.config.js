const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: [
    __dirname + '/index.js'
  ],
  module: {
    loaders: [
      // Import css files into Components
      { test: /\.css$/, loader: "style-loader!css-loader" },
      // ES6 JavaScript and JSX
      {
        test: /\.js$/, exclude: /node_modules/,
        include: __dirname + '/', loader: "babel-loader"
      }
    ]
  },
  output: {
    path: __dirname + '/../',
    filename: "index.js"
  },
  watch: true,
  devServer: {
    historyApiFallback: true, // Fallback URL
  },
  plugins: [
    // Insert JS bundle into HTML page
    new HtmlWebpackPlugin({
      "files": { "css": [__dirname + '/index.css']},
      template: __dirname + '/index.html',
      filename: __dirname + '/../index.html',
      inject: 'body'
    }),
    // Use minfied production build
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: JSON.stringify('production')
      }
    }),
    new webpack.optimize.UglifyJsPlugin({
      beautify: false,
      mangle: {
        screw_ie8: true,
        keep_fnames: true
      },
      compress: {
        screw_ie8: true
      },
      comments: false
    })
  ]
};
