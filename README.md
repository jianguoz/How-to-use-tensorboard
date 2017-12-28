# how-to-use-tensorboard
This is the code for the "How to Use Tensorboard" live session by Siraj Raval on Youtube

## Overview

This is the code for [this](https://www.youtube.com/watch?v=fBVEXKp4DIc) video on Youtube by Siraj Raval. We're going to classify handwritten characters using a convolutional neural network. Then we'll visualize the results in tensorboard, including a demo of the new embedding visualizer. 

## Dependencies

* os
* tensorflow 
* sys
* urllib

Install dependencies with [pip](https://packaging.python.org/installing/). 

## Usage

Run `python simple_mnist.py` in terminal to train the code. 


Visualize it with this command in terminal after training. 


`tensorboard --logdir=./logs/nn_logs` 


Run `python advanced_mnist.py` in terminal to train the code. 


Visualize it with this command in terminal after training. 

`tensorboard --logdir='/tmp/mnist_tutorial/'` 
