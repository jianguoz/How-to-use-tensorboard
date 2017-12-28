# How-to-use-tensorboard
### Jianguo Zhang, December 22, 2017

This is the implementation code for the "How to Use Tensorboard" live session by Siraj Raval on Youtube

## Overview

This is the code for [this](https://www.youtube.com/watch?v=fBVEXKp4DIc) and [this](https://www.youtube.com/watch?v=3bownM3L5zM) video on Youtube by Siraj Raval. We're going to classify handwritten characters using a convolutional neural network. Then we'll visualize the results in tensorboard, including a demo of the new embedding visualizer. 

## Dependencies

* os
* tensorflow 
* sys
* urllib

Install dependencies with [pip](https://packaging.python.org/installing/). 

## Usage


Run the following in terminal to train the code. 

```
python simple_mnist.py
```



Visualize it with this command in terminal after training. 

```
tensorboard --logdir=./logs/nn_logs 
```

Run the following in terminal to train the code. 

```
python advanced_mnist.py
```

Visualize it with this command in terminal after training. 

```
tensorboard --logdir='/tmp/mnist_tutorial/'
```
