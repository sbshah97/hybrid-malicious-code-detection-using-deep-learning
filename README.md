# Hybrid Malicious Code Detection using Deep Learning

## About

This is a Keras implementation of **A Hybrid Malicious Code Detection Method based on Deep
Learning**. Basically it is an hybrid model consisting of an autoencoder and a Deep Belief Network. 

Details about the dataset are explained at the [KDDCup website](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

## Python Dependencies

* Numpy
* Keras
* Pandas
* Scikit Learn
* Tensorflow

## Environment Setup

* It is preferable if you use Python Ananconda Environment. You can download it from [here](https://www.anaconda.com/download/#linux)

* Create a new conda environment using the following command:
```
conda create -n hybrid-code python=3.5
```

* Activate the environment by running the following code:
```
source activate hybrid-code
```

* To install the required libraries, run the following commands:
```
conda install numpy pandas sklearn
```

## Training

The basic usage is `python train.py`.

## Contributors
* The Project is created and maintained by [Salman Shah](https://github.com/salman-bhai).