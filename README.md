# TensorFlow to TMVA Adapter
The TensorFlow to TMVA Adapter aims to replicate the functionality of ROOT's TMVA Python API using TensorFlow. This gives us the opportunity to make use of TensorFlow's flexible architecture to run the statistical analysis typically reserved for TMVA, with minimal changes to existing code.
## Installation
Installation is best handled with a Linux (Ubuntu 14.04.3 LTS 64-bit) Virtualenv.
### Virtualenv
Follow the instructions to build and install Virtualenv from source [here](https://virtualenv.pypa.io/en/latest/installation.html).
### TensorFlow to TMVA Adapter
Clone this repository and `cd` into it:
```bash
$ git clone https://github.com/AidanGG/TensorFlow-to-TMVA-Adapter.git
$ cd TensorFlow-to-TMVA-Adapter
```
Create a Virtualenv:
```bash
$ virtualenv env
```
and activate with:
```bash
$ source env/bin/activate
```
### TensorFlow
Install prerequisites and TensorFlow:
```bash
(env)$ sudo apt-get install python-dev
(env)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
```
### ROOT
Install binaries from [here](https://root.cern.ch/downloading-root) and set the necessary environment variables.
### rootpy and root_numpy
Install rootpy and root_numpy:
```bash
(env)$ pip install rootpy root_numpy
```
## Relevant links
* [TensorFlow](http://www.tensorflow.org/)
* [ROOT](https://root.cern.ch/)
* [rootpy](http://www.rootpy.org/)
* [root_numpy](https://rootpy.github.io/root_numpy/index.html)
* [TMVA](http://tmva.sourceforge.net/)
