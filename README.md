# This repo is under development 
# Estimating high dimensional ODE models from convoluted observations with an application to fMRI
scdn is a Python-based package implementing sparse causal dynamic network analysis for convoluted observations, particular for Functional magnetic resonance imaging (fMRI) in our research. It aims to provide a sparse dynamic network estimation not only for fMRI data but for other possible convoluted observations. The introduciton and explaination of parameters and ODE models can be found in [(1)]. 


## Getting Started
This package supports both python 2.7 and python 3.6.

Example provided in the repo has been tested in mac os and Linux environment. 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

This package is also published in pypi.  For a quick installation, try

```
pip install scdn
```

### Prerequisites

What things you need to install the software and how to install them

```
See setup.py for details of packages requirements. 
```

### Installing from GitHub


Download the packages by using git clone https://github.com/xuefeicao/scdn.git

```
python setup.py install
```

If you experience problems related to installing the dependency Matplotlib on OSX, please see https://matplotlib.org/faq/osx_framework.html 

### Intro to our package
After installing our package locally, try to import scdn in your python environment and learn about package's function. 
```

```


### Examples
```
The examples subfolder includes a basic analysis of our sample data.
```

## Running the tests

The test is going to be added in the future.

## Built With

* Python 2.7

## Compatibility
* python 2.7
* python 3.6 

## Authors

* **Xuefei Cao** - *Maintainer* - (https://github.com/xuefeicao)
* **Xi Luo** (http://bigcomplexdata.com/)
* **Björn Sandstede** (http://www.dam.brown.edu/people/sandsted/)


## License

This project is licensed under the MIT License - see the LICENSE file for details

[(1)]:http://www.fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf