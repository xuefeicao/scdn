# Estimating high dimensional ODE models from convoluted observations with an application to fMRI
scdn is a Python-based package implementing sparse causal dynamic network analysis for convolution model, particular for Functional magnetic resonance imaging (fMRI) in our paper. It aims to provide a sparse dynamic network estimation not only for fMRI data but for other possible data that can be represented by convolution model. The introduction and explanation of parameters and ODE models can be found in [(1)]. For more details of convolution model, see [(2)]


## Getting Started

The examples provided in the repo have been tested in Mac os and Linux environment. This package supports both Python 2.7 and Python 3.6. 

These instructions will get you a copy of the project up running on your local machine for development and testing purposes. 

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
from scdn.scdn_analysis import scdn_multi_sub
help(scdn_multi_sub)
```


### Examples
```
The examples subfolder includes two examples.
The first is a simulation generated from our data and another is from DCM.
```

## Running the tests

The test is going to be added in the future.

## Built With

* Python 2.7

## Compatibility
* Python 2.7
* Python 3.6 


## Authors

* **Xuefei Cao** - *Maintainer* - (https://github.com/xuefeicao)
* **Xi Luo** (http://bigcomplexdata.com/)
* **Björn Sandstede** (http://www.dam.brown.edu/people/sandsted/)


## License

This project is licensed under the MIT License - see the LICENSE file for details

[(1)]:http://www.fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf
[(2)]:https://pdfs.semanticscholar.org/2127/7ee7b67970782bef59c9d657b144237bacbd.pdf
