<div align="center">

# InstaDeep - Machine Learning Engineer Coding Test

![A screenshot of the project](https://media.nature.com/lw1024/magazine-assets/d41586-020-03348-4/d41586-020-03348-4_18633156.jpg)


<!--
Conference
-->
</div>

## Description
This project is an implementation of a protein classifier inspired by the ProtCNN model as described
in the paper ["Using Deep Learning to Annotate the Protein Universe"](https://www.biorxiv.org/content/10.1101/626507v2.full).
The model is trained on the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) and its performance
is measured on multi-class accuracy.
This github repo was develped for the InstaDeep Machine Learning Engineer Coding Test

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Testing](#testing)
- [Contact](#contact)

## Installation
First, install dependencies
```bash
# clone project
git clone https://github.com/mattdias96/instadeep-test.git

# install project - adjust this for when I have the docker working
cd instadeep-test
pip install -e .
pip install -r requirements.txt
 ```
Also install make

## Usage
The make file allows 4 different commands Next, navigate to any file and run it.


## Examples


## Testing
To run the tests for this project, you can use the following command:
```python
make test
```
We also include a test coverage report which you can generate by running the command
```python
coverage run -m pytest
coverage report
```

## Contact
- Author: Matheus P. Dias
- Email: matheuspbfdias@gmail.com
- Website: [mattdias96.github.io/website/](mattdias96.github.io/website/)
- Twitter: [@themattdias](http://twitter.com/themattdias)
- LinkedIn: [Matheus Dias](https://www.linkedin.com/in/matheus-p-dias/)
- Kaggle: [Matheus Dias](https://www.kaggle.com/matheusdias1996)