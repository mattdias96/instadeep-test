<div align="center">

# InstaDeep - Machine Learning Engineer Coding Test

![Folded protein](https://singularityhub.com/wp-content/uploads/2021/07/AI-generated-protein-structure.jpg)


<!--
Conference
-->
</div>

## Description
This project is an implementation of a protein classifier inspired by the ProtCNN model as described
in the paper ["Using Deep Learning to Annotate the Protein Universe"](https://www.biorxiv.org/content/10.1101/626507v2.full).
The model is trained on the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) and its performance
is measured on multi-class accuracy.
This github repo was develped for the InstaDeep Machine Learning Engineer Coding Test.

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
To replicate the docker image, run
```python
docker build -t myimage .
 ```
Make sure you have docker and make installed in your computer.

## Usage
For quick usage, the make file allows 4 different commands:
```python
# To train the model on the full dataset
make train
# To predict the model on a given dataset
make predict
# To evaluate the model on a given dataset
make predict
```
For advanced usage, the user can call the function directly:
```python
```
Remember to explain arguments


## Example
To run a smaller version of the dataset and check train and dev set accuracies, run
```python
make train_small
```

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
Talk about lint here?

## Contact
- Author: Matheus P. Dias
- Email: matheuspbfdias@gmail.com
- Website: [mattdias96.github.io/website/](mattdias96.github.io/website/)
- Twitter: [@themattdias](http://twitter.com/themattdias)
- LinkedIn: [Matheus Dias](https://www.linkedin.com/in/matheus-p-dias/)
- Kaggle: [Matheus Dias](https://www.kaggle.com/matheusdias1996)