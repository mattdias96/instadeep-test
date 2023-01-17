<div align="center">

# Protein Classifier - InstaDeep - MLE Coding Test

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
First, clone the repository
```bash
# clone project
git clone https://github.com/mattdias96/instadeep-test.git

cd instadeep-test
 ```
To replicate the docker image, run
```python
docker build -t myimage .
 ```
 Alternatively, you can manually install all requirements:
 ```python
pip install -r requirements.txt
 ```
Make sure you have [docker](https://www.docker.com/) and [make](https://gnuwin32.sourceforge.net/packages/make.htm) installed in your computer.

## Usage
For quick usage, the make file allows 3 different commands: train, predict and evaluate.
The user should choose the associated parameters: train_dir (file path for dataset),
gpus (1 for use GPU or 0 for CPU), model_weights_file_path (file path for .pth file with weights
of trained model).
```python
# To train the model on the full dataset
make train train_dir="..." gpus=1
# To predict the model on a given dataset
make predict train_dir="..." model_weights_file_path="..."
# To evaluate the model on a given dataset
make evaluate train_dir="..." model_weights_file_path="..."
```

To have the ability to change parameters, you can call the function directly:
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