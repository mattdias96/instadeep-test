<div align="center">

# InstaDeep - Machine Learning Engineer Coding Test

Add Insta Deep Picture


<!--
Conference
-->
</div>

## Description
This project is an implementation of a protein classifier inspired by the ProtCNN model as described
in the paper ["Using Deep Learning to Annotate the Protein Universe"](https://www.biorxiv.org/content/10.1101/626507v2.full).
The model is trained on the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) and its performance
is measured on multi-class accuracy.

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project
cd deep-learning-project-template
pip install -e .
pip install -r requirements.txt
 ```
 Next, navigate to any file and run it.
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)
python lit_classifier_main.py
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

