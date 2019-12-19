# baseball_predictor
A baseball game outcome predictor with various ML classifier options.

# Baseball Predictor

A project intended to predict the outcome of baseball games through machine learning. Classifiers are trained on data
pulled from [Retrosheet](retrosheet.org). Current classifiers are decision trees, random forests, multi-layer perceptron
neural networks, and support vector machines. Classifiers are taken from [SciKitLearn](https://scikit-learn.org/stable/). All classifiers return similar results.

DISCLAIMER: This project is designed to learn about machine learning and data
analysis and should not be expected to return meaningful or accurate predictions. There is a surprising amount of noise
in large baseball datasets and it is difficult to determine which features are actually valuable for prediction.

## Getting Started

Feel free to pull this project and add to/play with it if you feel inclined.

### Prerequisites

Packages you'll need (I recommend setting up a virtual environment for your project):

```
sklearn
pandas
seaborn
matplotlib
os
```

### Installing

You can set up a virtual environment in the terminal with the command:

```
virtualenv [environment_name]
source [environment_name]/bin/activate
```

All of the packages that are not available natively in python are downloadable using [pip](https://pip.pypa.io/en/stable/).
After setting up your virtual environment, install all the project requirements into the virtual environment using pip:

```
pip install [package_name]
```

## Running

To run a classifier, navigate to the models directory in the terminal:

```
cd models
```

Then select the classifier model that you would like to run (e.g. decision_trees.py) and type:

```
python [classifier_file_name.py]
```

Configurations for each classifier can be edited within their respective files.

## Authors

* **Abel Romer** - *Initial work* - [ahblay](https://github.com/ahblay)


