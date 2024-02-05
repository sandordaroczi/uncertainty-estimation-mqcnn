# Uncertainty Estimation with MQ-CNN

Welcome to the repository for our collaborative project, dedicated to implementing various uncertainty estimation models tailored for a retail use-case. This project has a specific focus on the MQ-CNN (Quantile Convolutional Neural Network) model. This algorithm extends the capabilities of existing models such as LightGBM, Bootstrapping, NGBoost, PGBM, LSF, Conformalized Regression, and TFT.

## Notebooks and Kaggle Datasets
Explore the notebooks folder to find Python notebooks used to train and evaluate the models on Kaggle datasets. The Kaggle datasets utilized for evaluation include:

1. Blue Book For Bulldozers (bulldozer)
2. Rossmann Store Sales (rossmann)
3. Corporaci´on Favorita Grocery Sales Forecasting (favorita)

## Getting Started

### 1. Installation

Ensure you have [poetry](https://python-poetry.org/docs/) installed on your local machine. After cloning the project, create a virtual environment and install packages from the pyproject.toml file:

````commandline
poetry update
poetry install
poetry build
pip install uncertainty_estimation_models_mqcnn-0.1.0-py3-none-any.whl
````

Make sure to separately install the various dependencies listed below.

### Dependencies

For running the notebooks with CPU, install the MXNet package via `pip install mxnet`. For GPU usage, install the mxnet-cu113 package via `pip install mxnet-cu113`.

The notebooks have been tested to work on Windows 10 and MacOS Ventura.

## Build and Test

Navigate to the desired directory for the git repo and run:

````commandline
git clone https://github.com/daroczisandor/uncertainty-estimation-mqcnn.git
cd uncertainty_estimation_mqcnn
````

Obtain the required datasets from [Google Drive](https://drive.google.com/drive/folders/1WV-z19PntL_PhDEwZbvPhI7WOZdxfYrO?usp=sharing) and store them in a folder called "datasets" at the top level of this repo.

## Contribute

Contributions are encouraged! Feel free to create issues or submit pull requests to enhance and extend the project. 🚀

