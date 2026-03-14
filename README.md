# Automated Essay Grading using BERT

## Overview

This project implements an Automated Essay Scoring (AES) system using a pre-trained BERT model and PyTorch. The model learns to evaluate essays and predict a numerical score based on their content.

The system uses a regression approach where essay text is processed through BERT and a linear regression layer predicts the final essay score.

## Features

* Uses a pre-trained BERT model (bert-base-uncased)
* Essay tokenization using HuggingFace Transformers
* PyTorch based training pipeline
* Mean Squared Error loss for regression
* Evaluation using MSE and R² score
* Visualization of training and validation loss

## Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

## Dataset

The project expects a CSV dataset with the following columns:

| Column  | Description |
| ------- | ----------- |
| Essay   | Essay text  |
| Overall | Essay score |

Example:

```
Essay,Overall
"This essay discusses global poverty...",6.5
"The chart shows population growth...",5.0
```

## Project Workflow

1. Load essay dataset
2. Preprocess text using BERT tokenizer
3. Create a PyTorch Dataset and DataLoader
4. Fine-tune BERT for essay score regression
5. Evaluate model performance
6. Predict score for new essays

## Model Architecture

BERT Encoder
↓
CLS Token Representation
↓
Linear Regression Layer
↓
Predicted Essay Score

## Installation

Clone the repository:

```
git clone https://github.com/AbdullahAli2005/automated-essay-grading.git
cd automated-essay-grading
```

Create a virtual environment:

```
python -m venv .venv
```

Activate environment:

Windows:

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers
```

## Running the Project

Place your dataset file as:

```
essays.csv
```

Run the training script:

```
python main.py
```

## Example Output

```
Epoch 1, Train Loss: 0.52, Val Loss: 0.61, MSE: 0.61, R2: 0.42
Epoch 2, Train Loss: 0.31, Val Loss: 0.40, MSE: 0.40, R2: 0.58
Epoch 3, Train Loss: 0.21, Val Loss: 0.33, MSE: 0.33, R2: 0.65
```

Example prediction:

```
Predicted Score: 6.87
```

## Visualization

The project also visualizes training and validation loss across epochs using Seaborn.


## Author

Abdullah Ali

Flutter Developer | AI Enthusiast | MERN Stack Learner

## License

This project is open-source and available under the MIT License.
