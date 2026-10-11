# LSTM and GRU-Based Next-Word Prediction

## Project Overview

This project implements and compares two recurrent neural network architectures, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), for word-level next-word prediction using WikiText-103 text data.

Given a sequence of five words, the model predicts the most likely next word and returns the top five candidate words with their estimated probabilities.

## Objectives

- Prepare and tokenize text data for next-word prediction.
- Build LSTM and GRU models using PyTorch.
- Train models using a word-level vocabulary.
- Compare model performance using loss and accuracy metrics.
- Generate next-word predictions for custom text inputs.

## Dataset

The project uses WikiText-103 data stored in Parquet files.

- Training data: sampled 20,000 non-empty text records across the training shards.
- Validation data: used to monitor performance and select checkpoints.
- Test data: reserved for final model evaluation.
- Vocabulary size: 30,000 tokens, including the `<UNK>` token.
- Input sequence length: 5 words.

The data is cleaned by trimming text and normalizing whitespace. Punctuation is preserved.

## Model Architecture

Both models use an embedding layer, a recurrent layer, and a fully connected output layer.

**LSTM model**
1. Word embedding layer (128 dimensions).
2. LSTM layer (256 hidden units).
3. Fully connected layer producing scores for the vocabulary.

**GRU model**
1. Word embedding layer (128 dimensions).
2. GRU layer (256 hidden units).
3. Fully connected layer producing scores for the vocabulary.

The models are trained using cross-entropy loss and the Adam optimizer.

## Final Test Results

| Metric | LSTM | GRU |
|---|---:|---:|
| Test loss | 5.4936 | 5.6770 |
| Top-1 accuracy | 21.11% | 20.43% |
| Top-5 accuracy | 40.81% | 39.86% |

In this evaluation, LSTM achieved lower test loss and higher top-1 and top-5 accuracy than GRU.

These results are baseline results from the current training setup, not a claim of state-of-the-art performance.

## Overfitting Observation

During additional training, training loss continued to decrease while validation loss increased. This indicates overfitting. The project tracks validation loss to retain the best checkpoint within each additional training run.

Further improvements could include dropout, learning-rate scheduling, early stopping, a larger or better-filtered training sample, and systematic hyperparameter tuning.

## Visualizations

The project generates:

- `training_loss_comparison.png`
- `test_accuracy_comparison.png`
- `final_test_results.csv`

These files are saved in the `models/` directory.

## Technologies Used

- Python
- PyTorch
- pandas
- NumPy
- Matplotlib
- Jupyter Notebook
- WikiText-103 dataset

## How to Run

1. Clone or download this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Open the notebook in `src/inspect_dataset.ipynb`.
4. Run the notebook cells in order, ensuring the dataset paths match your local folder structure.
5. Review the evaluation metrics, graphs, and prediction examples.

## Limitations

- The vocabulary is limited to 30,000 tokens.
- The input context is limited to five words.
- Training uses a sample of the available training records.
- Word-level prediction can struggle with rare and out-of-vocabulary words.
- The models may overfit the training sample.

## Future Improvements

- Experiment with dropout and early stopping.
- Tune sequence length, vocabulary size, and model dimensions.
- Train on a larger portion of the dataset.
- Compare with a Transformer-based language model.
- Develop a simple interactive interface for entering text and viewing predictions.

## Author

Developed as a machine learning and deep learning project for educational purposes.
