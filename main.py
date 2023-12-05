import pandas as pd
from typing import List, Tuple
import numpy as np


def query_record(inputs: List[List[float]], output: List[float], index: int) -> None:
    print(f"Record {index}:")
    print("Inputs:", inputs[index])
    print("Output:", output[index])

def cost_function(X: List[List[float]], y: List[float], weights: List[float]) -> float:
    predictions = [sum(x_i * w_i for x_i, w_i in zip(x, weights)) for x in X]
    cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / len(y)
    return cost


def derivative_cost_function(X: List[List[float]], y: List[float], weights: List[float]) -> List[float]:
    m = len(y)  # Number of samples
    predictions = [sum(x_i * w_i for x_i, w_i in zip(x, weights)) for x in X]

    # Initialize derivatives as a list of zeros
    derivatives = [0 for _ in range(len(weights))]

    # Iterate through each feature
    for j in range(len(weights)):
        # Sum the derivative for each sample
        for i in range(m):
            derivatives[j] += (predictions[i] - y[i]) * X[i][j]
        # Average the derivative over all samples
        derivatives[j] /= m

    return derivatives




def optimizer(X: List[List[float]], y: List[float], weights: List[float], learning_rate: float, iterations: int) -> List[float]:
    m = len(y)
    derivatives = [0 for _ in range(len(weights))]  # Initialize derivatives

    for _ in range(iterations):
        # Update derivatives in each iteration
        for j in range(len(weights)):
            derivatives[j] = sum((sum(x_i * w_i for x_i, w_i in zip(x, weights)) - y[i]) * X[i][j] for i, x in enumerate(X)) / m

        # Update weights
        weights = [w - learning_rate * dw for w, dw in zip(weights, derivatives)]

    return weights



def train_model(X, y, learning_rate, iterations, l1_ratio, alpha) -> List[float]:
    weights = [0.0 for _ in range(len(X[0]))]  # Initialize weights as a Python list
    for _ in range(iterations):
        derivatives = derivative_cost_function(X, y, weights)
        clipped_derivatives = clip_gradients(derivatives, threshold=1.0)  # Implement gradient clipping
        weights = [w - learning_rate * dw for w, dw in zip(weights, clipped_derivatives)]
    return weights  # Return the final trained weights as a list



def sigmoid(z):
    if z < -20:  # Threshold to avoid too large negative values
        z = -20
    return 1 / (1 + np.exp(-z))

def clip_gradients(derivatives, threshold):
    return [max(min(dw, threshold), -threshold) for dw in derivatives]



def elastic_net_cost_function(X: List[List[float]], y: List[float], weights: List[float], l1_ratio: float, alpha: float) -> float:
    l1_penalty = l1_ratio * sum(abs(w) for w in weights)
    l2_penalty = (1 - l1_ratio) * sum(w ** 2 for w in weights)
    cost = cost_function(X, y, weights) + alpha * (l1_penalty + l2_penalty)
    return cost


def predict(X_new: List[List[float]], weights: List[float]) -> List[float]:
    predictions = [sum(x_i * w_i for x_i, w_i in zip(x, weights)) for x in X_new]
    return predictions



def read_dataset(file_path: str):
    # Read the Excel file, skipping initial non-relevant rows
    # Adjust 'skiprows' as needed to skip header/metadata rows
    df = pd.read_excel(file_path, skiprows=2)

    # Drop non-numeric columns and rows with missing values
    df = df.select_dtypes(include=[np.number]).dropna()

    # Split into inputs and output
    inputs = df.iloc[:, :-1].values.tolist()  # All columns except the last one as inputs
    output = df.iloc[:, -1].values.tolist()   # The last column as output

    return inputs, output

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataset by removing non-numeric columns, handling missing values,
    and normalizing the data.

    :param df: pd.DataFrame - The raw dataset.
    :return: pd.DataFrame - The preprocessed dataset.
    """
    # Remove non-numeric columns
    df = df.select_dtypes(include=[np.number])

    # Drop rows with NaN values or fill them
    df = df.dropna()  # or df.fillna(df.mean())

    # Normalize or standardize the data
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    return df



def main() -> None:
    # Path to the dataset file
    file_path = 'data/dataset.xlsx'

    # Read the dataset and extract inputs and outputs
    inputs, output = read_dataset(file_path)

    # Query a specific record, for example, the 4500th record
    record_index = 4500
    if record_index < len(inputs):
        print(f"Record {record_index}:")
        print("Inputs:", inputs[record_index])
        print("Output:", output[record_index])
    else:
        print(f"Record {record_index} is out of range.")

    # Model Hyperparameters
    learning_rate = 0.001
    iterations = 1000
    l1_ratio = 0.5  # Adjust as needed
    alpha = 0.1  # Adjust as needed

    # Train the model with Elastic Net regularization
    trained_weights = train_model(inputs, output, learning_rate, iterations, l1_ratio, alpha)
    print(trained_weights)
    # You can now use the trained model for predictions or further analysis


if __name__ == '__main__':
    main()



