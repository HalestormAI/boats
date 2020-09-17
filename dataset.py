import torch
import torch.utils.data
import pandas as pd


def load(filename, training_prop):
    raw_data = pd.read_csv(filename)

    # Some key data is missing. We'll just discard those rows
    data = raw_data[["Age", "Pclass", "Sex", "Survived"]].dropna()

    def create_dataset(data):
        dataset = pd.DataFrame()
        dataset["age"] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
        dataset["class"] = (data['Pclass'] - data['Pclass'].mean()) / data['Pclass'].std()
        dataset["gender"] = (data["Sex"] == "male").astype(int)
        return torch.utils.data.TensorDataset(
            torch.tensor(dataset.values, dtype=torch.float),
            torch.tensor(data["Survived"].astype(int).values, dtype=torch.long)
        )

    num_training_samples = int(len(data) * training_prop)

    training_data = data.iloc[:num_training_samples, :]
    validation_data = data.iloc[num_training_samples:, :]

    return create_dataset(training_data), create_dataset(validation_data)
