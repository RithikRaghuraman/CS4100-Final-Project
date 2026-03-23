import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Hyper params
HIDDEN_SIZES = [128, 64, 32]
LR = 0.001
EPOCHS = 20
DROPOUT = 0.3



class DefaultPredictorMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout_rate=0.0):
        super().__init__()

        layers = []
        curr_dim = in_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(curr_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.argmax(logits).item()

def main():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "cleaned_business_loans_2010_2025.csv"

    # processed data for modelling?
    data = pd.read_csv(data_path)
    labels = data["loanstatus"].to_numpy()
    classes, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    data = data.drop(["asofdate", "program", "borrname", "loanstatus", "firstdisbursementdate"], 
                     axis=1).to_numpy(dtype=np.float32)

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = DefaultPredictorMLP(train_data.shape[1], 2, HIDDEN_SIZES, DROPOUT)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        idx = np.arange(len(train_data))
        random.shuffle(idx)
        train_data = train_data[idx]
        train_labels = train_labels[idx]
        for i in range(0, len(train_data), 32):
            batch_data = train_data[i:i+32]
            batch_labels = train_labels[i:i+32]
            print(batch_data[0], batch_labels)
            inputs = torch.tensor(batch_data, dtype=torch.float32)
            labels = torch.tensor(batch_labels, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")




if __name__ == "__main__":
    main()
