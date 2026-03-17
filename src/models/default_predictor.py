import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
import pandas as pd
import random

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
    data = data.drop(["asofdate", "program", "borrname"], axis=1)
    numerical_data = data



    model = DefaultPredictorMLP(data.shape[1], 2, HIDDEN_SIZES, DROPOUT)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(data)




if __name__ == "__main__":
    main()
