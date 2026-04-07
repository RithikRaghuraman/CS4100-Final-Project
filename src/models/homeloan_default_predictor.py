import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_parquet('subset.parquet')

X = data[[
    'Original Interest Rate', 
    'Original UPB', 
    'Original Loan Term', 
    'Original Loan to Value Ratio (LTV)',
    'Original Combined Loan to Value Ratio (CLTV)',
    'Debt-To-Income (DTI)',
    'Current Actual UPB',
    'Loan Age',
    'Remaining Months To Maturity',
    'Channel'
    ]]

y = (data['Current Loan Delinquency Status'] > 0).astype(int)

X = X.fillna(0)
X = pd.get_dummies(X, columns=['Channel'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    
model = LoanClassifier(input_dim=X_train_tensor.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimzer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimzer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss/len(train_loader)


    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')


model.eval()
all_true = []
all_probs = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        
        all_true.extend(y_batch.numpy().flatten())
        all_probs.extend(probs.numpy().flatten()) 

for t in [0.5, 0.4, 0.3, 0.2]:
    preds = (np.array(all_probs) >= t).astype(int)

    print('Accuracy:', accuracy_score(all_true, preds))
    print('Precision:', precision_score(all_true, preds, zero_division=0))
    print('Recall:', recall_score(all_true, preds, zero_division=0))
    print('F1:', f1_score(all_true, preds, zero_division=0))

print('ROC-AUC', roc_auc_score(all_true, all_probs))
