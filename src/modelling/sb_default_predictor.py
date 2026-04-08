import copy
import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

# Hyper params
HIDDEN_SIZES = [128, 128, 64, 32]
LR = 0.0001
EPOCHS = 50
DROPOUT = 0.2
BATCH_SIZE = 32


class DefaultPredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout_rate=0.0):
        super().__init__()

        layers = []
        curr_dim = in_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(curr_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            curr_dim = hidden_dim

        # don't sigmoid the output since we're using BCEWithLogitsLoss
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def find_best_threshold(y_true, probs):
    """
    Find output probability threshold that accurately classifies defaults
    """
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.60, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1


def main():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    train_data = np.genfromtxt(data_dir / "sb_train_data.csv", delimiter=",")
    val_data   = np.genfromtxt(data_dir / "sb_val_data.csv",   delimiter=",")
    test_data  = np.genfromtxt(data_dir / "sb_test_data.csv",  delimiter=",")

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val,   y_val   = val_data[:,   :-1], val_data[:,   -1]
    X_test,  y_test  = test_data[:,  :-1], test_data[:,  -1]

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    # Oversampling done because some batches would be passed through with no default examples
    # Also pass a corresponding positive weight to the loss function to compensate for the 
    # over sampling and still existing class imbalance
    OVERSAMPLE_RATIO = 5
    n_oversample = n_neg // OVERSAMPLE_RATIO
    effective_pos_weight = n_neg / n_oversample 
    print(f"Class balance — PIF: {n_neg}, CHGOFF: {n_pos}")
    print(f"Oversampling CHGOFF to {n_oversample} per epoch (ratio 1:{OVERSAMPLE_RATIO}), pos_weight={effective_pos_weight:.1f}")

    chgoff_idx = np.where(y_train == 1)[0]
    pif_idx = np.where(y_train == 0)[0]
    rng = np.random.default_rng(42)

    model = DefaultPredictorMLP(X_train.shape[1], HIDDEN_SIZES, DROPOUT)
    pos_weight = torch.tensor([effective_pos_weight], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Early stopping: keep weights from the best val AUROC epoch
    best_auroc = 0.0
    best_weights = None

    for epoch in range(EPOCHS):
        model.train()
        # Moderate oversampling: expand CHGOFF to n_oversample samples per epoch
        oversampled_chgoff = rng.choice(chgoff_idx, size=n_oversample, replace=True)
        bal_idx = np.concatenate([pif_idx, oversampled_chgoff])
        rng.shuffle(bal_idx)
        X_shuf = X_train[bal_idx]
        y_shuf = y_train[bal_idx]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_shuf), BATCH_SIZE):
            inputs = torch.tensor(X_shuf[i:i+BATCH_SIZE],  dtype=torch.float32)
            labels = torch.tensor(y_shuf[i:i+BATCH_SIZE],  dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(torch.tensor(X_val, dtype=torch.float32)).numpy()
        val_auroc = roc_auc_score(y_val, val_probs)

        improved = " *" if val_auroc > best_auroc else ""
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  val_auroc={val_auroc:.4f}{improved}")

        # early stopping checks
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_weights = copy.deepcopy(model.state_dict())

    # Reload best model recorded
    model.load_state_dict(best_weights)

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "default_predictor.pt")
    print(f"Model saved to {models_dir / 'default_predictor.pt'}")

    model.eval()
    with torch.no_grad():
        val_probs  = model.predict_proba(torch.tensor(X_val,  dtype=torch.float32)).numpy()
        test_probs = model.predict_proba(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Find optimal probability threshold on val set, then apply to test
    best_thresh, val_f1 = find_best_threshold(y_val, val_probs)
    print(f"Optimal threshold (from val set): {best_thresh:.2f}  (val F1={val_f1:.4f})")

    test_preds = (test_probs >= best_thresh).astype(int)

    auroc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    print(f"\n--- Test Results (threshold={best_thresh:.2f}) ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1 (defaults): {f1:.4f}")
    print(f"\nConfusion matrix:")
    print(f"              Pred PIF  Pred CHGOFF")
    print(f"Actual PIF    {cm[0,0]}        {cm[0,1]}")
    print(f"Actual CHGOFF {cm[1,0]}          {cm[1,1]}")


if __name__ == "__main__":
    main()
