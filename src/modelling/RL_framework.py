from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np


# Paths
SB_CLASSIFIER_PATH  = "src/models/default_predictor.pt"
HL_CLASSIFIER_PATH  = "src/models/homeloan_default_predictor.pt"
SB_TRAIN_PATH       = "data/sb_train_data.csv"
HL_TRAIN_PATH       = "data/homeloan_train.csv"
SB_RAW_PATH         = "data/sb_train_raw.csv"

# DEFINE CONSTRAINTS:
LGD = 0.4
SB_RATE = 0.065  # Fixed interest rate for small business loans (annual)
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_STEPS = 500
NUM_EPISODES = 300
REPLAY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 10

# Action Space
REJECT = 0
APPROVE = 1

STATE_DIM = 4 # P(default)
              # loan amount (log-scaled)
              # norm_term (loan term / 360)
              # loan_type_flag (1 for small business, 0 for home loan)


# def sample_loans(sb_data, hl_data, n_samples=1000):
#     sb_samples = sb_data[np.random.choice(sb_data.shape[0], n_samples, replace=False)]
#     hl_samples = hl_data[np.random.choice(hl_data.shape[0], n_samples, replace=False)]
#     sb_feat = sb_samples[:, :-1]
#     hl_feat = hl_samples[:, :-1]
#     sb_labels = sb_samples[:, -1]
#     hl_labels = hl_samples[:, -1]

#     return sb_feat, hl_feat, sb_labels, hl_labels

def load_loan_records():
    records = []

    # Small Business Loans
    sb_data = np.genfromtxt(SB_TRAIN_PATH, delimiter=",")
    sb_raw  = np.genfromtxt(SB_RAW_PATH, delimiter=",", skip_header=1)
    for features, raw in zip(sb_data, sb_raw):
        records.append({
            'features':    features[:-1].astype(np.float32),
            'loan_amount': float(raw[0]),         # grossapproval
            'rate':        SB_RATE,
            'term':        float(raw[1]),          # terminmonths
            'loan_type':   0,
        })

    # Home loans
    hl_data = np.genfromtxt(HL_TRAIN_PATH, delimiter=",")
    for row in hl_data:
        records.append({
            'features':    row[:-1].astype(np.float32),
            'loan_amount': float(row[1]),          # Original UPB
            'rate':        float(row[0]) / 100.0,  # Original Interest Rate
            'term':        float(row[2]),           # Original Loan Term
            'loan_type':   1,
        })

    return records

class SBClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        layers = []
        curr = in_dim
        for h in [128, 128, 64, 32]:
            layers += [nn.Linear(curr, h), nn.ReLU(), nn.Dropout(0.2)]
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class HLClassifier(nn.Module):
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

class QNetwork(nn.Module):
    def __init__(self, state_dim = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Q(s, REJECT/APPROVE)
        )

    def forward(self, x):
        return self.net(x)

# INTEREST CALCULATION
def calculate_total_interest(loan_amount, annual_rate, term_months):
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return 0.0
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) \
                      / ((1 + monthly_rate)**term_months - 1)
    total_paid = monthly_payment * term_months

    return total_paid - loan_amount


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return(
            torch.FloatTensor(states), 
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork().to(DEVICE)
        self.target_net = QNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer()
        self.epsilon = EPSILON

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).to(DEVICE)).argmax().item()

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = (
            states.to(DEVICE), actions.to(DEVICE), rewards.to(DEVICE),
            next_states.to(DEVICE), dones.to(DEVICE)
        )
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * max_next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ENVIRONMENT
class LoanEnvironment:
    def __init__(self, records, sb_classifier, hl_classifier):
        self.records = records
        self.classifiers = {0: sb_classifier, 1: hl_classifier}
        self.step_count = 0
        self.episode_loans = []
        self.current_loan = None

    def reset(self):
        self.step_count = 0
        self.episode_loans = random.sample(self.records, MAX_STEPS)
        self.current_loan = self.episode_loans[0]
        return self.build_state(self.current_loan)
    
    def step(self, action):
        reward = self.compute_reward(action, self.current_loan)
        self.step_count += 1
        done = (self.step_count >= MAX_STEPS)
        if not done:
            self.current_loan = self.episode_loans[self.step_count]
            next_state = self.build_state(self.current_loan)
        else:
            next_state = np.zeros(STATE_DIM)
        return next_state, reward, done
    
    def get_p_default(self, loan):
        clf = self.classifiers[loan['loan_type']]
        features = torch.FloatTensor(loan['features']).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = clf(features)
        return torch.sigmoid(logit).item()
    
    def build_state(self, loan):
        p_default = self.get_p_default(loan)
        log_norm_amount = np.log10(loan['loan_amount'] + 1) / 7.0
        norm_term = loan['term'] / 360.0
        loan_type_flag = float(loan['loan_type'])
        return np.array([p_default, log_norm_amount, norm_term, loan_type_flag], dtype=np.float32)

    def compute_reward(self, action, loan):
        p_default = self.get_p_default(loan)
        if action == REJECT:
            return p_default * 0.1
        p_repay  = 1.0 - p_default
        interest = calculate_total_interest(loan['loan_amount'], loan['rate'], loan['term'])
        profit   = (interest * p_repay) - (loan['loan_amount'] * p_default * LGD)
        return profit / loan['loan_amount']


def train():
    sb_feat_dim = np.genfromtxt(SB_TRAIN_PATH, delimiter=",", max_rows=1).shape[0] - 1
    hl_feat_dim = np.genfromtxt(HL_TRAIN_PATH, delimiter=",", max_rows=1).shape[0] - 1

    sb_clf = SBClassifier(in_dim=sb_feat_dim)
    sb_clf.load_state_dict(torch.load(SB_CLASSIFIER_PATH, map_location="cpu"))
    sb_clf.to(DEVICE).eval()

    hl_clf = HLClassifier(input_dim=hl_feat_dim)
    hl_clf.load_state_dict(torch.load(HL_CLASSIFIER_PATH, map_location="cpu"))
    hl_clf.to(DEVICE).eval()

    records = load_loan_records()
    env     = LoanEnvironment(records, sb_clf, hl_clf)
    agent   = DQNAgent()

    for episode in range(NUM_EPISODES):
        state        = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action                   = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.learn()
            state        = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        print(f"Episode {episode:3d} | Reward: {total_reward:>10.4f} | Epsilon: {agent.epsilon:.3f}")


if __name__ == "__main__":
    train()



