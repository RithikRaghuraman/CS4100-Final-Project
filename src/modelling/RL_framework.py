import sys
import random
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent))
from sb_default_predictor import DefaultPredictorMLP
from homeloan_default_predictor import LoanClassifier




LGD = 0.75   # Percentage of loan amount lost on default
SB_RATE = 0.065  # Average SB loan interest rate - https://www.crestmontcapital.com/blog/sba-loan-interest-rates-historical-trends-and-2025-updates
SB_TERM = 84     # Assumed SB loan term in months (7 years)


SB_FEATURE_DIM = 94    # Number of features in SB loan data
HL_FEATURE_DIM = 11    # Number of features in HL loan data
STATE_DIM = 96    # SB_FEATURE_DIM + p_default + loan_type flag
ENCODED_DIM = 32    # Output size of each loan type encoder, used to manage mismatch of features between loan types
TRUNK_HIDDEN = 64    # Hidden layer size in shared trunk
N_ACTIONS = 2     # just approve or reject loans


GAMMA = 0.99
EPSILON = 0.99
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 1e-4
BATCH_SIZE = 128
BUFFER_SIZE = 10000 
TARGET_UPDATE_FREQ = 10 
NUM_EPISODES = 500
MAX_STEPS = 100

def load_classifiers():
    """Load in the trained default predictors """
    models_dir = Path(__file__).parent.parent / "models"

    sb_model = DefaultPredictorMLP(
        in_dim=SB_FEATURE_DIM, hidden_dims=[128, 128, 64, 32], dropout_rate=0.2
    )
    sb_model.load_state_dict(
        torch.load(models_dir / "default_predictor.pt", weights_only=True)
    )
    sb_model.eval()

    hl_model = LoanClassifier(input_dim=HL_FEATURE_DIM)
    hl_model.load_state_dict(
        torch.load(models_dir / "homeloan_default_predictor.pt", weights_only=True)
    )
    hl_model.eval()

    return sb_model, hl_model


class LoanEnvironment(gym.Env):
    """
    Gymnasium environment for loan approval decisions.
    """

    def __init__(self, sb_data, hl_data):
        super().__init__()
        self.sb_features = sb_data[:, :SB_FEATURE_DIM].astype(np.float32)
        self.hl_features = hl_data[:, :HL_FEATURE_DIM].astype(np.float32)

        self.sb_model, self.hl_model = load_classifiers()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.step_count = 0
        self.current_state = None

    def _sample_state(self):
        """Sample one loan at random, use corresponding classifier to get estimated
        default probability and return state representation.
        Note: state representation at this step is padded with 0s for dim mismatch, 
        but DQN enconders will collapse these down to ENCODER_DIM using nn
        """
        if random.random() < 0.5:
            idx = random.randrange(len(self.sb_features))
            features = self.sb_features[idx]
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            p_default = self.sb_model.predict_proba(x).item()
            loan_type = 0.0
        else:
            idx = random.randrange(len(self.hl_features))
            features = self.hl_features[idx]
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            p_default = self.hl_model.predict_proba(x).item()
            loan_type = 1.0

        padded = np.zeros(SB_FEATURE_DIM, dtype=np.float32)
        padded[:len(features)] = features
        return np.append(padded, [p_default, loan_type]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_state = self._sample_state()
        return self.current_state, {}

    def step(self, action):
        features  = self.current_state[:SB_FEATURE_DIM]
        p_default = float(self.current_state[-2])
        loan_type = int(self.current_state[-1])
        p_repay   = 1.0 - p_default

        if action == 0:  # REJECT
            reward = 0.0
        else:            # APPROVE
        # reward for both loan types is essentially expected profit minus the expected loss
        # Expectation is used to weight the profit and loss by the probability of repayment vs default 
            if loan_type == 0: # SB loan — rate and term use external constants
                loan_proxy = features[0]
                reward = (loan_proxy * SB_RATE * SB_TERM / 12) * p_repay \
                         - loan_proxy * p_default * LGD
            else: # HL loan — rate, amount, term are in features
                rate_proxy = features[0]
                loan_proxy = features[1]
                term_proxy = features[2]
                reward = (rate_proxy * term_proxy * loan_proxy) * p_repay \
                         - loan_proxy * p_default * LGD

        self.step_count += 1
        done = self.step_count >= MAX_STEPS
        self.current_state = self._sample_state()

        return self.current_state, float(reward), done, False, {}


class ReplayBuffer:
    """ Data struct to store past experiences of network.
        Used to decouple learning from any ordering in the data."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Q-network with type-specific input encoders and a shared decision trunk.

    SB and HL features have different dimensions and semantics, so each loan
    type is compressed by its own encoder before the shared trunk produces
    Q-values.  Batches containing mixed loan types are handled with index masks
    so each encoder only ever sees its own features.

    Takes in a state vector of loan info with estimated default prob and type tacked on at the end
    Outputs estimated Q-values for both actions
    """

    def __init__(self):
        super().__init__()
        self.sb_encoder = nn.Sequential(nn.Linear(SB_FEATURE_DIM, ENCODED_DIM), nn.ReLU())
        self.hl_encoder = nn.Sequential(nn.Linear(HL_FEATURE_DIM, ENCODED_DIM), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(ENCODED_DIM + 1, TRUNK_HIDDEN),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN, N_ACTIONS),
        )

    def forward(self, features, p_default, loan_type):
        batch_size = features.shape[0]
        encoded = torch.zeros(batch_size, ENCODED_DIM)

        # mask features to ensure the encoder only looks at relevant features for that loan type
        # before enocding to consistent dim
        sb_mask = (loan_type == 0)
        hl_mask = (loan_type == 1)
        if sb_mask.any():
            encoded[sb_mask] = self.sb_encoder(features[sb_mask])
        if hl_mask.any():
            encoded[hl_mask] = self.hl_encoder(features[hl_mask, :HL_FEATURE_DIM])

        combined = torch.cat([encoded, p_default.unsqueeze(1)], dim=1)
        return self.trunk(combined)


class DQNAgent:
    """
    DQN agent with epsilon-greedy exploration, experience replay, and a
    target network for stable Bellman targets.
    """

    def __init__(self):
        self.policy_net = DQNNetwork()
        self.target_net = DQNNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON

    def _unpack_state(self, state):
        """Helper to split state into the three inputs we need"""
        features  = torch.tensor(state[:SB_FEATURE_DIM], dtype=torch.float32).unsqueeze(0)
        p_default = torch.tensor([state[-2]], dtype=torch.float32)
        loan_type = torch.tensor([int(state[-1])], dtype=torch.long)
        return features, p_default, loan_type

    def act(self, state):
        """ Epsilon-greedy action selection """
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        self.policy_net.eval()
        with torch.no_grad():
            features, p_default, loan_type = self._unpack_state(state)
            q_vals = self.policy_net(features, p_default, loan_type)
        return q_vals.argmax().item()

    def learn(self):
        """ Sample a batch from the buffer and perform one update """
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        features = states[:, :SB_FEATURE_DIM]
        p_defaults = states[:, -2]
        loan_types = states[:, -1].long()

        next_features  = next_states[:, :SB_FEATURE_DIM]
        next_p_defaults = next_states[:, -2]
        next_loan_types = next_states[:, -1].long()

        self.policy_net.train()
        current_q = self.policy_net(features, p_defaults, loan_types) \
                        .gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_features, next_p_defaults, next_loan_types) \
                             .max(1).values
            target_q = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self):
        """Copy policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())




def train(env, agent):
    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            action                          = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            if done:
                break

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            agent.sync_target()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:>4} | Reward: {total_reward:>10.2f} | Epsilon: {agent.epsilon:.3f}")


def main():
    import pandas as pd
    project_root = Path(__file__).parent.parent.parent

    sb_data = pd.read_csv(
        project_root / "data" / "sb_train_data.csv", header=None
    ).values.astype("float32")

    hl_data = pd.read_csv(
        project_root / "data" / "homeloan_train.csv", header=None, nrows=100_000
    ).values.astype("float32")
    print("Loaded data")

    env   = LoanEnvironment(sb_data, hl_data)
    agent = DQNAgent()
    train(env, agent)

    out_path = project_root / "src" / "models" / "rl_policy.pt"
    torch.save(agent.policy_net.state_dict(), out_path)
    print(f"Policy saved to {out_path}")


if __name__ == "__main__":
    main()