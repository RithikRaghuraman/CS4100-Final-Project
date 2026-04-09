import numpy as np
"""
Load packages for RL Model
"""

def sample_loans(sb_data, hl_data, n_samples=1000):
    sb_samples = sb_data[np.random.choice(sb_data.shape[0], n_samples, replace=False)]
    hl_samples = hl_data[np.random.choice(hl_data.shape[0], n_samples, replace=False)]
    sb_feat = sb_samples[:, :-1]
    hl_feat = hl_samples[:, :-1]
    sb_labels = sb_samples[:, -1]
    hl_labels = hl_samples[:, -1]

    return sb_feat, hl_feat, sb_labels, hl_labels

"""
PREREQUISITES

LOAD BOTH BINARY CLASSIFIERS FOR INDIVIDUAL DATA SETS
    - input: feature vector
    - Output: P(default) as Float 0 - 1

DEFINE CONSTRAINTS:
    - LGD  = #              # Loss given default assumption
    - GAMMA = #             # discount factor (favoring future rewards)
    - EPSILON = #           # exploration factor
    - EPSILON_MIN = #       # minimum exploration factor
    - EPSILON_DECAY = #     # decay rate for exploration factor
    - LEARNING_RATE = #     # learning rate
"""

"""
DEFINE THE ENVIRONMENT

CLASS loanEnvironment:

    FOR episode in data:
    reset():            # Reset the environment for a new episode // 
                          Sample the next state for each episode

                        p_default = output from binary classifier for current episode
                        state = concatinate the episode features with p_default
                        return state

    step(action):
        IF action == REJECT:
            reward = 0

        IF action == APPROVE:
            reward = (interest * P(repay)) - (loan_amount * P(default) * LGD)

            # Calculate Interest:
            # Interest = calculate_interest(applicant.loan_amount,
            #                               applicant.rate,
                                            applicant.term)

            def calculate_interest():
                monthly_rate = annual_rate / 12
                monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)

            The interest value will determine the reward for the APPROVE action.
                - the higher the interest the more worth the loan is

        THEN:
        SAMPLE next state
        Calculate p_default
        Concatenate the episode features with p_default
        Repeat

        FINISHED WHEN episode_step_count > max_steps_per_episodes

        RETURN reward
"""

"""
AGENT



"""

"""
TRAINING LOOP

AGENT = agent
ENV = LoanEnvironment()

FOR EPISODE IN RANGE(NUM_EPISODES):

    state = env.reset()
    total_reward = 0
    FOR steps IN range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        IF done:
            BREAK
    



"""
def main():
    sb_data = np.genfromtxt("sb_data.csv", delimiter=",")
    hl_data = np.genfromtxt("hl_data.csv", delimiter=",")

    sb_features, hl_features, sb_labels, hl_labels = sample_loans(sb_data, hl_data, n_samples=1000)


if __name__ == "__main__":
    main()