## AI Credit Portfolio Manager

### Project Overview
This project explores how Reinforcement and Machine Learning can be used to optimize credit lending decisions and limit default risk. Given a stream of loan applicants, a DQN-based agent decides whether to approve or reject each loan while balancing expected profit against default risk and a finite capital budget.

The pipeline combines two supervised default-probability predictors (one for SBA small-business loans, one for Fannie Mae single-family home loans) with a shared RL policy that learns over a mixed pool of loan types. The agent observes loan features, the classifier's predicted probability of default, and a loan-type flag, then chooses to approve or reject. Rewards model interest income on performing loans and loss-given-default on bad loans. The trained policy is benchmarked against a random baseline and a greedy threshold-on-p_default baseline across cumulative profit, approval rate, default rate, and loan-type mix.

### Running The Project
1. Clone the repository
```bash
git clone https://github.com/<owner>/CS4100-Final-Project.git
cd CS4100-Final-Project
```

2. Create and activate a virtual environment, then install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. (Optional) Retrain the default predictors and RL policy
```bash
python src/modelling/sb_default_predictor.py
python src/modelling/homeloan_default_predictor.py
python src/modelling/RL_framework.py
```

4. Evaluate the trained policy against the baselines and regenerate plots in `results/`
```bash
python src/modelling/evaluate_policy.py
```


### Directory Structure
```
CS4100-Final-Project/
├── data/                                    # Cleaned and raw train/val/test splits for SB and HL loans
├── results/                                 # Evaluation plots (profit, approval vs. default, loan mix, etc.)
├── single family home loan subset/          # Pointer to the downloadable Fannie Mae subset
├── src/
│   ├── preprocessing/                       # Data cleaning and feature-engineering scripts/notebooks
│   │   ├── clean_small_business_data.ipynb
│   │   ├── transform_sba.py
│   │   ├── business_loan_sampling.py
│   │   └── fannie_mae_*.ipynb
│   ├── modelling/
│   │   ├── sb_default_predictor.py          # MLP default classifier for SBA loans
│   │   ├── homeloan_default_predictor.py    # MLP default classifier for home loans
│   │   ├── RL_framework.py                  # Gym env, DQN network, and training loop
│   │   └── evaluate_policy.py               # Benchmarks DQN vs. Greedy and Random baselines
│   └── models/                              # Saved model weights (.pt)
├── Glossary.ipynb / Glossary.xlsx           # Feature definitions for the loan datasets
├── requirements.txt
└── README.md
```

### Key Findings
- **DQN outperforms both baselines on cumulative profit** across mixed SB/HL loan pools under a fixed capital budget, confirming that RL can learn a policy that beats a simple p_default threshold rule.
- **Raw default-probability predictors alone are insufficient.** The greedy agent (approve if p_default below threshold) leaves profit on the table by over-rejecting marginally risky loans that are still positive-EV given their interest rate and term.
- **The agent was initially over-averse to SB loans** because loss-given-default dominated expected interest income. Introducing a `DEFAULT_PENALTY_SCALE` of 0.1 rebalanced the reward and let the agent approve profitable SB loans without collapsing into a reject-all policy.
- **Loan-type encoding matters.** SB and HL loans have very different feature dimensions (95 vs. 11), so the policy uses per-loan-type encoders projecting into a shared 32-dim space before a common trunk. This lets a single policy generalize across both products.
- **Default rates stay comparable to the greedy baseline** while approval rate and profit both increase, suggesting the DQN is selecting better loans rather than simply approving more.

### Collaborators
- Jack Carroll
- Jaden Hu
- Elton Neman
- Rithik Raghuraman
