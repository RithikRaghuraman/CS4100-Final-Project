"""
Evaluate the trained DQN loan policy against random agent and greedy based on default prob agent
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from RL_framework import (
    DQNNetwork,
    load_classifiers,
    SB_FEATURE_DIM,
    HL_FEATURE_DIM,
    LGD,
    SB_RATE,
    SB_TERM,
)

N_LOANS = 1000
N_TRIALS = 100
HL_CAP = 100000
CAPITAL_LIMIT = 50000000
SEED = 0

COLORS = {"DQN": "#2196F3", "Greedy": "#C50753", "Random": "#FFED22"}
STRATEGIES = ["DQN", "Greedy", "Random"]

def load_test_data(project_root: Path):
    sb_data = pd.read_csv(
        project_root / "data" / "sb_test_data.csv", header=None
    ).values.astype("float32")

    hl_data = pd.read_csv(
        project_root / "data" / "homeloan_test.csv", header=None, nrows=HL_CAP
    ).values.astype("float32")

    sb_raw = pd.read_csv(
        project_root / "data" / "sb_test_raw.csv"
    ).values.astype("float32")

    hl_raw = pd.read_csv(
        project_root / "data" / "homeloan_test_raw.csv", header=None, nrows=HL_CAP
    ).values.astype("float32")

    sb_feats  = sb_data[:, :SB_FEATURE_DIM]
    sb_labels = sb_data[:, -1].astype(int)
    hl_feats  = hl_data[:, :HL_FEATURE_DIM]
    hl_labels = hl_data[:, -1].astype(int)

    print(f"Loaded {len(sb_feats)} SB test loans, {len(hl_feats)} HL test loans")
    return sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw

def build_state(features: np.ndarray, p_default: float, loan_type: int) -> np.ndarray:
    """ Helper to build state vector that RL agent needs as input"""
    padded = np.zeros(SB_FEATURE_DIM, dtype=np.float32)
    padded[: len(features)] = features
    return np.append(padded, [p_default, float(loan_type)]).astype(np.float32)


def build_loan_pool(
    sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw,
    sb_model, hl_model,
    n_loans: int = N_LOANS,
):
    """
    Sample equal amount of home and business loans
    """
    pool = []
    n_each = n_loans // 2

    sb_idx = np.random.choice(len(sb_feats), size=n_each, replace=False)
    for i in sb_idx:
        features = sb_feats[i]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        p_default = sb_model.predict_proba(x).item()
        pool.append({
            "features":    features,
            "state":       build_state(features, p_default, 0),
            "p_default":   p_default,
            "label":       int(sb_labels[i]),
            "loan_type":   0,
            "gross_usd":   float(sb_raw[i, 0]),
            "term_months": float(sb_raw[i, 1]),
        })


    hl_idx = np.random.choice(len(hl_feats), size=n_each, replace=False)
    for i in hl_idx:
        features = hl_feats[i]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        p_default = hl_model.predict_proba(x).item()
        pool.append({
            "features":    features,
            "state":       build_state(features, p_default, 1),
            "p_default":   p_default,
            "label":       int(hl_labels[i]),
            "loan_type":   1,
            "rate_pct":    float(hl_raw[i, 0]),
            "upb_usd":     float(hl_raw[i, 1]),
            "term_months": float(hl_raw[i, 2]),
        })

    random.shuffle(pool)
    return pool


def get_loan_amount(loan: dict) -> float:
    """ Helprer to get dollar amt of loan"""
    return loan["gross_usd"] if loan["loan_type"] == 0 else loan["upb_usd"]

def realized_profit(loan: dict, action: int) -> float:
    """
    Calculate profit/loss in real dollars for a given loan based on its delinquency status 
    and whether or not the model approved it
    """
    if action == 0:
        return 0.0

    label = loan["label"]

    if loan["loan_type"] == 0:
        gross = loan["gross_usd"]
        term  = loan["term_months"]
        if label == 0:
            return gross * SB_RATE * (term / 12)
        else:
            return -gross * LGD
    else:
        rate   = loan["rate_pct"] / 100.0 
        upb    = loan["upb_usd"]
        term   = loan["term_months"]
        if label == 0:
            return rate * upb * (term / 12)  # interest over loan life
        else:
            return -upb * LGD

def run_dqn(pool: list, policy_net: DQNNetwork,
            capital: float = CAPITAL_LIMIT) -> tuple[list, list]:
    """
    Score every loan in the pool based on the DQN policy evaluation on that loan, then
    rank the take the best loans based on that score within the pool
    """
    policy_net.eval()

    # Score every loan in the pool
    advantages = []
    with torch.no_grad():
        for loan in pool:
            s = loan["state"]
            feat_t = torch.tensor(s[:SB_FEATURE_DIM], dtype=torch.float32).unsqueeze(0)
            pd_t   = torch.tensor([s[-2]],             dtype=torch.float32)
            lt_t   = torch.tensor([int(s[-1])],        dtype=torch.long)
            q_vals = policy_net(feat_t, pd_t, lt_t)
            advantages.append((q_vals[0, 1] - q_vals[0, 0]).item())

    # Rank by advantage descending and fill greedily within capital
    ranked = sorted(range(len(pool)), key=lambda i: advantages[i], reverse=True)
    approved = set()
    remaining = capital
    for i in ranked:
        if advantages[i] <= 0:
            break
        amt = get_loan_amount(pool[i])
        if amt <= remaining:
            approved.add(i)
            remaining -= amt

    actions = [1 if i in approved else 0 for i in range(len(pool))]
    profits = [realized_profit(loan, actions[i]) for i, loan in enumerate(pool)]
    return actions, profits


def run_random(pool: list, capital: float = CAPITAL_LIMIT) -> tuple[list, list]:
    """
    Take loans at random until capital runs out
    """
    actions, profits = [], []
    remaining = capital
    for loan in pool:
        action = random.randint(0, 1)
        if action == 1:
            amt = get_loan_amount(loan)
            if amt > remaining:
                action = 0
            else:
                remaining -= amt
        actions.append(action)
        profits.append(realized_profit(loan, action))
    return actions, profits


def run_greedy(pool: list, capital: float = CAPITAL_LIMIT) -> tuple[list, list]:
    """
    Greedily approve the loans sorted by default probability 
    """
    sorted_idx = sorted(range(len(pool)), key=lambda i: pool[i]["p_default"])
    approved = set()
    remaining = capital
    for i in sorted_idx:
        loan = pool[i]
        amt = get_loan_amount(loan)
        if amt <= remaining:
            approved.add(i)
            remaining -= amt
    actions = [1 if i in approved else 0 for i in range(len(pool))]
    profits = [realized_profit(loan, actions[i]) for i, loan in enumerate(pool)]
    return actions, profits

def run_trials(sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw, 
               sb_model, hl_model, policy_net, n_trials: int = N_TRIALS):
    """
    Run evaluations for each agent
    Returns dictionaries where keys are the strategy name, and values are the differnt kinds of results      
    results = total_profit_per_trial
    approvals = percentage of loans approved
    default_rates = percent of loans apporved that defaulted
    cum_t0 = array of profit over time
    p_defaults_t0 = default probability of approved loans
    """
    results          = {s: [] for s in STRATEGIES}
    approvals        = {s: [] for s in STRATEGIES}
    default_rates    = {s: [] for s in STRATEGIES}
    capital_deployed = {s: [] for s in STRATEGIES}
    # loan-type breakdown: counts of approved SB and HL loans per trial
    sb_approved_counts = {s: [] for s in STRATEGIES}
    hl_approved_counts = {s: [] for s in STRATEGIES}
    # profit split by loan type per trial
    sb_profits = {s: [] for s in STRATEGIES}
    hl_profits = {s: [] for s in STRATEGIES}
    cum_t0        = {}
    p_defaults_t0 = {}

    runners = {
        "DQN":    lambda p: run_dqn(p, policy_net),
        "Random": run_random,
        "Greedy": run_greedy,
    }

    for t in range(n_trials):
        pool = build_loan_pool(
            sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw, sb_model, hl_model
        )

        for s in STRATEGIES:
            acts, profs = runners[s](pool)

            results[s].append(sum(profs))

            n_approved = sum(acts)
            approvals[s].append(n_approved / len(pool) * 100)

            approved_labels = [pool[i]["label"] for i, a in enumerate(acts) if a == 1]
            default_rates[s].append(
                np.mean(approved_labels) * 100 if approved_labels else 0.0
            )
            capital_deployed[s].append(
                sum(get_loan_amount(pool[i]) for i, a in enumerate(acts) if a == 1)
            )

            # loan-type breakdown
            sb_approved_counts[s].append(
                sum(1 for i, a in enumerate(acts) if a == 1 and pool[i]["loan_type"] == 0)
            )
            hl_approved_counts[s].append(
                sum(1 for i, a in enumerate(acts) if a == 1 and pool[i]["loan_type"] == 1)
            )
            sb_profits[s].append(
                sum(profs[i] for i, a in enumerate(acts) if pool[i]["loan_type"] == 0)
            )
            hl_profits[s].append(
                sum(profs[i] for i, a in enumerate(acts) if pool[i]["loan_type"] == 1)
            )

            # collect data on first iteration for plots
            if t == 0:
                cum_t0[s] = np.cumsum(profs)
                p_defaults_t0[s] = [pool[i]["p_default"] for i, a in enumerate(acts) if a == 1]

        if (t + 1) % 5 == 0:
            print(f"  Completed trial {t + 1}/{n_trials}")

    return (results, approvals, default_rates, capital_deployed,
            sb_approved_counts, hl_approved_counts, sb_profits, hl_profits,
            cum_t0, p_defaults_t0)


def print_summary(results, approvals, default_rates, capital_deployed):
    print(f"Capital limit: ${CAPITAL_LIMIT}")
    print(f"{'Strategy':<10} {'Mean Profit ($)':>15} {'Std':>12} {'Approval%':>11} {'Default%':>10} {'Capital Used ($)':>17}")
    print("-" * 50)
    for s in STRATEGIES:
        print(
            f"{s:<10}"
            f"{np.mean(results[s]):>15,.0f}"
            f"{np.std(results[s]):>12,.0f}"
            f"{np.mean(approvals[s]):>11.1f}"
            f"{np.mean(default_rates[s]):>10.1f}"
            f"{np.mean(capital_deployed[s]):>17,.0f}"
        )

def plot_results(results, approvals, default_rates, capital_deployed,
                 sb_approved_counts, hl_approved_counts, sb_profits, hl_profits,
                 cum_t0, p_defaults_t0, out_dir: Path):
    subtitle = (
        f"{N_TRIALS} trials × {N_LOANS} loans, capital limit ${CAPITAL_LIMIT/1e6:.0f}M"
    )

    # Mean total profit bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(results[s]) for s in STRATEGIES]
    stds  = [np.std(results[s])  for s in STRATEGIES]
    bar_colors = [COLORS[s] for s in STRATEGIES]
    bars = ax.bar(STRATEGIES, means, yerr=stds, capsize=6,
                  color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Mean Total Profit per Trial (± 1 std)\n{subtitle}")
    ax.set_ylabel("Profit (USD)")
    ax.set_xlabel("Strategy")
    y_range = max(abs(v) for v in means) or 1.0
    for bar, mean in zip(bars, means):
        va = "bottom" if mean >= 0 else "top"
        offset = y_range * 0.02 * (1 if mean >= 0 else -1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + offset,
            f"{mean:,.0f}",
            ha="center", va=va, fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    p = out_dir / "profit_bar.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

    # Profit over time on first trial
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in STRATEGIES:
        ax.plot(cum_t0[s], label=s, color=COLORS[s], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Cumulative Profit — Single Trial\n{subtitle}")
    ax.set_xlabel("Loan Number")
    ax.set_ylabel("Cumulative Profit (USD)")
    ax.legend()
    plt.tight_layout()
    p = out_dir / "cumulative_profit.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

    # Approval rate vs default rate among approved
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.35
    appr_means = [np.mean(approvals[s])     for s in STRATEGIES]
    def_means  = [np.mean(default_rates[s]) for s in STRATEGIES]
    appr_stds  = [np.std(approvals[s])      for s in STRATEGIES]
    def_stds   = [np.std(default_rates[s])  for s in STRATEGIES]

    ax.bar(x - width / 2, appr_means, width, yerr=appr_stds, capsize=4,
           color=[COLORS[s] for s in STRATEGIES], alpha=0.85,
           edgecolor="black", linewidth=0.8, label="Approval Rate (%)")
    ax.bar(x + width / 2, def_means, width, yerr=def_stds, capsize=4,
           color=[COLORS[s] for s in STRATEGIES], alpha=0.45,
           edgecolor="black", linewidth=0.8, hatch="//",
           label="Default Rate of Approved (%)")
    ax.set_title(f"Approval Rate vs. Default Rate of Approved Loans\n{subtitle}")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.legend()
    plt.tight_layout()
    p = out_dir / "approval_vs_default.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

    # default prob distribution of approved loans on singular trial
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 30)
    for s in STRATEGIES:
        vals = p_defaults_t0[s]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.5, color=COLORS[s],
                    label=f"{s} (n={len(vals)})", edgecolor="none", density=True)
    ax.set_title(f"p_default of Approved Loans — Representative Trial\n{subtitle}")
    ax.set_xlabel("Estimated Default Probability")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    p = out_dir / "pdefault_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

    # SB vs HL loan mix among approved (stacked bar)
    fig, ax = plt.subplots(figsize=(7, 5))
    sb_means = [np.mean(sb_approved_counts[s]) for s in STRATEGIES]
    hl_means = [np.mean(hl_approved_counts[s]) for s in STRATEGIES]
    sb_stds  = [np.std(sb_approved_counts[s])  for s in STRATEGIES]
    hl_stds  = [np.std(hl_approved_counts[s])  for s in STRATEGIES]
    x = np.arange(len(STRATEGIES))
    width = 0.35
    ax.bar(x - width / 2, sb_means, width, yerr=sb_stds, capsize=4,
           color="#FF8C00", alpha=0.85, edgecolor="black", linewidth=0.8, label="Small Business")
    ax.bar(x + width / 2, hl_means, width, yerr=hl_stds, capsize=4,
           color="#7B68EE", alpha=0.85, edgecolor="black", linewidth=0.8, label="Home Loan")
    ax.set_title(f"Approved Loan Mix: SB vs HL\n{subtitle}")
    ax.set_ylabel("Mean # Loans Approved per Trial")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.legend()
    plt.tight_layout()
    p = out_dir / "loan_type_mix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

    # Profit contribution by loan type per strategy
    fig, ax = plt.subplots(figsize=(7, 5))
    sb_profit_means = [np.mean(sb_profits[s]) for s in STRATEGIES]
    hl_profit_means = [np.mean(hl_profits[s]) for s in STRATEGIES]
    sb_profit_stds  = [np.std(sb_profits[s])  for s in STRATEGIES]
    hl_profit_stds  = [np.std(hl_profits[s])  for s in STRATEGIES]
    ax.bar(x - width / 2, sb_profit_means, width, yerr=sb_profit_stds, capsize=4,
           color="#FF8C00", alpha=0.85, edgecolor="black", linewidth=0.8, label="Small Business")
    ax.bar(x + width / 2, hl_profit_means, width, yerr=hl_profit_stds, capsize=4,
           color="#7B68EE", alpha=0.85, edgecolor="black", linewidth=0.8, label="Home Loan")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Mean Profit by Loan Type per Strategy\n{subtitle}")
    ax.set_ylabel("Mean Profit (USD)")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.legend()
    plt.tight_layout()
    p = out_dir / "profit_by_loan_type.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}")
    plt.close()

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    project_root = Path(__file__).parent.parent.parent
    models_dir   = project_root / "src" / "models"

    sb_model, hl_model = load_classifiers()

    policy_net = DQNNetwork()
    policy_net.load_state_dict(
        torch.load(models_dir / "rl_policy.pt", weights_only=True)
    )
    policy_net.eval()

    sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw = load_test_data(project_root)

    print(f"Running eval. {N_TRIALS} trials with {N_LOANS} loans each")
    (results, approvals, default_rates, capital_deployed,
     sb_approved_counts, hl_approved_counts, sb_profits, hl_profits,
     cum_t0, p_defaults_t0) = run_trials(
        sb_feats, sb_labels, sb_raw, hl_feats, hl_labels, hl_raw,
        sb_model, hl_model, policy_net,
    )

    print_summary(results, approvals, default_rates, capital_deployed)

    out_dir = project_root / "results"
    plot_results(results, approvals, default_rates, capital_deployed,
                 sb_approved_counts, hl_approved_counts, sb_profits, hl_profits,
                 cum_t0, p_defaults_t0, out_dir)


if __name__ == "__main__":
    main()
