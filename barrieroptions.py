import numpy as np
import matplotlib.pyplot as plt
from asianoptions import simulate_gbm_paths

def up_and_out_call_payoff(paths, K, B):
    max_path = paths.max(axis=1)
    terminal = paths[:, -1]
    payoff = np.where(
        max_path < B,
        np.maximum(terminal - K, 0.0),
        0.0
    )
    return payoff

def brownian_bridge_crossing_prob(x0, x1, logB, sigma, dt):
    num = -2.0 * (logB - x0) * (logB - x1)
    den = sigma**2 * dt
    return np.exp(num / den)

def up_and_out_call_bb(paths, K, B, sigma, T):
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    logB = np.log(B)

    alive = np.ones(paths.shape[0], dtype=bool)

    log_paths = np.log(paths)

    for i in range(n_steps):
        x0 = log_paths[:, i]
        x1 = log_paths[:, i + 1]

        crossed_discrete = (x0 >= logB) | (x1 >= logB)

        p_cross = brownian_bridge_crossing_prob(
            x0, x1, logB, sigma, dt
        )
        U = np.random.rand(len(p_cross))
        crossed_continuous = U < p_cross

        alive &= ~(crossed_discrete | crossed_continuous)

    terminal = paths[:, -1]
    payoff = np.where(
        alive,
        np.maximum(terminal - K, 0.0),
        0.0
    )
    return payoff

def monte_carlo_price(payoffs, r, T):
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, stderr


# Parameters
S0 = 100.0
K = 100.0
B = 130.0
r = 0.05
sigma = 0.2
T = 1.0
n_paths = 200_000
time_steps_list = [25, 50, 100, 250, 500]

naive_prices = []
bb_prices = []
naive_errs = []
bb_errs = []

for n_steps in time_steps_list:
    paths = simulate_gbm_paths(
        S0, r, sigma, T, n_steps, n_paths, seed=42
    )

    payoff_naive = up_and_out_call_payoff(paths, K, B)
    price_naive, err_naive = monte_carlo_price(payoff_naive, r, T)

    payoff_bb = up_and_out_call_bb(paths, K, B, sigma, T)
    price_bb, err_bb = monte_carlo_price(payoff_bb, r, T)

    naive_prices.append(price_naive)
    bb_prices.append(price_bb)
    naive_errs.append(err_naive)
    bb_errs.append(err_bb)

    print(
        f"Steps={n_steps:4d} | "
        f"Naive={price_naive:.4f} ± {err_naive:.4f} | "
        f"BB={price_bb:.4f} ± {err_bb:.4f}"
    )

plt.figure(figsize=(7, 5))
plt.plot(time_steps_list, naive_prices, marker='o', label='Naive MC')
plt.plot(time_steps_list, bb_prices, marker='s', label='Brownian Bridge MC')

plt.xlabel("Number of Time Steps")
plt.ylabel("Option Price")
plt.title("Barrier Option Monte Carlo Convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("barrier_convergence.pdf")
plt.show()
