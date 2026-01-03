import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb
from scipy.stats import norm
from BlackScholes import BS_call

def simulate_gbm_terminal(S0, T, r, sig, n_paths, seed=None):
    """
    Simulate terminal values of a GBM under the risk neutral measure using the exact solution.

    :param n_paths: number of Monte Carlo paths

    :param seed: random seed for reproducibility
    """

    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_paths)

    ST = S0 * np.exp(
        (r - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z
    )

    return ST

# Example values
S0, r, sig, T = 100, 0.05, 0.2, 1.0
ST = simulate_gbm_terminal(S0, T, r, sig, 1_000_000, seed=21)

# Mean simulation
sim_mean = np.mean(ST)
theory_mean = S0 * np.exp(r * T)
mean_rel_error = np.abs(sim_mean - theory_mean) / theory_mean

print("Simulated mean:", sim_mean)
print("Theoretical mean:", theory_mean)
print("Relative error:", mean_rel_error)

# Variance simulation
theoretical_var = S0**2 * np.exp(2 * r * T) * (np.exp(sig**2 * T) - 1)
var_rel_error = np.abs(np.var(ST) - theoretical_var) / theoretical_var

print("Simulated variance:", np.var(ST))
print("Theoretical variance:", theoretical_var)
print("Relative error:", var_rel_error)

# Checking for lognormal distribution
plt.hist(np.log(ST), bins=100, density=True)
plt.title("Histogram of log(S_T)")
plt.xlabel("log(S_T)")
plt.ylabel("Density")
plt.show()

def european_MC(S0, K, T, r, sig, n_paths, seed=None):
    """
    Estimate the price of a European call option using Monte Carlo
    
    :param n_paths: number of Monte Carlo paths

    :param seed: random seed for reproducibility
    """

    ST = simulate_gbm_terminal(S0, T, r, sig, n_paths, seed=seed)
    payoff = np.maximum(ST - K, 0)
    discounted_payoff = np.exp(-r*T) * payoff

    price = np.mean(discounted_payoff)

    # Calculting standard error using delta degrees of freedom = 1 
    # since we need to adjust 1/N to 1/(N-1) to estimate variance
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(n_paths)

    return price, std_error
    
# Computing confidence interval for some example values
sim_european_price = european_MC(S0, 120, T, r, sig, 1_000_000, seed=21)
CI = [sim_european_price[0] - 1.96*sim_european_price[1], sim_european_price[0] + 1.96*sim_european_price[1]]

print("Simulated price:", sim_european_price)
print("Confidence interval:", CI)

# Comparing to Black Scholes
print("Black Scholes price:", BS_call(S0, 120, T, r, sig))
print("Relative Error:", np.abs(sim_european_price[0] - BS_call(S0, 120, T, r, sig))/BS_call(S0, 120, T, r, sig))


# Convergence Analysis
for i in [1000,5000,10000,50000]:
    print(f"MC price for {i} paths:", european_MC(S0, 120, T, r, sig, i, seed=21)[0])
    print(f"MC standard error for {i} paths:", european_MC(S0, 120, T, r, sig, i, seed=21)[1])

K = 120

# Convergence rate
N_values = np.array([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1000000])

errors = []
std_errors = []

BS_call_rate_example = BS_call(S0, 120, T, r, sig)

for N in N_values:
    price, std_error = european_MC(
        S0, K, T, r, sig, n_paths=N
    )
    errors.append(abs(price - BS_call_rate_example))
    std_errors.append(std_error)

errors = np.array(errors)
std_errors = np.array(std_errors)

plt.figure(figsize=(7, 5))
plt.loglog(N_values, errors, 'o-', label="MC error")

# Reference slope N^{-1/2}
ref_line = errors[0] * (N_values / N_values[0]) ** (-0.5)
plt.loglog(N_values, ref_line, '--', label=r"$N^{-1/2}$")

plt.xlabel("Number of Monte Carlo paths (N)")
plt.ylabel("Absolute error |MC − BS|")
plt.title("Monte Carlo Convergence Rate")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Euler Maruyama

def simulate_gbm_euler(
    S0, T, r, sig, n_steps, n_paths, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    S = np.full(n_paths, S0, dtype=np.float64)

    for _ in range(n_steps):
        Z = np.random.normal(size=n_paths)
        S += r * S * dt + sig * S * np.sqrt(dt) * Z

    return S

def price_european_call_from_terminal(ST, K, T, r):
    payoff = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(ST))
    return price, std_error


# Running time discretisation

n_paths = 200_000
n_steps_list = [1, 2, 5, 10, 20, 50, 100]

prices = []
errors = []

for n_steps in n_steps_list:
    ST = simulate_gbm_euler(
        S0, T, r, sig,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=21
    )
    price, _ = price_european_call_from_terminal(ST, K, T, r)
    prices.append(price)

# Exact benchmark
ST_exact = simulate_gbm_terminal(S0, T, r, sig, n_paths, seed=21)
price_exact, _ = price_european_call_from_terminal(ST_exact, K, T, r)


# Price vs time steps

plt.figure()
plt.plot(n_steps_list, prices, marker='o', label="Euler–Maruyama")
plt.axhline(price_exact, linestyle='--', label="Exact MC")
plt.xlabel("Number of time steps")
plt.ylabel("Option price")
plt.legend()
plt.show()

# Weak convergence plot

dt = np.array([T / n for n in n_steps_list])
errors = np.abs(np.array(prices) - price_exact)

plt.figure()
plt.loglog(dt, errors, marker='o')
plt.xlabel(r"$\Delta t$")
plt.ylabel("Absolute pricing error")
plt.title("Weak convergence of Euler–Maruyama")
plt.show()