import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def arithmetic_asian_call_payoff(paths, strike):
    """
    paths: np.ndarray of shape (n_paths, n_steps + 1)
    strike: float
    """
    # Exclude S0
    average_price = paths[:, 1:].mean(axis=1)
    return np.maximum(average_price - strike, 0.0)

def geometric_asian_call_payoff(paths, strike):
    """
    paths: np.ndarray of shape (n_paths, n_steps + 1)
    strike: float
    """
    log_prices = np.log(paths[:, 1:])
    geo_average = np.exp(log_prices.mean(axis=1))
    return np.maximum(geo_average - strike, 0.0)

def geometric_asian_call_price_bs(
    S0, K, r, sig, T, n_steps
):
    """
    Closed-form price of discretely monitored geometric Asian call
    under Black–Scholes
    """
    dt = T / n_steps

    sigma_hat = sig * np.sqrt(
        (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2)
    )

    mu_hat = (
        0.5 * sigma_hat**2
        + (r - 0.5 * sig**2) * (n_steps + 1) / (2 * n_steps)
    )

    d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)

    price = np.exp(-r * T) * (
        S0 * np.exp(mu_hat * T) * norm.cdf(d1)
        - K * norm.cdf(d2)
    )

    return price

def monte_carlo_price(payoffs, r, T):
    """
    payoffs: np.ndarray
    """
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, std_error

def control_variate_price(
    target_payoffs,
    control_payoffs,
    control_price_exact,
    r,
    T
):
    """
    Implements optimal control variate estimator
    """
    X = np.exp(-r * T) * target_payoffs
    Y = np.exp(-r * T) * control_payoffs

    cov = np.cov(X, Y, ddof=1)[0, 1]
    var = np.var(Y, ddof=1)

    beta = cov / var

    adjusted = X + beta * (control_price_exact - Y)

    price = adjusted.mean()
    std_error = adjusted.std(ddof=1) / np.sqrt(len(adjusted))

    return price, std_error, beta

def simulate_gbm_paths(
    S0, r, sig, T, n_steps, n_paths, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps

    Z = np.random.normal(size=(n_paths, n_steps))

    increments = (
        (r - 0.5 * sig**2) * dt
        + sig * np.sqrt(dt) * Z
    )

    log_paths = np.cumsum(increments, axis=1)

    log_paths = np.hstack(
        [np.zeros((n_paths, 1)), log_paths]
    )

    paths = S0 * np.exp(log_paths)

    return paths

# Parameters
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
n_steps = 50
n_paths = 100_000

# Simulate paths
paths = simulate_gbm_paths(
    S0=S0,
    r=r,
    sig=sigma,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=42
)

# Payoffs
arith_payoff = arithmetic_asian_call_payoff(paths, K)
geo_payoff = geometric_asian_call_payoff(paths, K)

# Plain MC
plain_price, plain_se = monte_carlo_price(arith_payoff, r, T)

# Control variate
geo_exact = geometric_asian_call_price_bs(S0, K, r, sigma, T, n_steps)
cv_price, cv_se, beta = control_variate_price(
    arith_payoff, geo_payoff, geo_exact, r, T
)

print(f"Plain MC price: {plain_price:.4f} ± {1.96 * plain_se:.4f}")
print(f"CV MC price:    {cv_price:.4f} ± {1.96 * cv_se:.4f}")
print(f"Variance reduction factor: {(plain_se / cv_se)**2:.1f}x")
print(f"Optimal beta: {beta:.3f}")

# Convergence plot

path_counts = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1_000_000]

plain_prices = []
cv_prices = []

geo_exact = geometric_asian_call_price_bs(S0, K, r, sigma, T, n_steps)

for n_paths in path_counts:
    paths = simulate_gbm_paths(
        S0=S0,
        r=r,
        sig=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=1234
    )

    arith = arithmetic_asian_call_payoff(paths, K)
    geo = geometric_asian_call_payoff(paths, K)

    plain_price, _ = monte_carlo_price(arith, r, T)
    cv_price, _, _ = control_variate_price(arith, geo, geo_exact, r, T)

    plain_prices.append(plain_price)
    cv_prices.append(cv_price)

# Plot
plt.figure()
plt.plot(path_counts, plain_prices, marker='o', label='Plain Monte Carlo')
plt.plot(path_counts, cv_prices, marker='s', label='Control Variate MC')
plt.xscale('log')
plt.xlabel('Number of paths')
plt.ylabel('Option price')
plt.title('Asian Call Option Price Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Variance reduction plot

# Simulate once
paths = simulate_gbm_paths(
    S0=S0,
    r=r,
    sig=sigma,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=999
)

arith = arithmetic_asian_call_payoff(paths, K)
geo = geometric_asian_call_payoff(paths, K)

geo_exact = geometric_asian_call_price_bs(S0, K, r, sigma, T, n_steps)

# Get estimators
plain_price, plain_se = monte_carlo_price(arith, r, T)
cv_price, cv_se, _ = control_variate_price(arith, geo, geo_exact, r, T)

variances = [plain_se**2, cv_se**2]
labels = ['Plain MC', 'Control Variate MC']

# Plot
plt.figure()
plt.bar(labels, variances)
plt.ylabel('Estimator Variance')
plt.yscale('log')
plt.title('Variance Reduction via Control Variates')
plt.show()

print(f"Variance reduction factor: {variances[0] / variances[1]:.1f}x")

