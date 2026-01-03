import numpy as np
from scipy.stats import norm

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    M: int,
    N: int,
    antithetic: bool = False,
    seed: int = 42
):
    """
    Simulate GBM paths using exact discretisation.
    Returns array of shape (N, M+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / M

    Z = rng.standard_normal((N, M))

    if antithetic:
        Z = np.vstack((Z, -Z))
        N = 2 * N

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)

    log_paths = np.hstack((np.zeros((N, 1)), log_paths))
    S_paths = S0 * np.exp(log_paths)

    return S_paths

def arithmetic_asian_call(S_paths, K):
    """
    Arithmetic Asian call payoff.
    """
    A = np.mean(S_paths[:, 1:], axis=1)
    return np.maximum(A - K, 0.0)


def geometric_asian_call(S_paths, K):
    """
    Geometric Asian call payoff.
    """
    G = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
    return np.maximum(G - K, 0.0)

def control_variate_estimator(
    H: np.ndarray,
    Y: np.ndarray,
    EY: float
):
    """
    Apply control variates to payoff H using control Y with known expectation EY.
    """
    cov = np.cov(H, Y, ddof=1)[0, 1]
    varY = np.var(Y, ddof=1)

    beta = cov / varY
    H_cv = H + beta * (EY - Y)

    return H_cv, beta

def geometric_asian_call_price(S0, K, r, sigma, T, M):
    """
    Closed-form price of geometric Asian call under Black-Scholes.
    """
    sigma_hat = sigma * np.sqrt((2*M + 1) / (6*(M + 1)))
    mu_hat = 0.5 * sigma_hat**2 + (r - 0.5 * sigma**2) * (M + 1) / (2*M)

    d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)

    price = np.exp(-r*T) * (
        S0 * np.exp(mu_hat*T) * norm.cdf(d1) - K * norm.cdf(d2)
    )

    return price

# Parameters
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
M = 50
N = 100_000

# --- Baseline Monte Carlo ---
paths = simulate_gbm_paths(S0, r, sigma, T, M, N)
H = arithmetic_asian_call(paths, K)

price_mc = np.exp(-r*T) * np.mean(H)
stderr_mc = np.exp(-r*T) * np.std(H, ddof=1) / np.sqrt(N)

# --- Antithetic Variates ---
paths_anti = simulate_gbm_paths(S0, r, sigma, T, M, N // 2, antithetic=True)
H_anti = arithmetic_asian_call(paths_anti, K)

price_anti = np.exp(-r*T) * np.mean(H_anti)
stderr_anti = np.exp(-r*T) * np.std(H_anti, ddof=1) / np.sqrt(len(H_anti))

# --- Control Variates ---
Y = geometric_asian_call(paths, K)
EY = geometric_asian_call_price(S0, K, r, sigma, T, M)

H_cv, beta = control_variate_estimator(H, Y, EY)

price_cv = np.exp(-r*T) * np.mean(H_cv)
stderr_cv = np.exp(-r*T) * np.std(H_cv, ddof=1) / np.sqrt(N)

# --- Combined ---
Y_anti = geometric_asian_call(paths_anti, K)
H_cv_anti, _ = control_variate_estimator(H_anti, Y_anti, EY)

price_combined = np.exp(-r*T) * np.mean(H_cv_anti)
stderr_combined = np.exp(-r*T) * np.std(H_cv_anti, ddof=1) / np.sqrt(len(H_cv_anti))

# --- Output ---
print(f"Baseline MC:        {price_mc:.4f} ± {1.96*stderr_mc:.4f}")
print(f"Antithetic:         {price_anti:.4f} ± {1.96*stderr_anti:.4f}")
print(f"Control variate:    {price_cv:.4f} ± {1.96*stderr_cv:.4f}")
print(f"Combined estimator: {price_combined:.4f} ± {1.96*stderr_combined:.4f}")


