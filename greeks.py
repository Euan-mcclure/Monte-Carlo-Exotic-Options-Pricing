import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

def simulate_gbm(S0, r, sigma, T, N, M):
    """
    Simulate N paths of Geometric Brownian Motion with M time steps.
    
    Parameters:
        S0 : float - initial stock price
        r : float - risk-free rate
        sigma : float - volatility
        T : float - time to maturity
        N : int - number of paths
        M : int - number of time steps
    
    Returns:
        S : ndarray - shape (N, M+1) GBM paths
    """
    dt = T / M
    S = np.zeros((N, M+1))
    S[:,0] = S0
    Z = np.random.normal(size=(N, M))
    
    for t in range(1, M+1):
        S[:,t] = S[:,t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t-1])
    return S

def asian_call_payoff(S, K):
    """
    Arithmetic Asian call option payoff
    """
    S_avg = np.mean(S[:,1:], axis=1)  # exclude S0
    return np.maximum(S_avg - K, 0)

def barrier_call_payoff(S, K, H):
    """
    Up-and-out call option
    S : ndarray - simulated paths
    K : strike
    H : barrier level
    """
    knocked_out = (S[:,1:] >= H).any(axis=1)
    payoff = np.where(knocked_out, 0, np.maximum(S[:,-1] - K, 0))
    return payoff

def monte_carlo_price(S0, K, r, sigma, T, N, M, payoff_func, **kwargs):
    """
    General Monte Carlo pricing
    """
    S = simulate_gbm(S0, r, sigma, T, N, M)
    payoff = payoff_func(S, K, **kwargs)
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.exp(-r*T) * np.std(payoff)/np.sqrt(N)
    return price, stderr

def delta_fd(S0, K, r, sigma, T, N, M, payoff_func, h=0.01, **kwargs):
    price_up, _ = monte_carlo_price(S0 + h, K, r, sigma, T, N, M, payoff_func, **kwargs)
    price_down, _ = monte_carlo_price(S0 - h, K, r, sigma, T, N, M, payoff_func, **kwargs)
    delta = (price_up - price_down) / (2*h)
    return delta

def vega_fd(S0, K, r, sigma, T, N, M, payoff_func, h=0.01, **kwargs):
    price_up, _ = monte_carlo_price(S0, K, r, sigma + h, T, N, M, payoff_func, **kwargs)
    price_down, _ = monte_carlo_price(S0, K, r, sigma - h, T, N, M, payoff_func, **kwargs)
    vega = (price_up - price_down) / (2*h)
    return vega

# Parameters
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
N = 100_000
M = 50
H = 120  # Barrier

# Price Asian call
price_asian, err_asian = monte_carlo_price(S0, K, r, sigma, T, N, M, asian_call_payoff)
delta_asian = delta_fd(S0, K, r, sigma, T, N, M, asian_call_payoff)
vega_asian = vega_fd(S0, K, r, sigma, T, N, M, asian_call_payoff)

print(f"Asian Call: Price={price_asian:.4f}, Delta={delta_asian:.4f}, Vega={vega_asian:.4f}")

# Price Barrier call
price_barrier, err_barrier = monte_carlo_price(S0, K, r, sigma, T, N, M, barrier_call_payoff, H=H)
delta_barrier = delta_fd(S0, K, r, sigma, T, N, M, barrier_call_payoff, H=H)
vega_barrier = vega_fd(S0, K, r, sigma, T, N, M, barrier_call_payoff, H=H)

print(f"Barrier Call: Price={price_barrier:.4f}, Delta={delta_barrier:.4f}, Vega={vega_barrier:.4f}")

S0_values = np.linspace(80, 120, 9)  # 80,85,...,120
delta_asian_vals = []
delta_barrier_vals = []

# Compute Delta for each S0
for S0 in S0_values:
    delta_asian_vals.append(delta_fd(S0, K, r, sigma, T, N, M, asian_call_payoff))
    delta_barrier_vals.append(delta_fd(S0, K, r, sigma, T, N, M, barrier_call_payoff, H=H))

# Delta plot
plt.figure(figsize=(8,5))
plt.plot(S0_values, delta_asian_vals, marker='o', label='Asian Call')
plt.plot(S0_values, delta_barrier_vals, marker='s', label='Barrier Call')
plt.xlabel('Underlying Price $S_0$')
plt.ylabel('Delta')
plt.title('Delta vs Underlying Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_vs_S0.png", dpi=300)
plt.show()

sigma_values = np.linspace(0.1, 0.4, 7)  # 0.1 to 0.4
vega_asian_vals = []
vega_barrier_vals = []

S0_fixed = 100

for sigma_i in sigma_values:
    vega_asian_vals.append(vega_fd(S0_fixed, K, r, sigma_i, T, N, M, asian_call_payoff))
    vega_barrier_vals.append(vega_fd(S0_fixed, K, r, sigma_i, T, N, M, barrier_call_payoff, H=H))

# Vega plot
plt.figure(figsize=(8,5))
plt.plot(sigma_values, vega_asian_vals, marker='o', label='Asian Call')
plt.plot(sigma_values, vega_barrier_vals, marker='s', label='Barrier Call')
plt.xlabel('Volatility $\sigma$')
plt.ylabel('Vega')
plt.title('Vega vs Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vega_vs_sigma.png", dpi=300)
plt.show()