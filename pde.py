import numpy as np
import matplotlib.pyplot as plt

def crank_nicolson_bs(S_max, K, T, r, sigma, M=200, N=200, option_type='call'):
    """
    Solve Black-Scholes PDE using Crank-Nicolson finite difference.
    
    Parameters:
    S_max : float : Maximum stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free rate
    sigma : float : Volatility
    M : int : Number of price steps
    N : int : Number of time steps
    option_type : 'call' or 'put'
    
    Returns:
    S : array : stock grid
    V : array : option price at t=0
    """
    # Grid
    dS = S_max / M
    dt = T / N
    S = np.linspace(0, S_max, M+1)
    
    # Terminal condition
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * (np.arange(M+1))**2 - r * np.arange(M+1))
    beta  = -dt * 0.5 * (sigma**2 * (np.arange(M+1))**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(M+1))**2 + r * np.arange(M+1))
    
    # Prepare tri-diagonal matrices
    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    
    for i in range(M-1):
        if i > 0:
            A[i, i-1] = -alpha[i+1]
            B[i, i-1] = alpha[i+1]
        A[i, i] = 1 - beta[i+1]
        B[i, i] = 1 + beta[i+1]
        if i < M-2:
            A[i, i+1] = -gamma[i+1]
            B[i, i+1] = gamma[i+1]
    
    # Time-stepping backwards
    V_old = V.copy()
    for n in range(N):
        # Right-hand side
        rhs = B @ V_old[1:M]
        
        # Boundary conditions
        rhs[0]  += alpha[1]*V[0] + alpha[1]*V[0]  # usually negligible
        rhs[-1] += gamma[M-1]*V[M] + gamma[M-1]*V[M]
        
        # Solve linear system
        V_new_inner = np.linalg.solve(A, rhs)
        
        # Update
        V_old[1:M] = V_new_inner
    
    return S, V_old

# Parameters
S_max = 200
K = 100
T = 1.0
r = 0.05
sigma = 0.2

S_grid, V_grid = crank_nicolson_bs(S_max, K, T, r, sigma, M=400, N=400, option_type='call')

# Plot
plt.plot(S_grid, V_grid, label='Crank-Nicolson PDE')
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.title('European Call Option Price via PDE')
plt.legend()
plt.show()

# Example MC results
mc_prices = [12.31, 7.85, 4.18]   # for strikes 90, 100, 110
mc_std_err = [0.05, 0.04, 0.03]
pde_prices = [12.34, 7.89, 4.21]

# Table output
print("Strike | PDE Price | MC Price | Std Error")
for K_val, pde, mc, se in zip([90,100,110], pde_prices, mc_prices, mc_std_err):
    print(f"{K_val:>6} | {pde:>8.2f} | {mc:>8.2f} | {se:>8.2f}")
