import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.linalg import sqrtm

import yfinance as yf
goog = yf.Ticker('AAPL')
data = goog.history(period='3mo')
print(data)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(data.index, data['Close'], color='blue')
plt.title('GOOG Closing Price (Last 3 Months)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(data.index, data['Volume'], color='grey', width=1)
plt.title('GOOG Trading Volume (Last 3 Months)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()

average_daily_volume = np.mean(data['Volume'])
average_daily_spread = np.mean(data['High'] - data['Low'])

sigma = np.std(data['Close'])
epsilon = average_daily_spread/2
eta = average_daily_spread/(0.01*average_daily_volume)
gamma = average_daily_spread/(0.1*average_daily_volume)
tau = 1

params = {
    'lambda': 1e-08,
    'sigma': sigma,
    'epsilon': epsilon,
    'eta': eta,
    'gamma': gamma,
    'tau': tau
}

print('Parameters:')
for k,v in params.items():
    print('  {} = {}'.format(k,v))

class AlmgrenChriss1D:
    
    def __init__(self, params):
        # Initialize Parameters
        self._lambda = params['lambda']
        self._sigma = params['sigma']
        self._epsilon = params['epsilon']
        self._eta = params['eta']
        self._gamma = params['gamma']
        self._tau = params['tau']
        
        self._eta_tilda = self._eta - 0.5*self._gamma*self._tau
        
        # Ensure Quadratic (for optimization)
        assert self._eta_tilda > 0
        
        self._kappa_tilda_squared = (self._lambda*self._sigma**2)/self._eta_tilda
        
        self._kappa = np.arccosh(0.5*(self._kappa_tilda_squared*self._tau**2) + 1)/self._tau
        
    def trajectory(self, X, T):
        # Optimal Liquidation Trajectory
        ans = []
        for t in range(T):
            x = int(np.sinh(self._kappa*(T - t))/np.sinh(self._kappa*T)*X)
            ans.append(x)
        ans.append(0)
        return np.array(ans)
    
    def strategy(self, X, T):
        # Optimal Liquidation Trade List
        return np.diff(self.trajectory(X,T))

X = 250000
T = len(data)

model = AlmgrenChriss1D(params)
trajectory = model.trajectory(X,T)

plt.figure(figsize=(12,7))
plt.plot(range(T+1),trajectory,'o-')
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.show()


lambdas = [1e-2, 1e-7, 1e-8, 1e-9, 1e-11]

plt.figure(figsize=(12,7))
for _lambda in lambdas:
    new_params = params.copy()
    new_params['lambda'] = _lambda
    model = AlmgrenChriss1D(new_params)
    # Check for overflow risk
    if model._kappa * T > 700 or np.isnan(model._kappa):
        print(f"Skipping lambda={_lambda}: kappa*T too large or invalid.")
        continue
    trajectory = model.trajectory(X, T)
    if np.any(np.isnan(trajectory)):
        print(f"Skipping lambda={_lambda}: trajectory contains NaN.")
        continue
    plt.plot(range(T + 1), trajectory, 'o-', ms=4, label=f'$\lambda$ = {_lambda}')
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)', fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

gammas = [0.0001,0.002,0.005,0.008 , 0.08 , 0.8 , 8]

for gamma in gammas:
    new_params = params.copy()
    new_params['gamma'] = gamma
    eta_tilda = new_params['eta'] - 0.5 * gamma * new_params['tau']
    print(f"gamma={gamma}, eta_tilda={eta_tilda}")  # Debug print
    if eta_tilda <= 0:
        print(f"Skipping gamma={gamma}: eta_tilda <= 0 (invalid for model)")
        continue
    model = AlmgrenChriss1D(new_params)
    trajectory = model.trajectory(X, T)
    print(f"trajectory for gamma={gamma}: {trajectory}")  # Debug print
    plt.plot(range(T + 1), trajectory, 'o-', ms=4, label=f'$\\gamma$ = {gamma}')

etas = [0.001,0.01,0.1,1]

plt.figure(figsize=(12,7))
for eta in etas:
    new_params = params.copy()
    new_params['eta'] = eta
    model = AlmgrenChriss1D(new_params)
    trajectory = model.trajectory(X,T)
    plt.plot(range(T + 1),trajectory,'o-',ms=4, label=f'$\eta$ = {eta}')
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()
