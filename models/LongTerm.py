import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Download Data
symbol = 'AAPL'
data = yf.Ticker(symbol).history(period='3mo')
data = data.dropna(subset=['Close', 'Volume'])
T = len(data)

# 2. Parameter Calibration
average_daily_volume = np.mean(data['Volume'])
average_daily_spread = np.mean(data['High'] - data['Low'])
sigma = np.std(np.log(data['Close']).diff().dropna())  # Realized volatility
eta = average_daily_spread / (0.01 * average_daily_volume)
gamma = average_daily_spread / (0.1 * average_daily_volume)
epsilon = average_daily_spread / 2
tau = 1
lambda_ = 1e-8  # Risk aversion

params = {
    'lambda': lambda_,
    'sigma': sigma,
    'epsilon': epsilon,
    'eta': eta,
    'gamma': gamma,
    'tau': tau
}

# 3. Almgren–Chriss Model
class AlmgrenChriss1D:
    def __init__(self, params):
        self._lambda = params['lambda']
        self._sigma = params['sigma']
        self._epsilon = params['epsilon']
        self._eta = params['eta']
        self._gamma = params['gamma']
        self._tau = params['tau']
        self._eta_tilda = self._eta - 0.5*self._gamma*self._tau
        assert self._eta_tilda > 0
        self._kappa_tilda_squared = (self._lambda*self._sigma**2)/self._eta_tilda
        self._kappa = np.arccosh(0.5*(self._kappa_tilda_squared*self._tau**2) + 1)/self._tau
    def trajectory(self, X, T):
        ans = []
        for t in range(T):
            x = np.sinh(self._kappa*(T - t))/np.sinh(self._kappa*T)*X
            ans.append(x)
        ans.append(0)
        return np.array(ans)
    def strategy(self, X, T):
        return -np.diff(self.trajectory(X,T))

# 4. VWAP and TWAP Schedules
X = 250000  # Total shares to sell
volumes = data['Volume'].values
prices = data['Close'].values

def twap_schedule(total_shares, num_periods):
    base = total_shares // num_periods
    schedule = [base] * num_periods
    for i in range(total_shares % num_periods):
        schedule[i] += 1
    return np.array(schedule)

def vwap_schedule(total_shares, volumes):
    weights = volumes / volumes.sum()
    schedule = (weights * total_shares).astype(int)
    diff = total_shares - schedule.sum()
    for i in range(abs(diff)):
        idx = i % len(volumes)
        schedule[idx] += np.sign(diff)
    return schedule

twap_trades = twap_schedule(X, T)
vwap_trades = vwap_schedule(X, volumes)

# 5. Almgren–Chriss Optimal Schedule
model = AlmgrenChriss1D(params)
ac_trades = model.strategy(X, T)

# 6. Calculate Execution Costs

def execution_cost(trades, prices):
    return np.sum(trades * prices)

ac_cost = execution_cost(ac_trades, prices)
vwap_cost = execution_cost(vwap_trades, prices)
twap_cost = execution_cost(twap_trades, prices)

print(f"Almgren–Chriss cost: {ac_cost:.2f}")
print(f"VWAP cost: {vwap_cost:.2f}")
print(f"TWAP cost: {twap_cost:.2f}")

# 7. Plot Comparison
plt.figure(figsize=(10,6))
plt.bar(['Almgren–Chriss', 'VWAP', 'TWAP'], [ac_cost, vwap_cost, twap_cost], color=['tab:blue', 'tab:orange', 'tab:green'])
plt.ylabel('Total Cost (USD)')
plt.title('Strategy Comparison: Almgren–Chriss vs VWAP vs TWAP')
plt.show()

# Optional: Plot trading trajectories
plt.figure(figsize=(12,6))
plt.plot(np.cumsum(ac_trades), label='Almgren–Chriss')
plt.plot(np.cumsum(vwap_trades), label='VWAP')
plt.plot(np.cumsum(twap_trades), label='TWAP')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Shares Sold')
plt.title('Trading Trajectories')
plt.legend()
plt.show()
