import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------------------
# Download intraday data
# ---------------------------
symbol = 'AAPL'
data = yf.Ticker(symbol).history(period='5d', interval='1m')
data = data.between_time('09:30', '16:00')
data = data.dropna(subset=['Close', 'Volume'])
T = len(data)

# ---------------------------
# Market parameters
# ---------------------------
average_volume = np.mean(data['Volume'])
average_spread = np.mean(data['High'] - data['Low'])
sigma = np.std(np.log(data['Close']).diff().dropna())  # Realized volatility
eta = average_spread / (0.01 * average_volume)
gamma = average_spread / (0.1 * average_volume)
epsilon = average_spread / 2
tau = 1  # 1 minute per step
lambda_ = 1e-4  # Risk aversion

params = {
    'lambda': lambda_,
    'sigma': sigma,
    'epsilon': epsilon,
    'eta': eta,
    'gamma': gamma,
    'tau': tau
}

# ---------------------------
# Almgren–Chriss model
# ---------------------------
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

# ---------------------------
# Trading schedules
# ---------------------------
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

model = AlmgrenChriss1D(params)
ac_trades = model.strategy(X, T)

# ---------------------------
# Performance metrics
# ---------------------------
def execution_cost(trades, prices):
    return np.sum(trades * prices)

ac_cost = execution_cost(ac_trades, prices)
vwap_cost = execution_cost(vwap_trades, prices)
twap_cost = execution_cost(twap_trades, prices)

# Slippage (relative to VWAP)
ac_slippage = ac_cost - vwap_cost
twap_slippage = twap_cost - vwap_cost
vwap_slippage = 0

# Shortfall (relative to initial price benchmark)
initial_price = prices[0]
ac_shortfall = ac_cost - X * initial_price
vwap_shortfall = vwap_cost - X * initial_price
twap_shortfall = twap_cost - X * initial_price

# PnL (difference from avg price execution)
avg_price = np.mean(prices)
ac_pnl = np.sum(ac_trades * prices) - X * avg_price
vwap_pnl = np.sum(vwap_trades * prices) - X * avg_price
twap_pnl = np.sum(twap_trades * prices) - X * avg_price

metrics = pd.DataFrame({
    'Strategy': ['Almgren–Chriss', 'VWAP', 'TWAP'],
    'Cost': [ac_cost, vwap_cost, twap_cost],
    'Slippage': [ac_slippage, vwap_slippage, twap_slippage],
    'Shortfall': [ac_shortfall, vwap_shortfall, twap_shortfall],
    'PnL': [ac_pnl, vwap_pnl, twap_pnl]
})
metrics.set_index('Strategy', inplace=True)
print(metrics)

# ---------------------------
# Plot: Trading trajectories
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(np.cumsum(ac_trades), label='Almgren–Chriss')
plt.plot(np.cumsum(vwap_trades), label='VWAP')
plt.plot(np.cumsum(twap_trades), label='TWAP')
plt.xlabel('Time (minutes)')
plt.ylabel('Cumulative Shares Sold')
plt.title('Trading Trajectories (Intraday)')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Plot: Cost, Slippage, PnL
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12,6))

# Cost on primary axis
metrics['Cost'].plot(kind='bar', ax=ax1, color='tab:blue', position=0, width=0.4)
ax1.set_ylabel('Cost (USD)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Slippage and PnL on secondary axis
ax2 = ax1.twinx()
metrics[['Slippage','PnL']].plot(kind='bar', ax=ax2, color=['tab:orange','tab:green'], position=1, width=0.4)
ax2.set_ylabel('USD', color='black')
ax2.tick_params(axis='y', labelcolor='black')

plt.title('Trading Strategy Performance: Cost vs Slippage & PnL')
plt.grid(True)
plt.show()
