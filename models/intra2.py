import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- Download 1-minute data for last 5 days ---
symbol = 'AAPL'
data = yf.Ticker(symbol).history(period='5d', interval='1m')
data = data.between_time('09:30', '16:00')
data = data.dropna(subset=['Close', 'Volume'])
T = len(data)

# --- Market parameters ---
average_volume = np.mean(data['Volume'])
average_spread = np.mean(data['High'] - data['Low'])
sigma = np.std(np.log(data['Close']).diff().dropna())  # Realized volatility
eta = average_spread / (0.01 * average_volume)
gamma = average_spread / (0.1 * average_volume)
epsilon = average_spread / 2
tau = 1  # 1 minute per step
lambda_ = 1e-4  # Risk aversion

params = {'lambda': lambda_, 'sigma': sigma, 'epsilon': epsilon,
          'eta': eta, 'gamma': gamma, 'tau': tau}

# --- Almgren–Chriss 1D Model ---
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
        return np.array([np.sinh(self._kappa*(T - t))/np.sinh(self._kappa*T)*X for t in range(T)] + [0])

    def strategy(self, X, T):
        return -np.diff(self.trajectory(X,T))

# --- Trading setup ---
X = 250_000
volumes = data['Volume'].values
prices = data['Close'].values

# --- TWAP & VWAP ---
def twap_schedule(total_shares, num_periods):
    base = total_shares // num_periods
    schedule = [base]*num_periods
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

# --- Cost function ---
def execution_cost(trades, prices):
    return np.sum(trades * prices)

ac_cost = execution_cost(ac_trades, prices)
vwap_cost = execution_cost(vwap_trades, prices)
twap_cost = execution_cost(twap_trades, prices)

# --- Performance metrics ---
initial_price = prices[0]
avg_price = np.mean(prices)

metrics = pd.DataFrame({
    'Strategy': ['Almgren–Chriss', 'VWAP', 'TWAP'],
    'Cost': [ac_cost, vwap_cost, twap_cost],
    'Slippage_vs_VWAP': [ac_cost - vwap_cost, 0, twap_cost - vwap_cost],
    'Shortfall_vs_Initial': [ac_cost - X*initial_price, vwap_cost - X*initial_price, twap_cost - X*initial_price],
    'PnL_vs_Initial': [execution_cost(ac_trades, prices) - X*avg_price,
                       execution_cost(vwap_trades, prices) - X*avg_price,
                       execution_cost(twap_trades, prices) - X*avg_price]
})

print(metrics)

# --- Enhanced Plots ---
fig, ax1 = plt.subplots(figsize=(12,6))
width = 0.25
x = np.arange(len(metrics))

# Plot absolute costs
ax1.bar(x - width, metrics['Cost']/1e6, width, label='Cost (in millions USD)', color='tab:blue')
ax1.set_ylabel('Cost (millions USD)', color='tab:blue')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics['Strategy'])
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Secondary axis for slippage & PnL
ax2 = ax1.twinx()
ax2.bar(x, metrics['Slippage_vs_VWAP'], width, label='Slippage vs VWAP', color='tab:orange')
ax2.bar(x + width, metrics['PnL_vs_Initial'], width, label='PnL vs Avg Price', color='tab:green')
ax2.set_ylabel('USD', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.legend(loc='upper right', bbox_to_anchor=(0.85,0.85))
plt.title('Execution Strategy Performance: Cost, Slippage, and PnL')
plt.show()

# --- Cumulative shares sold ---
plt.figure(figsize=(12,6))
plt.plot(np.cumsum(ac_trades), label='Almgren–Chriss')
plt.plot(np.cumsum(vwap_trades), label='VWAP')
plt.plot(np.cumsum(twap_trades), label='TWAP')
plt.xlabel('Time (minutes)')
plt.ylabel('Cumulative Shares Sold')
plt.title('Trading Trajectories (Intraday)')
plt.legend()
plt.show()
