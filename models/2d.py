import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.linalg import sqrtm

# Download data
goog = yf.Ticker('GOOG')
data1 = goog.history(period='3mo')
fb = yf.Ticker('META')  # Use 'META' for Meta Platforms, Inc. (formerly FB)
data2 = fb.history(period='3mo')

fig = plt.figure(figsize=(12,8))

# Main axis for FB
top_plt = plt.subplot2grid((5,4), (0, 0), rowspan=3, colspan=4)
top_plt.plot(data2.index, data2["Close"], color='tab:orange', label='META')
top_plt.set_ylabel('Price (Close)')
top_plt.set_xticks([])
top_plt.legend(loc='upper left')
top_plt.tick_params(axis='y', labelcolor='tab:orange')

# Twin axis for GOOG
top_plt2 = top_plt.twinx()
top_plt2.plot(data1.index, data1["Close"], color='tab:blue', label='GOOG')
top_plt2.set_ylabel('Price (Close)')
top_plt2.tick_params(axis='y', labelcolor='tab:blue')
top_plt2.legend(loc='upper right')

plt.title('Meta Platforms, Inc. (NASDAQ:META) & Alphabet Inc Class C (NASDAQ:GOOG)')
plt.tight_layout()
plt.show()

average_daily_volume1 = np.mean(data1['Volume'])
average_daily_volume2 = np.mean(data2['Volume'])
average_daily_spread1 = np.mean(data1['High'] - data1['Low'])
average_daily_spread2 = np.mean(data2['High'] - data2['Low'])

C = np.cov(data1['Close'], data2['Close'])

epsilon = (average_daily_spread1 + average_daily_spread2)/2


H = np.array([[average_daily_spread1/(0.01*average_daily_volume1), 0],
              [0, average_daily_spread2/(0.01*average_daily_volume2)]])

Gamma = np.array([[average_daily_spread1/(0.1*average_daily_volume1), 0],
                  [0, average_daily_spread2/(0.1*average_daily_volume2)]])

tau = 1

params = {
    'lambda': 1e-08,
    'C': C,
    'epsilon': epsilon,
    'H': H,
    'Gamma': Gamma,
    'tau': tau
}

print('Parameters:')
for k,v in params.items():
    print(f'{k} =\n {v}')

def decompose(A):
    # Return symmetric and anit-symmetric parts of a given matrix (A = A_s + A_a)
    A_s = 0.5 * (A + A.T)
    A_a = 0.5 * (A - A.T)
    return (A_s, A_a)

def is_diagonal(A):
    # Return True if matrix is diagonal
    i, j = np.nonzero(A)
    return np.all(i == j)

class AlmgrenChriss:
    def __init__(self, params):
        
        # Initialize Parameters
        self._lambda = params['lambda']
        self._C = params['C']
        self._epsilon = params['epsilon']
        self._H = params['H']
        self._Gamma = params['Gamma']
        self._tau = params['tau']
        
        # Dimensions
        self._dims = self._C.shape[0]
        
        # Decompose martrices
        self._H_s, self._H_a = decompose(self._H)
        self._Gamma_s, self._Gamma_a = decompose(self._Gamma)
        
        self._H_tilda = self._H_s - 0.5*self._Gamma_s*self._tau
        
        # Ensure positive-definite (for invertibility) 
        assert np.all(np.linalg.eigvals(self._H_tilda) > 0)
        
        self._A = np.linalg.inv(sqrtm(self._H_tilda))@self._C@np.linalg.inv(sqrtm(self._H_tilda))
        self._B = np.linalg.inv(sqrtm(self._H_tilda))@self._Gamma_a@np.linalg.inv(sqrtm(self._H_tilda))
        
        if is_diagonal(self._H_tilda):
            self._kappa_tilda_squareds, self._U = np.linalg.eig(self._lambda*self._A)            
            self._kappas = np.arccosh(0.5*(self._kappa_tilda_squareds*self._tau**2) + 1)/self._tau

    def trajectory(self, X, T, general=False):
        # Optimal Liquidation Trajectory
        traj = []
        if not general:
            # Diagonal Model
            if not is_diagonal(self._H_tilda): raise ValueError
            z0 = self._U.T@sqrtm(self._H_tilda)@X
            for t in range(T+1):
                z = np.sinh(self._kappas*(T - t))/np.sinh(self._kappas*T)*z0
                x = np.floor(np.linalg.inv(sqrtm(self._H_tilda))@self._U@z)
                traj.append(x)
        else:
            # General Model (only supported for 2 dimensions - did not have time to generalize)
            if self._dims != 2: raise ValueError
            # Transformation
            y0 = sqrtm(self._H_tilda)@X
            # Build Linear System
            rhs = np.zeros(2*(T+1))
            # Initial Conditions
            rhs[0] = y0[0]
            rhs[1] = y0[1]
            init1 = np.zeros(2*(T+1))
            init1[0] = 1
            init2 = np.zeros(2*(T+1))
            init2[1] = 1
            system = [init1,init2]
            # System
            for k in range (0,T-1):
                a = (1/self._tau**2)
                b = 1/(2*self._tau)
                c = -2/(self._tau**2)
                l = self._lambda
                A = self._A
                B = self._B
                equation1_coeff = [
                    a - b*B[0,0],
                    -b*B[0,1],
                    c - l*A[0,0],
                    -l*A[0,1],
                    a + b*B[0,0],
                    b*B[0,1] 
                ]
                equation2_coeff = [
                    -b*B[1,0],
                    a - b*B[1,1],
                    -l*A[1,0],
                    c - l*A[1,1],
                    b*B[1,0],
                    a + b*B[1,1] 
                ]
                system.append(np.array([0]*2*k + equation1_coeff + [0]*2*(T-k-2)))
                system.append(np.array([0]*2*k + equation2_coeff + [0]*2*(T-k-2)))
            # Final Conditions
            final1 = np.zeros(2*(T+1))
            final1[-2] = 1 
            system.append(final1) # y_{1,N} = 0
            final2 = np.zeros(2*(T+1))
            final2[-1] = 1
            system.append(final2) # y_{2,N} = 0
            # Solve Linear System
            sol = np.linalg.solve(np.array(system), rhs)
            y = sol.reshape(T+1,2)
            # Undo Transformation
            for yk in y:
                traj.append(np.linalg.inv(sqrtm(self._H_tilda))@yk)
        return np.array(traj).T

    
    def strategy(self, X, T):
        # Optimal Liquidation Trade List
        return np.diff(self.trajectory(X,T))
    
X = np.array([25000,25000]).T
T = len(data1)

model = AlmgrenChriss(params)
trajectory = model.trajectory(X,T, general=False)
assert np.any(trajectory == model.trajectory(X,T, general=True))

plt.figure(figsize=(12,7))
plt.plot(range(T+1),trajectory[0],'o-',ms=4,label='GOOG')
plt.plot(range(T+1),trajectory[1],'o-',ms=4,label='FB')
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend(loc=4)
plt.show()


lambdas = [(1e-10, 'red'),(1e-8, 'green'),(1e-4,'blue')]

plt.figure(figsize=(12,7))
for _lambda, color in lambdas:
    new_params = params.copy()
    new_params['lambda'] = _lambda
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG $\lambda$ = {_lambda}', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB $\lambda$ = {_lambda}', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

covs = [(0, 'red'),(100, 'green'),(1000,'blue')]

plt.figure(figsize=(12,7))
for cov, color in covs:
    new_params = params.copy()
    C = np.cov(data1['Close'], data2['Close'])
    C[1][0] = cov
    C[0][1] = cov
    new_params['C'] = C.copy()
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T, general=True)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG Cov = {cov}', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB Cov = {cov}', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

gammas = [(1, 'red'),(10, 'green'),(15,'blue')]

plt.figure(figsize=(12,7))
for gamma, color in gammas:
    new_params = params.copy()
    new_params['Gamma'] = gamma*params['Gamma']
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T, general=True)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG $\Gamma = {gamma}\Gamma_0$', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB $\Gamma = {gamma}\Gamma_0$', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

gammas = [(1, 'red'),(10, 'green'),(15,'blue')]

plt.figure(figsize=(12,7))
for gamma, color in gammas:
    new_params = params.copy()
    new_params['Gamma'] = np.diag([gamma,1])@params['Gamma']
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T, general=True)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG $\Gamma = \Gamma_0({gamma})$', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB $\Gamma = \Gamma_0({gamma})$', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

Hs = [(1, 'red'),(5, 'green'),(10,'blue')]

plt.figure(figsize=(12,7))
for H, color in Hs:
    new_params = params.copy()
    new_params['H'] = H*params['H']
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T, general=True)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG $\H = {gamma}H_0$', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB $\H = {gamma}H_0$', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()

Hs = [(1, 'red'),(10, 'green'),(15,'blue')]

plt.figure(figsize=(12,7))
for H, color in Hs:
    new_params = params.copy()
    new_params['H'] = np.diag([H,1])@params['H']
    model = AlmgrenChriss(new_params)
    trajectory = model.trajectory(X,T, general=True)
    plt.plot(range(T + 1),trajectory[0],'o-',ms=5, label=f'GOOG $\H = H_0({H})$', color=color)
    plt.plot(range(T + 1),trajectory[1],'o-',ms=5, label=f'FB $\H = H_0({H})$', color='tab:'+color)
plt.title(f'Optimal Liquidation Trajectory ({X} shares in {T} days)',fontsize=14)
plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of shares held', fontsize=12)
plt.legend()
plt.show()
