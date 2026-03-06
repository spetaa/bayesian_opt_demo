import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize

# ── 1. 复杂目标函数 ────────────────────────────────────────────
BOUNDS = (-1, 5)

def true_f(x):
    return (
        np.sin(3 * x)
        + 1.5 * np.sin(5 * x + 0.5)
        + 0.3 * np.cos(9 * x)
        - 0.2 * x
        + 1.2 * np.exp(-0.5 * ((x - 1.5) / 0.4) ** 2)
    )

x_plot = np.linspace(*BOUNDS, 400).reshape(-1, 1)
y_true = true_f(x_plot)

# ── 2. GP 核 & 预测 ────────────────────────────────────────────
def rbf_kernel(X1, X2, ls=0.6, sf=1.0):
    d = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    return sf**2 * np.exp(-0.5 / ls**2 * d)

def gp_predict(X_tr, y_tr, X_te, noise=1e-6):
    K    = rbf_kernel(X_tr, X_tr) + noise * np.eye(len(X_tr))
    K_s  = rbf_kernel(X_tr, X_te)
    K_ss = rbf_kernel(X_te, X_te) + noise * np.eye(len(X_te))
    L    = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
    mu    = K_s.T @ alpha
    v     = np.linalg.solve(L, K_s)
    std   = np.sqrt(np.clip(np.diag(K_ss - v.T @ v), 0, None))
    return mu.ravel(), std

# ── 3. EI 采集函数 ─────────────────────────────────────────────
def ei_acquisition(X_te, X_tr, y_tr, xi=0.02):
    mu, sigma = gp_predict(X_tr, y_tr, X_te)
    best = np.max(y_tr)
    imp  = mu - best - xi
    Z    = imp / (sigma + 1e-9)
    ei   = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei

def next_sample(X_tr, y_tr, n_restarts=20):
    best_x, best_val = None, -np.inf
    for _ in range(n_restarts):
        x0 = np.random.uniform(*BOUNDS)
        res = minimize(
            lambda x: -ei_acquisition(np.array([[x[0]]]), X_tr, y_tr),
            x0=[x0], bounds=[BOUNDS], method='L-BFGS-B'
        )
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x[0]
    return best_x

# ── 4. 初始采样 ────────────────────────────────────────────────
np.random.seed(7)
N_INIT  = 1
N_ITERS = 25

X_train = np.random.uniform(*BOUNDS, N_INIT).reshape(-1, 1)
y_train = true_f(X_train).ravel()

# ── 5. 实时逐帧绘制 ────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(11, 6))

for i in range(N_ITERS):
    mu, std = gp_predict(X_train, y_train, x_plot)
    ei      = ei_acquisition(x_plot, X_train, y_train)
    x_next  = next_sample(X_train, y_train)
    y_next  = true_f(np.array([[x_next]])).ravel()[0]

    # ── 上半：GP 拟合 ──────────────────────────────────────────
    fig.clf()
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(x_plot, y_true, 'k--', lw=1.5, alpha=0.5, label='True f(x)')
    ax1.plot(x_plot, mu, color='steelblue', lw=2, label='GP mean')
    ax1.fill_between(x_plot.ravel(), mu-2*std, mu+2*std,
                     color='steelblue', alpha=0.2, label='95% CI')
    ax1.scatter(X_train, y_train, c='black', s=60, zorder=5, label='Observed')
    ax1.axvline(x_next, color='red', lw=1.8, ls=':', label=f'Next x={x_next:.3f}')
    ax1.scatter([x_next], [y_next], c='red', s=150, zorder=6, marker='*')

    best_idx = np.argmax(y_train)
    ax1.scatter(X_train[best_idx], y_train[best_idx],
                c='gold', s=220, zorder=7, marker='*',
                edgecolors='k', label=f'Best so far: x={X_train[best_idx,0]:.3f}  y={y_train[best_idx]:.3f}')

    ax1.set_xlim(*BOUNDS)
    ax1.set_ylabel('f(x)')
    ax1.legend(loc='upper left', fontsize=7.5, ncol=3)
    ax1.set_title(
        f'Bayesian Optimization  |  Iter {i+1}/{N_ITERS}  |  n_obs={len(X_train)}'
        '        [click anywhere to continue]',
        fontsize=11
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── 下半：EI 曲线 ──────────────────────────────────────────
    ax2.fill_between(x_plot.ravel(), 0, ei, color='orange', alpha=0.5, label='EI')
    ax2.axvline(x_next, color='red', lw=1.8, ls=':')
    ax2.set_xlim(*BOUNDS)
    ax2.set_ylabel('EI')
    ax2.set_xlabel('x')
    ax2.legend(fontsize=8)

    plt.draw()
    plt.waitforbuttonpress()   # 按任意键或点击鼠标进入下一迭代

    # 加入新观测
    X_train = np.vstack([X_train, [[x_next]]])
    y_train = np.append(y_train, y_next)

# ── 6. 最终结果 ────────────────────────────────────────────────
fig.clf()
ax = fig.add_subplot(111)
mu, std = gp_predict(X_train, y_train, x_plot)
ax.plot(x_plot, y_true, 'k--', lw=1.5, alpha=0.5, label='True f(x)')
ax.plot(x_plot, mu, 'steelblue', lw=2, label='Final GP mean')
ax.fill_between(x_plot.ravel(), mu-2*std, mu+2*std, color='steelblue', alpha=0.2)
ax.scatter(X_train, y_train, c='black', s=50, zorder=5, label='All observed')
best_idx = np.argmax(y_train)
ax.scatter(X_train[best_idx], y_train[best_idx],
           c='gold', s=260, zorder=7, marker='*',
           edgecolors='k', label=f'Best: x={X_train[best_idx,0]:.3f}  y={y_train[best_idx]:.3f}')
ax.set_title(f'Final Result  |  n_obs={len(X_train)}', fontsize=12)
ax.set_xlim(*BOUNDS)
ax.legend(fontsize=8, ncol=3)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.ioff()
plt.savefig('bayesian_opt_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done. 图已保存为 bayesian_opt_demo.png")
