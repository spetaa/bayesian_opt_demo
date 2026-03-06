import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.stats import norm
from scipy.optimize import minimize

# ── 1. 复杂 2D 目标函数（多峰 + 高频扰动）─────────────────────
BOUNDS = [(-2, 3), (-2, 3)]

def true_f(X):
    """X: (n, 2) array"""
    x, y = X[:, 0], X[:, 1]
    return (
        np.sin(3 * x) * np.cos(2.5 * y)
        + 2.0 * np.exp(-((x - 1.0)**2 + (y - 1.0)**2) / 0.25)   # 主峰 (1,1)
        + 1.3 * np.exp(-((x + 0.8)**2 + (y - 2.2)**2) / 0.35)   # 次峰 (-0.8,2.2)
        + 0.9 * np.exp(-((x - 2.4)**2 + (y + 0.6)**2) / 0.28)   # 次峰 (2.4,-0.6)
        - 0.7 * np.exp(-((x + 0.2)**2 + (y + 1.3)**2) / 0.5)    # 凹陷
        + 0.25 * np.cos(4 * x * y)                               # 高频扰动
    )

# 绘图网格（60×60）
RES = 60
x_lin = np.linspace(*BOUNDS[0], RES)
y_lin = np.linspace(*BOUNDS[1], RES)
XX, YY = np.meshgrid(x_lin, y_lin)
X_plot = np.column_stack([XX.ravel(), YY.ravel()])
Z_true = true_f(X_plot).reshape(RES, RES)
TRUE_MAX = Z_true.max()

# ── 2. GP 核 & 预测（避免构造大矩阵，只算对角方差）──────────────
def rbf_kernel(X1, X2, ls=0.55, sf=1.5):
    diff = X1[:, None, :] - X2[None, :, :]   # (n1, n2, d)
    return sf**2 * np.exp(-0.5 / ls**2 * np.sum(diff**2, axis=-1))

def rbf_diag(n, sf=1.5):
    return sf**2 * np.ones(n)

def gp_predict(X_tr, y_tr, X_te, noise=1e-5):
    K    = rbf_kernel(X_tr, X_tr) + noise * np.eye(len(X_tr))
    K_s  = rbf_kernel(X_tr, X_te)
    L    = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
    mu    = K_s.T @ alpha
    v     = np.linalg.solve(L, K_s)
    var   = rbf_diag(len(X_te)) - np.sum(v**2, axis=0)
    std   = np.sqrt(np.clip(var, 0, None))
    return mu.ravel(), std

# ── 3. EI 采集函数 ─────────────────────────────────────────────
def ei_acquisition(X_te, X_tr, y_tr, xi=0.05):
    mu, sigma = gp_predict(X_tr, y_tr, X_te)
    imp  = mu - np.max(y_tr) - xi
    Z    = imp / (sigma + 1e-9)
    ei   = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei

def next_sample(X_tr, y_tr, n_restarts=30):
    best_x, best_val = None, -np.inf
    for _ in range(n_restarts):
        x0 = np.array([np.random.uniform(*b) for b in BOUNDS])
        res = minimize(
            lambda x: -ei_acquisition(x.reshape(1, -1), X_tr, y_tr),
            x0=x0, bounds=BOUNDS, method='L-BFGS-B'
        )
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x.copy()
    return best_x

# ── 4. 初始随机采样 ────────────────────────────────────────────
np.random.seed(42)
N_INIT  = 5
N_ITERS = 25

X_train = np.column_stack([np.random.uniform(*b, N_INIT) for b in BOUNDS])
y_train = true_f(X_train)

# ── 5. 实时逐帧绘制（手动步进）────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(17, 7))

for i in range(N_ITERS):
    mu, std = gp_predict(X_train, y_train, X_plot)
    ei      = ei_acquisition(X_plot, X_train, y_train)
    x_next  = next_sample(X_train, y_train)
    y_next  = true_f(x_next.reshape(1, -1))[0]

    Z_mu  = mu.reshape(RES, RES)
    Z_std = std.reshape(RES, RES)
    Z_ei  = ei.reshape(RES, RES)
    best_idx = np.argmax(y_train)

    fig.clf()
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.04, right=0.97,
                           top=0.90, bottom=0.06,
                           wspace=0.38, hspace=0.42)

    # ── (A) 3D 曲面：GP均值（彩色）+ 真实函数（灰色透明）────────
    ax3 = fig.add_subplot(gs[:, 0], projection='3d')
    ax3.plot_surface(XX, YY, Z_true, alpha=0.22, color='silver',
                     rstride=2, cstride=2, linewidth=0)
    surf = ax3.plot_surface(XX, YY, Z_mu, alpha=0.75, cmap='viridis',
                            rstride=2, cstride=2, linewidth=0)
    ax3.scatter(X_train[:, 0], X_train[:, 1], y_train,
                c='black', s=35, zorder=5, depthshade=True, label='Obs')
    ax3.scatter([x_next[0]], [x_next[1]], [y_next],
                c='red', s=140, marker='*', zorder=6, depthshade=False)
    ax3.scatter([X_train[best_idx, 0]], [X_train[best_idx, 1]], [y_train[best_idx]],
                c='gold', s=180, marker='*', edgecolors='k', zorder=7, depthshade=False)
    ax3.set_xlabel('x₁', labelpad=2); ax3.set_ylabel('x₂', labelpad=2)
    ax3.set_zlabel('f', labelpad=2)
    ax3.set_title('GP Mean (color) + True f (gray)', fontsize=9, pad=4)
    ax3.tick_params(labelsize=7)

    # ── (B) GP 均值热图 ──────────────────────────────────────────
    ax_mu = fig.add_subplot(gs[0, 1])
    cm1 = ax_mu.contourf(XX, YY, Z_mu, levels=30, cmap='viridis')
    ax_mu.contour(XX, YY, Z_true, levels=12, colors='white', linewidths=0.4, alpha=0.5)
    ax_mu.scatter(X_train[:, 0], X_train[:, 1],
                  c='white', s=30, edgecolors='k', lw=0.7, zorder=5)
    ax_mu.scatter(*x_next, c='red', s=100, marker='*', zorder=6,
                  label=f'Next ({x_next[0]:.2f}, {x_next[1]:.2f})')
    ax_mu.scatter(X_train[best_idx, 0], X_train[best_idx, 1],
                  c='gold', s=130, marker='*', edgecolors='k', zorder=7,
                  label=f'Best y={y_train[best_idx]:.2f}')
    fig.colorbar(cm1, ax=ax_mu, fraction=0.046, pad=0.04)
    ax_mu.set_title('GP Mean', fontsize=9)
    ax_mu.legend(fontsize=7, loc='lower right')
    ax_mu.tick_params(labelsize=7)

    # ── (C) GP 不确定性（探索地图）──────────────────────────────
    ax_std = fig.add_subplot(gs[1, 1])
    cm2 = ax_std.contourf(XX, YY, Z_std, levels=25, cmap='plasma')
    ax_std.scatter(X_train[:, 0], X_train[:, 1],
                   c='white', s=25, edgecolors='k', lw=0.7, zorder=5)
    ax_std.scatter(*x_next, c='red', s=100, marker='*', zorder=6)
    fig.colorbar(cm2, ax=ax_std, fraction=0.046, pad=0.04)
    ax_std.set_title('GP Std  (Uncertainty)', fontsize=9)
    ax_std.tick_params(labelsize=7)

    # ── (D) EI 热图 ──────────────────────────────────────────────
    ax_ei = fig.add_subplot(gs[0, 2])
    cm3 = ax_ei.contourf(XX, YY, Z_ei, levels=25, cmap='hot')
    ax_ei.scatter(X_train[:, 0], X_train[:, 1],
                  c='cyan', s=25, edgecolors='k', lw=0.7, zorder=5)
    ax_ei.scatter(*x_next, c='white', s=120, marker='*', zorder=6,
                  edgecolors='red', lw=1.2, label='argmax EI')
    fig.colorbar(cm3, ax=ax_ei, fraction=0.046, pad=0.04)
    ax_ei.set_title('EI Acquisition', fontsize=9)
    ax_ei.legend(fontsize=7, loc='lower right')
    ax_ei.tick_params(labelsize=7)

    # ── (E) 收敛曲线 ─────────────────────────────────────────────
    ax_cv = fig.add_subplot(gs[1, 2])
    best_curve = np.maximum.accumulate(y_train)
    ax_cv.plot(range(1, len(y_train) + 1), best_curve,
               'o-', color='steelblue', lw=1.8, ms=4)
    ax_cv.axhline(TRUE_MAX, color='red', ls='--', lw=1.2,
                  label=f'True max={TRUE_MAX:.2f}')
    ax_cv.fill_between(range(1, len(y_train) + 1),
                       best_curve, TRUE_MAX, alpha=0.15, color='red')
    ax_cv.set_xlabel('n_obs', fontsize=8)
    ax_cv.set_ylabel('Best y found', fontsize=8)
    ax_cv.set_title('Convergence', fontsize=9)
    ax_cv.legend(fontsize=7)
    ax_cv.tick_params(labelsize=7)

    fig.suptitle(
        f'Bayesian Opt 2D  |  Iter {i+1}/{N_ITERS}  |  n_obs={len(X_train)}'
        '        [click anywhere to continue]',
        fontsize=11
    )

    plt.draw()
    plt.waitforbuttonpress()

    X_train = np.vstack([X_train, x_next])
    y_train = np.append(y_train, y_next)


# ── 6. 最终结果 ────────────────────────────────────────────────
fig.clf()
gs_f = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32,
                         left=0.04, right=0.97, top=0.88, bottom=0.08)

ax3f = fig.add_subplot(gs_f[0], projection='3d')
mu_f, _ = gp_predict(X_train, y_train, X_plot)
Z_f = mu_f.reshape(RES, RES)
ax3f.plot_surface(XX, YY, Z_true, alpha=0.22, color='silver', rstride=2, cstride=2)
ax3f.plot_surface(XX, YY, Z_f,    alpha=0.78, cmap='viridis', rstride=2, cstride=2)
best_idx = np.argmax(y_train)
ax3f.scatter(X_train[:, 0], X_train[:, 1], y_train,
             c='black', s=25, depthshade=True)
ax3f.scatter([X_train[best_idx, 0]], [X_train[best_idx, 1]], [y_train[best_idx]],
             c='gold', s=260, marker='*', edgecolors='k', depthshade=False)
ax3f.set_title(
    f'Final GP Mean\nBest: ({X_train[best_idx,0]:.2f}, {X_train[best_idx,1]:.2f})'
    f'  y={y_train[best_idx]:.3f}',
    fontsize=9
)
ax3f.set_xlabel('x₁'); ax3f.set_ylabel('x₂'); ax3f.set_zlabel('f')

ax_cv2 = fig.add_subplot(gs_f[1])
best_curve = np.maximum.accumulate(y_train)
ax_cv2.plot(range(1, len(y_train) + 1), best_curve, 'o-', color='steelblue', lw=2, ms=5)
ax_cv2.axhline(TRUE_MAX, color='red', ls='--', lw=1.5,
               label=f'True global max = {TRUE_MAX:.3f}')
ax_cv2.fill_between(range(1, len(y_train) + 1),
                    best_curve, TRUE_MAX, alpha=0.15, color='red', label='Optimality gap')
ax_cv2.set_xlabel('Number of observations')
ax_cv2.set_ylabel('Best f found')
ax_cv2.set_title('Convergence Curve')
ax_cv2.legend()

fig.suptitle(
    f'Bayesian Optimization 2D — Final  |  n_obs={len(X_train)}  '
    f'|  Gap={TRUE_MAX - y_train[best_idx]:.4f}',
    fontsize=12
)

plt.ioff()
plt.savefig('bayesian_opt_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Best found : ({X_train[best_idx,0]:.4f}, {X_train[best_idx,1]:.4f})  "
      f"f = {y_train[best_idx]:.4f}")
print(f"True global max : {TRUE_MAX:.4f}")
print("图已保存为 bayesian_opt_2d.png")
