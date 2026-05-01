import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    return x1 ** 2 + 5 * (x2 ** 2)

def grad_f(x1, x2):
    df_dx1 = 2 * x1
    df_dx2 = 10 * x2
    return np.array([df_dx1, df_dx2])

def gradient_descent(init_x, lr, max_iter, tol=1e-8):
    x = np.array(init_x, dtype=np.float64)
    trajectory = [x.copy()]
    diverge_flag = False
    converge_iter = None

    for iter_idx in range(max_iter):
        grad = grad_f(x[0], x[1])

        if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.linalg.norm(x) > 1e10:
            diverge_flag = True
            break

        if np.linalg.norm(grad) < tol:
            converge_iter = iter_idx + 1
            break

        x = x - lr * grad
        trajectory.append(x.copy())

    final_f = f(x[0], x[1]) if not diverge_flag else np.nan
    return x, np.array(trajectory), final_f, diverge_flag, converge_iter

init_point = [5, 5]
lr_basic = 0.1
iter_basic = 50
final_x, traj_basic, final_f, _, _ = gradient_descent(init_point, lr_basic, iter_basic, tol=1e-20)

print("=" * 50)
print("2.1(a) Gradient Descent Basic Results")
print(f"Final x1: {final_x[0]:.8f}")
print(f"Final x2: {final_x[1]:.8f}")
print(f"Final function value f(x1,x2): {final_f:.8f}")
print("=" * 50)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(8, 6))

x1_range = np.linspace(-6, 6, 100)
x2_range = np.linspace(-6, 6, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
F_val = f(X1, X2)

contour_plot = plt.contour(X1, X2, F_val, levels=20, cmap='viridis')
plt.clabel(contour_plot, inline=True, fontsize=8)

plt.plot(traj_basic[:, 0], traj_basic[:, 1], 'r-o', markersize=4, linewidth=1, label='GD Trajectory')
plt.scatter(init_point[0], init_point[1], c='red', s=50, marker='*', label='Start (5,5)')
plt.scatter(0, 0, c='blue', s=50, marker='x', label='Global Min (0,0)')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour Plot of $f(x_1,x_2)=x_1^2+5x_2^2$ & GD Trajectory')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('Q2_1_contour_plot.png', dpi=300, bbox_inches='tight')
plt.close()

test_lrs = [0.01, 0.1, 1.1]
max_iter_test = 5000
lr_analysis = {}

for lr in test_lrs:
    x_test, _, f_test, diverge, conv_iter = gradient_descent(init_point, lr, max_iter_test)

    if diverge:
        status = "Diverged"
        iter_info = "—"
    elif conv_iter is not None:
        status = "Converged"
        iter_info = str(conv_iter)
    else:
        status = "Not converged (insufficient iterations)"
        iter_info = "—"

    lr_analysis[lr] = {
        "final_point": f"({x_test[0]:.8f}, {x_test[1]:.8f})" if not diverge else "Numerical overflow (diverged)",
        "final_f": f"{f_test:.8f}" if not diverge else "NaN",
        "status": status,
        "iter": iter_info
    }

for lr in test_lrs:
    res = lr_analysis[lr]
    print(f"\nLearning rate η={lr}:")
    print(f"  Final point: {res['final_point']}")
    print(f"  Final function value: {res['final_f']}")
    print(f"  Status: {res['status']} (iterations: {res['iter']})")

print("\n" + "=" * 50)
print("2.1(c) Learning Rate Analysis Summary")
for lr, res in lr_analysis.items():
    print(f"η={lr}: {res['status']} (iterations: {res['iter']})")
print("η=1.1 divergence reason: learning rate too large, step size exceeds optimal range, causing exponential parameter growth and numerical overflow (divergence).")
print("=" * 50)