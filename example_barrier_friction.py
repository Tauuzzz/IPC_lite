"""可视化 barrier 势能与摩擦能的近似和取舍 ---- 配合 README「取舍与近似」章节.

三个子图:
  1. barrier 能量 b(d) 与其力 N(d) = -b'(d): 看 log 发散、"只在 d < d_hat 激活"、
     以及本实现把 d² 钳到 1e-12 防止 log(0) 发散的保险丝位置.
  2. 摩擦光滑化 f0(y) 与不光滑的 |y|: 看 y_eps 附近把尖角抹平成 C1,
     这是 Newton 法能收敛的关键.
  3. 摩擦能 D(y) = μ·λ·f0(‖y‖): 一维切片,λ 越大曲线越陡(法向压得越紧,
     切向越难滑).

所有曲线都用 numpy 复算公式(和 kernel 代码逐式对应),不依赖 warp,
跑起来秒出.运行后在当前目录生成 barrier_friction_curves.png.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

# 中文字体:Noto Sans CJK JP(本机 ttc 里的 JP 变体实际同时覆盖中文字形和
# 拉丁字符;系统装的是 Noto CJK 全家桶,SC/TC 也在同一个 ttc 里).
# 找不到就退回默认字体(中文会显示为方框,不影响程序运行)
import matplotlib.font_manager as _fm
_available = {font.name for font in _fm.fontManager.ttflist}
for _f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC"]:
    if _f in _available:
        matplotlib.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

import barrier_energy
import friction_energy

# 与 main.py 一致的参数(只影响 λ 采样,不影响曲线形状)
D_TILDE = 1e-3
KAPPA = 1e5
FRICTION_MU = 0.3
EPS_V = 1e-3


def barrier_energy_np(d, d_tilde, kappa):
    """barrier_energy kernel 的 numpy 复算(含 d² 钳位保险丝)."""
    d = np.asarray(d, dtype=np.float64)
    d_tilde_sq = d_tilde * d_tilde
    u = d * d
    u = np.where(u < d_tilde_sq, u, d_tilde_sq)   # 未激活时能量为 0
    u = np.maximum(u, 1e-12)                      # 保险丝:防 log(0)
    diff = u - d_tilde_sq
    return np.where(
        d < d_tilde,
        -kappa * diff * diff * np.log(u / d_tilde_sq),
        0.0,
    )


def barrier_force_np(d, d_tilde, kappa):
    """barrier_force_magnitude 的 numpy 复算:N = -db/dd."""
    d = np.asarray(d, dtype=np.float64)
    d_tilde_sq = d_tilde * d_tilde
    u = np.maximum(d * d, 1e-24)
    diff = u - d_tilde_sq
    ratio = u / d_tilde_sq
    db_du = -kappa * (2.0 * diff * np.log(ratio) + diff * diff / u)
    N = -db_du * 2.0 * np.sqrt(u)
    return np.where((d >= d_tilde) | (d <= 1e-12), 0.0, N)


def friction_f0_np(y, y_eps):
    """friction_f0 的 numpy 复算:|y| 的 C1 光滑化."""
    y = np.abs(np.asarray(y, dtype=np.float64))
    return np.where(
        y >= y_eps,
        y,
        y * y / y_eps - y**3 / (3.0 * y_eps * y_eps) + y_eps / 3.0,
    )


def main():
    y_eps = 0.01 * EPS_V  # = DT·eps_v,与 main.py 的 demo 一致(DT = 0.01)

    # 让输出数值和 kernel 实现一致(交叉验证)
    print("== 与 warp kernel 的一致性检查 ==")
    d_check = np.array([2e-4, 5e-4, 9e-4])
    print(f"b(d)  numpy: {barrier_energy_np(d_check, D_TILDE, KAPPA)}")
    print(f"N(d)  numpy: {barrier_force_np(d_check, D_TILDE, KAPPA)}")
    print(f"f0(y) numpy: {friction_f0_np(np.array([5e-7, 2e-6]), y_eps)}")
    print("(kernel 实现在 barrier_energy.py / friction_energy.py,公式逐式对应)\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    d = np.linspace(1e-5, 2e-3, 2000)

    # ---- 子图 1: barrier 能量与力 ----
    ax = axes[0]
    b = barrier_energy_np(d, D_TILDE, KAPPA)
    N = barrier_force_np(d, D_TILDE, KAPPA)
    ax.plot(d, b / 1e3, label="能量 b(d)  [×1e3 J]", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(d, N / 1e6, label="力 N(d) = −b′(d)  [×1e6 N]", color="tab:red", ls="--")
    ax.axvline(D_TILDE, color="gray", ls=":", lw=1)
    ax.text(D_TILDE * 1.05, ax.get_ylim()[1] * 0.05, "d_hat = d_tilde\n(激活阈值)", fontsize=9, color="gray")
    ax.axvline(1e-6, color="black", ls=":", lw=1)
    ax.text(1.1e-6, ax.get_ylim()[1] * 0.35, "d²钳位 1e-12\n(防 log(0) 发散)", fontsize=9)
    ax.set_xlabel("距离 d [m]")
    ax.set_ylabel("能量 b(d)")
    ax2.set_ylabel("法向力 N(d)")
    ax.set_title("Barrier: 只在 d < d_hat 激活,d→0 时 log 发散顶住物体")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # ---- 子图 2: 摩擦光滑化 f0(y) vs |y| ----
    ax = axes[1]
    y = np.linspace(-2 * y_eps, 2 * y_eps, 2000)
    ax.plot(y / y_eps, np.abs(y), label="|y|(原始,C0 尖角)", color="gray", ls="--")
    ax.plot(y / y_eps, friction_f0_np(y, y_eps), label="f0(y)(C1 光滑)", color="tab:green", lw=2)
    ax.axvspan(-1, 1, alpha=0.08, color="green")
    ax.text(0, 1.55 * y_eps, "静摩擦区\n|y| < y_eps", ha="center", fontsize=9, color="green")
    ax.set_xlabel("滑动位移 y / y_eps")
    ax.set_ylabel("f0(y)")
    ax.set_title("摩擦光滑化: 把 |y| 的尖角抹平,y_eps 处值与导数接续")
    ax.legend(fontsize=9)

    # ---- 子图 3: 不同法向力下的摩擦能 D(y) ----
    ax = axes[2]
    lamdas = [0.0, 50.0, 200.0, 800.0]   # N(d) 在 d = 0.4·d_hat 附近的量级
    for lam in lamdas:
        D = FRICTION_MU * lam * friction_f0_np(y, y_eps)
        ax.plot(y / y_eps, D, label=f"λ = {lam:.0f} N")
    ax.set_xlabel("滑动位移 y / y_eps")
    ax.set_ylabel("摩擦能 D(y) = μ·λ·f0(‖y‖)")
    ax.set_title("摩擦能: λ 越大(压得越紧)曲线越陡,越难滑动")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = "barrier_friction_curves.png"
    fig.savefig(out, dpi=140)
    print(f"已生成 {out}")

    print("\n== 取舍与近似速查(详见 README) ==")
    print("1. barrier 只对固定候选集计算,不每步重建候选(教学简化)")
    print(f"2. d² 钳位 1e-12: log(0) 发散的保险丝;正常解 d 停在 ~d_tilde,永远到不了这")
    print(f"3. λ 用滞后构型 x^n 冻结: Newton 期间目标函数固定才收敛;代价是摩擦力晚一步")
    print(f"4. f0 三次多项式光滑化: |y| 在 0 处不可导,Newton 需要至少 C1;y_eps = DT·eps_v")


if __name__ == "__main__":
    main()
