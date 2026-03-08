import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def make_pdf(f, a, b, c, d):
    def unnorm(x):
        exponent = f + a*x + b*x**2 + c*x**3 + d*x**4
        return np.exp(np.clip(exponent, -500, 500))   # clip avoids overflow

    g, _ = integrate.quad(unnorm, -np.inf, np.inf, limit=200)

    def pdf(x):
        return unnorm(x) / g

    return pdf, g


def compute_moments(pdf):
    mean, _ = integrate.quad(lambda x: x * pdf(x), -np.inf, np.inf, limit=200)
    var,  _ = integrate.quad(lambda x: (x - mean) **
                             2 * pdf(x), -np.inf, np.inf, limit=200)
    std = np.sqrt(var)
    skew, _ = integrate.quad(lambda x: ((x - mean) / std)
                             ** 3 * pdf(x), -np.inf, np.inf, limit=200)
    kurt, _ = integrate.quad(lambda x: ((x - mean) / std)
                             ** 4 * pdf(x), -np.inf, np.inf, limit=200)
    return mean, var, skew, kurt


param_sets = [
    dict(f=0, a=1, b=-1, c=0, d=0),
    dict(f=0, a=1, b=-1, c=0, d=-2),
    dict(f=0, a=1, b=4, c=0, d=-2),
]

titles = [
    r"$f=0,\ a=1,\ b=-1,\ c=0,\ d=0$",
    r"$f=0,\ a=1,\ b=-1,\ c=0,\ d=-2$",
    r"$f=0,\ a=1,\ b=4,\ c=0,\ d=-2$",
]

colors = ["#4C9BE8", "#E8764C", "#4CE87A"]
print(f"{'':>6}  {'Mean':>10}  {'Variance':>10}  {
      'Skewness':>10}  {'Kurtosis':>10}")
print("-------")

pdfs = []
moments = []

for p, title in zip(param_sets, titles):
    pdf, g = make_pdf(**p)
    mean, var, skew, kurt = compute_moments(pdf)
    pdfs.append(pdf)
    moments.append((mean, var, skew, kurt))

    label = f"b={p['b']}, d={p['d']}"
    print(f"{label:>12}  {mean:>10.4f}  {
          var:>10.4f}  {skew:>10.4f}  {kurt:>10.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    r"PDF: $p(x) = \frac{1}{g}\exp\{f + ax + bx^2 + cx^3 + dx^4\}$",
    fontsize=13, y=1.02
)

x = np.linspace(-4, 4, 1000)

for ax, pdf, title, color, (mean, var, skew, kurt) in zip(
        axes, pdfs, titles, colors, moments):

    y = np.array([pdf(xi) for xi in x])

    ax.plot(x, y, color=color, lw=2.5, label="p(x)")
    ax.fill_between(x, y, alpha=0.15, color=color)
    ax.axvline(mean, color=color, lw=1.2, ls="--",
               alpha=0.8, label=f"Mean = {mean:.3f}")

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$p(x)$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.6)

    stats_str = (
        f"$\\mu$    = {mean:.3f}\n"
        f"$\\sigma^2$ = {var:.3f}\n"
        f"Skew  = {skew:.3f}\n"
        f"Kurt  = {kurt:.3f}"
    )
    ax.text(
        0.03, 0.97, stats_str,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.7, edgecolor="#ccc"),
    )

plt.tight_layout()
plt.savefig("pdf_plots.png", dpi=150, bbox_inches="tight")
