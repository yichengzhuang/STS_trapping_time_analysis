#!/usr/bin/env python3
"""fit_potential.py — Manual point picking + quadratic (parabolic) potential fit.

Manually click several points on the LDOS map (along what you consider the potential / band-bottom trend),
and fit a quadratic to these (position x, energy E) points, written in vertex form V(x) = a·(x - x0)^2 + V0, giving:
  a   : curvature coefficient (meV/nm^2); a>0 is a potential well (minimum at x0), a<0 is a barrier (maximum)
  x0  : center (nm)
  V0  : vertex energy (meV)
  d2V/dx^2 = 2a
major / minor are each fitted independently and do not affect each other (this is a different matter from the box selection in trapping_roi.py).

Dependency: reuses the data loading and plotting from trapping_roi.py (load_dataset / imshow_map),
so keep it in the same folder as trapping_roi.py and run it inside that folder.

Usage:
  python fit_potential.py                     # default data=ext2, both axes, filter=raw
  python fit_potential.py --axis major        # only major
  python fit_potential.py --data ext2 --filter raw

Operation (local interactive, requires a GUI):
  Left click  = add a point along the potential trend
  Right click = delete the last point
  Close window = finish this axis and start fitting (if both is selected, the next axis window pops up afterward)
  At least 3 points (and not collinear) are needed to fit a quadratic.

Output:
  potential_fit_{axis}_{data}[_{filter}].png  fit figure (base map + your points + parabola + parameters)
  potential_points_{axis}_{data}.csv          your original picked (x,E) points (reproducible / reusable)
  potential_fits.csv                          parameter summary (appended, one fit per row)

Note: filter defaults to raw — the potential / band bottom is a slowly varying envelope, only visible in raw; hp (envelope removed) subtracts
it away, and is generally not used to trace the potential. Use --filter hp if you want to see the structure after envelope removal.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import trapping_roi as tr   # reuse load_dataset / imshow_map

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_points(d, x, E, filt, title):
    """Pop up a figure; left click adds a point / right click deletes a point / close window to finish. Returns [(x, E), ...]."""
    pts, markers = [], []
    fig, ax = plt.subplots(figsize=(11, 7))
    tr.imshow_map(ax, d, x, E, filt, title)

    def _retitle():
        ax.set_title("%s  |  %d pts  (L-click add / R-click undo / close to fit)"
                     % (title, len(pts)), fontsize=9)

    def onclick(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        if ev.button == 1:                       # left click: add point
            pts.append((ev.xdata, ev.ydata))
            m, = ax.plot(ev.xdata, ev.ydata, "o", mfc="lime", mec="k", ms=7, zorder=5)
            markers.append(m)
            print("  + point %d: x=%.2f nm  E=%.2f meV" % (len(pts), ev.xdata, ev.ydata))
        elif ev.button == 3 and pts:             # right click: delete the last one
            pts.pop(); markers.pop().remove()
            print("  - deleted, %d points left" % len(pts))
        _retitle(); fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return pts


def fit_quadratic(xp, Ep):
    """E = a·x^2 + b·x + c  →  vertex form V(x)=a(x-x0)^2+V0.
    Returns dict: a,b,c,x0,V0,d2V,R2 and their errors sa,sb,sc,sx0,sV0,sd2V (errors are nan when n<4).
    Errors: polyfit(cov=True) gives the coefficient covariance matrix, then the Jacobian propagates to x0=-b/2a and V0=c-b²/4a.
    """
    xp = np.asarray(xp, float); Ep = np.asarray(Ep, float)
    n = xp.size
    if n >= 4:                                    # cov needs residual degrees of freedom n-(deg+1)>=1
        coef, cov = np.polyfit(xp, Ep, 2, cov=True)
    else:
        coef = np.polyfit(xp, Ep, 2); cov = np.full((3, 3), np.nan)
    a, b, c = (float(v) for v in coef)
    sa, sb, sc = (float(v) for v in np.sqrt(np.clip(np.diag(cov), 0, None)))
    finite_cov = bool(np.all(np.isfinite(cov)))
    if abs(a) < 1e-12:                            # nearly collinear → quadratic degenerates, vertex undefined
        x0 = V0 = float("nan"); sx0 = sV0 = float("nan")
    else:
        x0 = -b / (2 * a)
        V0 = c - b * b / (4 * a)
        if finite_cov:
            Jx0 = np.array([b / (2 * a * a), -1.0 / (2 * a), 0.0])       # ∂x0/∂(a,b,c)
            JV0 = np.array([b * b / (4 * a * a), -b / (2 * a), 1.0])     # ∂V0/∂(a,b,c)
            sx0 = float(np.sqrt(max(Jx0 @ cov @ Jx0, 0.0)))
            sV0 = float(np.sqrt(max(JV0 @ cov @ JV0, 0.0)))
        else:
            sx0 = sV0 = float("nan")
    pred = a * xp ** 2 + b * xp + c
    ss_res = float(np.sum((Ep - pred) ** 2))
    ss_tot = float(np.sum((Ep - Ep.mean()) ** 2)) + 1e-30
    R2 = 1 - ss_res / ss_tot
    return dict(a=a, b=b, c=c, x0=float(x0), V0=float(V0), d2V=2 * a, R2=R2,
                sa=sa, sb=sb, sc=sc, sx0=sx0, sV0=sV0, sd2V=2 * sa, n=int(n))


def combine_axes(fmaj, fmin):
    """major/minor two 1D parabolas → 2D anisotropic harmonic-oscillator potential
    V(x,y) = a_x(x-x0)^2 + a_y(y-y0)^2 + V0.
    a_x,a_y = curvatures of the two axes; V0 = the two axis vertex energies averaged with 1/σ² weights (simple average if σ is invalid);
    V0_diff = difference of the two axis vertex energies (consistency check); aniso_ratio = a_x/a_y.
    """
    ax, ay = fmaj["a"], fmin["a"]
    V0x, V0y = fmaj["V0"], fmin["V0"]
    sV0x, sV0y = fmaj.get("sV0", float("nan")), fmin.get("sV0", float("nan"))
    if np.isfinite(sV0x) and np.isfinite(sV0y) and sV0x > 0 and sV0y > 0:
        wx, wy = 1.0 / sV0x ** 2, 1.0 / sV0y ** 2
        V0 = (V0x * wx + V0y * wy) / (wx + wy)
        sV0 = 1.0 / np.sqrt(wx + wy)
    else:
        V0 = 0.5 * (V0x + V0y); sV0 = float("nan")
    return dict(ax=float(ax), ay=float(ay),
                sax=float(fmaj.get("sa", float("nan"))), say=float(fmin.get("sa", float("nan"))),
                x0=float(fmaj["x0"]), y0=float(fmin["x0"]),
                sx0=float(fmaj.get("sx0", float("nan"))), sy0=float(fmin.get("sx0", float("nan"))),
                V0=float(V0), sV0=float(sV0),
                V0_major=float(V0x), V0_minor=float(V0y), V0_diff=float(abs(V0x - V0y)),
                aniso_ratio=float(ax / ay) if ay != 0 else float("nan"))


def save_outputs(axis, data, filt, xp, Ep, fit, d, x, E):
    xp = np.asarray(xp, float); Ep = np.asarray(Ep, float)
    # 1) original points (reproducible / reusable)
    pp = os.path.join(HERE, "potential_points_%s_%s.csv" % (axis, data))
    with open(pp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x_nm", "E_meV"]); w.writerows(zip(xp, Ep))
    # 2) parameter summary (append)
    fp = os.path.join(HERE, "potential_fits.csv")
    new = not os.path.exists(fp)
    with open(fp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["axis", "data", "filter", "n_pts", "a_meV_per_nm2", "b", "c",
                        "x0_nm", "V0_meV", "d2V_meV_per_nm2", "R2"])
        w.writerow([axis, data, filt, len(xp),
                    round(fit["a"], 6), round(fit["b"], 4), round(fit["c"], 3),
                    round(fit["x0"], 2), round(fit["V0"], 2),
                    round(fit["d2V"], 6), round(fit["R2"], 4)])
    # 3) fit figure
    fig, ax = plt.subplots(figsize=(11, 7))
    tr.imshow_map(ax, d, x, E, filt, "%s %s (%s) — quadratic potential fit" % (data.upper(), axis, filt))
    xs = np.linspace(float(xp.min()), float(xp.max()), 300)
    ax.plot(xs, fit["a"] * xs ** 2 + fit["b"] * xs + fit["c"], "-", color="lime", lw=2.2,
            label="fit V(x)=a(x-x0)²+V0")
    ax.plot(xp, Ep, "o", mfc="cyan", mec="k", ms=7, label="picked points")
    if np.isfinite(fit["x0"]):
        ax.plot([fit["x0"]], [fit["V0"]], "*", color="yellow", mec="k", ms=17,
                label="vertex (x0,V0)")
    txt = ("a = %.4g meV/nm²\nx0 = %.2f nm\nV0 = %.2f meV\nd²V/dx² = %.4g meV/nm²\nR² = %.4f"
           % (fit["a"], fit["x0"], fit["V0"], fit["d2V"], fit["R2"]))
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=9, va="bottom", color="w",
            bbox=dict(boxstyle="round", fc="0.2", alpha=0.75))
    ax.legend(fontsize=8, loc="upper right")
    suffix = "" if filt == "raw" else "_" + filt
    p = os.path.join(HERE, "potential_fit_%s_%s%s.png" % (axis, data, suffix))
    fig.savefig(p, dpi=140); plt.close(fig)
    print("  saved:", p)
    print("         ", pp)
    print("         ", fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ext2",
                    help="dataset (reads {data}_{axis}.txt), default ext2")
    ap.add_argument("--axis", choices=["major", "minor", "both"], default="both")
    ap.add_argument("--filter", choices=["raw", "hp"], default="raw",
                    help="raw=view potential / band bottom (default, recommended); hp=view structure after envelope removal")
    args = ap.parse_args()

    maps = tr.load_dataset(args.data, args.filter)
    axes = ("major", "minor") if args.axis == "both" else (args.axis,)
    for axis in axes:
        d, x, E = maps[axis]
        print("\n=== %s %s (%s): left click to pick points along the potential, right click to delete, close window to fit ==="
              % (args.data, axis, args.filter))
        pts = pick_points(d, x, E, args.filter,
                          "%s %s (%s) — pick points along potential, close to fit"
                          % (args.data.upper(), axis, args.filter))
        if len(pts) < 3:
            print("  only picked %d points (<3), cannot fit a quadratic, skipping %s." % (len(pts), axis))
            continue
        xp = [p[0] for p in pts]; Ep = [p[1] for p in pts]
        fit = fit_quadratic(xp, Ep)
        print("  V(x)=a(x-x0)²+V0 :  a=%.4g meV/nm²  x0=%.2f nm  V0=%.2f meV"
              "  (d²V/dx²=%.4g)  R²=%.4f"
              % (fit["a"], fit["x0"], fit["V0"], fit["d2V"], fit["R2"]))
        save_outputs(axis, args.data, args.filter, xp, Ep, fit, d, x, E)


if __name__ == "__main__":
    main()
