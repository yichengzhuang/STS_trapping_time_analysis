#!/usr/bin/env python3
"""trapping_roi.py — manual ROI + integration along distance + Lorentzian fit → trapping time τ=ℏ/Γ.

Runs locally and interactively (requires a GUI / X11). Depends only on numpy, scipy, matplotlib; data is read from
the txt files in this folder (no igor2 / cluster needed).

Two switches (each run processes only one dataset → EXP and SIM use their own independent ROIs, with no cross-effect):
  --data   exp / sim   choose experimental or simulated data (this run only plots and computes this one set)
  --filter raw / hp    whether to high-pass filter (hp = subtract envelope, σ=6.5nm × 8.5meV)

Workflow:
  1. Read the selected dataset's {data}_major.txt and {data}_minor.txt (LDOS, paper_data format);
     also save an EXP|SIM comparison preview (raw/hp, major/minor) to help you decide which --filter to use.
  2. Pop up two figures in sequence: first major (drag boxes to draw several ROIs, close window) → then minor (draw the same, close window).
  3. Each ROI: the intensity inside the box is **integrated along position (distance)** → I(E); fit I(E) with a Lorentzian;
     τ = ℏ/Γ (ℏ=658.2 meV·fs).
  4. Output three figures + one csv:
       fits_major_{data}_{filter}.png  — fits of all major ROIs (one subplot each)
       fits_minor_{data}_{filter}.png  — fits of all minor ROIs (one subplot each)
       ldos_overview_{data}_{filter}.png — long (major) / short (minor) axis LDOS side by side,
                                            with all ROI boxes drawn, τ labeled in green inside each box
       roi_results.csv                  — summary (Γ, τ, R², etc.; appended)

Usage:
  python trapping_roi.py --data exp --filter hp
  python trapping_roi.py --data sim --filter raw

Notes:
  - The SIM peak width includes KPM η=2meV (Gaussian broadening); it is a lower-bound reference for lifetime, not a pure physical lifetime.
  - The EXP peak width includes thermal broadening (~3.5kT) + lock-in instrument broadening (not removed) → τ is a lower bound.
  - The hp (filtered) map is bipolar (red/blue); when drawing ROIs, box the positive (red) blob; the raw map has strictly positive intensity,
    with peaks sitting on a background (the fit includes a constant term).
  - Run EXP and SIM once each (each with its own ROIs), then use roi_results.csv to compare τ_exp vs τ_sim.
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

HBAR = 658.2            # meV·fs  → τ[fs] = HBAR / Γ[meV]
SX_NM, SE_MEV = 6.5, 8.5   # high-pass filter (envelope subtraction) scales
HERE = os.path.dirname(os.path.abspath(__file__))


def load_txt(path):
    """paper_data format: row0=[NaN, E...], col0=[NaN, pos...], inner block=LDOS(pos, E)."""
    a = np.loadtxt(path, comments="#")
    return a[1:, 1:], a[1:, 0], a[0, 1:]   # ldos(pos,E), x(nm), E(meV)


# ---------------------------------------------------------------------------
# load_any: adaptively read two kinds of dI/dV txt —— the project-standard paper_data format, or a
# plain-matrix format exported by the user (comment header contains pos_nm=/bias_mV=/shape=). Uniformly returns (ldos[pos,E], x_nm, E_meV).
# Used by the GUI to import files from arbitrary paths; the CLI's load_txt stays unchanged.
# ---------------------------------------------------------------------------
_FLOAT = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_POS_RE = re.compile(r"pos_nm=" + _FLOAT + r"\.\." + _FLOAT + r"(?:\s*\((\d+)\))?")
_BIAS_RE = re.compile(r"bias_mV=" + _FLOAT + r"\.\." + _FLOAT + r"(?:\s*\((\d+)\))?")
_SHAPE_RE = re.compile(r"shape\s*=?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")


def _read_header(path, max_lines=30):
    """Read the comment lines (#) at the start of the file and join them into one string, stopping at the first data line."""
    out = []
    with open(path) as f:
        for _ in range(max_lines):
            ln = f.readline()
            if not ln:
                break
            s = ln.lstrip()
            if s.startswith("#"):
                out.append(ln)
            elif s.strip() == "":
                continue
            else:
                break
    return " ".join(out)


def _mono(v):
    """Finite and strictly monotonic (increasing or decreasing) —— an axis must satisfy this, the first row/column of an intensity matrix will not."""
    v = np.asarray(v, float)
    return v.size >= 2 and np.all(np.isfinite(v)) and (
        np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0))


def _axis_from_header(header, regex, n):
    """Parse start..stop (N) from the comment → linspace; if the range is missing, fall back to pixel indices and mark as uncalibrated."""
    m = regex.search(header)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        nn = int(m.group(3)) if m.group(3) else n
        return np.linspace(lo, hi, nn), True
    return np.arange(n, dtype=float), False


def load_any(path, return_meta=False):
    """Adaptively read two kinds of dI/dV txt, uniformly returning (ldos[pos,E], x_nm, E_meV)[, meta].
    Both have rows=position / cols=bias, not transposed.
      (a) paper_data: NaN at top-left corner, row 0=E, col 0=pos, inner block=LDOS(pos,E).
      (b) plain matrix: comment header contains pos_nm=/bias_mV=/shape=, matrix rows=pos, cols=bias.
    meta = dict(fmt, x_cal, E_cal); x_cal/E_cal=False → that axis falls back to pixel indices (uncalibrated).
    (c) Igor .pxp: dispatched to load_pxp by extension, reads the didv_matrix wave + restores axes from sfA/sfB.
    """
    if str(path).lower().endswith(".pxp"):
        ldos, x, E = load_pxp(path)
        meta = dict(fmt="pxp", x_cal=True, E_cal=True)
        return (ldos, x, E, meta) if return_meta else (ldos, x, E)
    header = _read_header(path)
    a = np.atleast_2d(np.loadtxt(path, comments="#"))
    if a.size == 0:
        raise ValueError("empty file / no numeric values: %s" % path)

    # --- (a) paper_data: NaN at top-left corner + first row/column finite and strictly monotonic ---
    if (a.shape[0] >= 2 and a.shape[1] >= 2 and np.isnan(a[0, 0])
            and _mono(a[0, 1:]) and _mono(a[1:, 0])):
        ldos = a[1:, 1:].astype(float)
        x = a[1:, 0].astype(float)
        E = a[0, 1:].astype(float)
        meta = dict(fmt="paper_data", x_cal=True, E_cal=True)
        return (ldos, x, E, meta) if return_meta else (ldos, x, E)

    # --- (b) plain matrix: parse axes from comments ---
    M = a.astype(float)
    npos, nbias = M.shape
    sm = _SHAPE_RE.search(header)
    if sm:
        s0, s1 = int(sm.group(1)), int(sm.group(2))
        if (s0, s1) == (nbias, npos) and s0 != s1:        # transposed relative to comment → set it straight
            M = M.T
            npos, nbias = M.shape
        elif (s0, s1) != (npos, nbias):
            raise ValueError("matrix %s does not match comment shape=(%d,%d): %s"
                             % ((npos, nbias), s0, s1, path))
    x, x_cal = _axis_from_header(header, _POS_RE, npos)
    E, E_cal = _axis_from_header(header, _BIAS_RE, nbias)
    if x.size != npos or E.size != nbias:                 # axis length does not match the matrix
        if x.size == nbias and E.size == npos:            # rows and columns swapped → transpose
            M = M.T
            npos, nbias = M.shape
        else:                                             # fallback: keep endpoints, resample to matrix dimensions
            if x.size != npos:
                x = (np.linspace(x[0], x[-1], npos) if x_cal and x.size >= 2
                     else np.arange(npos, dtype=float))
            if E.size != nbias:
                E = (np.linspace(E[0], E[-1], nbias) if E_cal and E.size >= 2
                     else np.arange(nbias, dtype=float))
    if not x_cal:
        print("[load_any] warning: pos axis uncalibrated (pixel indices): %s" % path)
    if not E_cal:
        print("[load_any] warning: bias axis uncalibrated (pixel indices): %s" % path)
    meta = dict(fmt="matrix", x_cal=x_cal, E_cal=E_cal)
    return (M, x, E, meta) if return_meta else (M, x, E)


def load_pxp(path, wave_name="didv_matrix"):
    """Read an Igor packed experiment (.pxp), take the didv_matrix wave (rows=position, cols=bias),
    and restore the nm/meV axes from the wave_header's sfA (per-dimension delta) / sfB (per-dimension offset).
    Returns (ldos[pos,E], x_nm, E_meV). Requires igor2.

    Does not use igor2's packed.load: it constructs a VariablesRecord, and the variable
    records of some pxp files (e.g. minor) trigger an internal igor2 bug (binarywave indexes a list as a dict → 'list indices...str').
    Here we read record by record at the byte level ourselves, constructing only WaveRecord and skipping other records to avoid the bug, stopping once found.
    """
    import igor2.packed as pk
    from igor2.record.wave import WaveRecord
    f = open(path, "rb")
    byte_order = None
    ibo = "="
    hit = None
    try:
        while True:
            hs = pk.setup_packed_file_record_header(byte_order=ibo)
            b = bytes(f.read(hs.size))
            if not b or len(b) < hs.size:
                break
            header = hs.unpack_from(b)
            if header["version"] and not byte_order:
                need = pk._need_to_reorder_bytes(header["version"])
                byte_order = ibo = pk._byte_order(need)
                if need:
                    hs = pk.setup_packed_file_record_header(byte_order=byte_order)
                    header = hs.unpack_from(b)
            data = bytes(f.read(header["numDataBytes"]))     # read the full data segment to advance the file pointer correctly
            if len(data) < header["numDataBytes"]:
                break
            rt = pk._RECORD_TYPE.get(header["recordType"] & pk.PACKEDRECTYPE_MASK, None)
            if rt is not WaveRecord:                          # only touch waves: skip Variables/History/... to avoid the bug
                continue
            try:
                rec = rt(header, data, byte_order=byte_order)
            except Exception:                                 # skip a single bad wave too, without affecting the rest
                continue
            wh = rec.wave["wave"]["wave_header"]
            name = wh["bname"]
            name = name.decode() if isinstance(name, bytes) else name
            if name == wave_name:
                hit = (rec.wave["wave"]["wData"], wh)
                break                                         # stop once found (do not read through thousands of waves)
    finally:
        f.close()
    if hit is None:
        raise ValueError("wave '%s' not found in pxp: %s" % (wave_name, path))
    wData, wh = hit
    arr = np.asarray(wData, dtype=float)                     # (npos, nbias) = (position, bias)
    if arr.ndim != 2:
        raise ValueError("wave %s is not 2-D (shape=%s): %s" % (wave_name, arr.shape, path))
    sfA = wh["sfA"]; sfB = wh["sfB"]
    npos, nbias = arr.shape
    x = sfB[0] + sfA[0] * np.arange(npos)                    # position (nm)
    E = sfB[1] + sfA[1] * np.arange(nbias)                   # bias (mV)
    return arr, x.astype(float), E.astype(float)


def hp_filter(d, x, E):
    dx = abs(x[1] - x[0]); dE = abs(E[1] - E[0])
    return d - gaussian_filter(d, sigma=(SX_NM / dx, SE_MEV / dE))


def lorentz(E, A, E0, G, c):
    return A * (G / 2) ** 2 / ((E - E0) ** 2 + (G / 2) ** 2) + c


def get_maps(axis, filt):
    """Read both sets (EXP, SIM) —— only for save_previews to make the comparison figure."""
    de, xe, Ee = load_txt(os.path.join(HERE, "exp_%s.txt" % axis))
    ds, xs, Es = load_txt(os.path.join(HERE, "sim_%s.txt" % axis))
    if filt == "hp":
        de, ds = hp_filter(de, xe, Ee), hp_filter(ds, xs, Es)
    return (de, xe, Ee), (ds, xs, Es)


def load_dataset(data, filt):
    """Read only the selected dataset's (exp or sim) major + minor, filtering as needed. Returns {axis:(d,x,E)}."""
    maps = {}
    for axis in ("major", "minor"):
        d, x, E = load_txt(os.path.join(HERE, "%s_%s.txt" % (data, axis)))
        if filt == "hp":
            d = hp_filter(d, x, E)
        maps[axis] = (d, x, E)
    return maps


def imshow_map(ax, d, x, E, filt, title):
    if filt == "hp":
        v = max(abs(np.percentile(d, 1)), np.percentile(d, 99))
        kw = dict(cmap="RdBu_r", vmin=-v, vmax=v)
    else:
        kw = dict(cmap="inferno", vmin=np.percentile(d, 1), vmax=np.percentile(d, 99))
    ax.imshow(d.T, aspect="auto", origin="lower",
              extent=[x.min(), x.max(), E.min(), E.max()], **kw)
    ax.set_xlabel("position (nm)"); ax.set_ylabel("E / bias (meV / mV)")
    ax.set_title(title, fontsize=10)


def save_previews(axis):
    """EXP|SIM comparison figure (one each for raw and hp), to help decide --filter."""
    for filt in ("raw", "hp"):
        (de, xe, Ee), (ds, xs, Es) = get_maps(axis, filt)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        imshow_map(axs[0], de, xe, Ee, filt, "EXP %s (%s)" % (axis, filt))
        imshow_map(axs[1], ds, xs, Es, filt, "SIM %s (%s)" % (axis, filt))
        p = os.path.join(HERE, "preview_%s_%s.png" % (filt, axis))
        fig.savefig(p, dpi=130); plt.close(fig); print("saved", p)


def integrate_roi(d, x, E, roi):
    """ROI=(x0,x1,E0,E1): integrate the intensity inside the box along position → I(E). Returns (Ew, I)."""
    x0, x1, E0, E1 = roi
    ix = (x >= x0) & (x <= x1); iE = (E >= E0) & (E <= E1)
    return E[iE], d[ix][:, iE].sum(axis=0)


def fit_lorentz(Ew, I, bg="const"):
    """Fit I(E) with a Lorentzian peak. bg="const" adds a constant background (4 params);
    bg="quad" adds a quadratic background a2·t²+a1·t+a0 with t=E−mean(E) (6 params, energy
    centered for numerical stability) — a joint peak+background fit, useful on raw data."""
    need = 7 if bg == "quad" else 4      # quad has 6 params → keep >=1 residual DOF for a finite covariance
    if Ew.size < need:                   # box too small, not enough points to fit
        print("  fit skip: <%d points" % need); return None
    A0 = I.max() - np.median(I); E0 = Ew[int(np.argmax(I))]
    Em = float(np.mean(Ew))
    try:
        if bg == "quad":
            # robust seeds: strip an endpoint linear baseline first, so a strongly sloped raw
            # background does not drag the peak position / amplitude guesses to the window edge
            base = I[0] + (I[-1] - I[0]) * (Ew - Ew[0]) / (Ew[-1] - Ew[0] + 1e-9)
            e0q = Ew[int(np.argmax(I - base))]; Aq = float((I - base).max())
            def _model(E, A, e0, G, a2, a1, a0):
                t = E - Em
                return A * (G / 2) ** 2 / ((E - e0) ** 2 + (G / 2) ** 2) + a2 * t * t + a1 * t + a0
            a1_0 = (I[-1] - I[0]) / (Ew[-1] - Ew[0] + 1e-9)
            popt, pcov = curve_fit(_model, Ew, I, p0=[Aq, e0q, 4.0, 0.0, a1_0, float(np.median(I))],
                                   bounds=([0, Ew.min(), 0.3, -np.inf, -np.inf, -np.inf],
                                           [np.inf, Ew.max(), 50, np.inf, np.inf, np.inf]), maxfev=30000)
            pred = _model(Ew, *popt)
        else:
            popt, pcov = curve_fit(lorentz, Ew, I, p0=[A0, E0, 4.0, np.median(I)],
                                   bounds=([0, Ew.min(), 0.3, -np.inf],
                                           [np.inf, Ew.max(), 50, np.inf]), maxfev=20000)
            pred = lorentz(Ew, *popt)
        G = abs(popt[2]); sG = float(np.sqrt(abs(pcov[2, 2])))
        R2 = 1 - np.sum((I - pred) ** 2) / (np.sum((I - I.mean()) ** 2) + 1e-30)
        return dict(G=G, sG=sG, E0=popt[1], tau=HBAR / G, stau=HBAR / G * sG / max(G, 1e-9),
                    R2=R2, popt=popt, model="lorentz", bg=bg, Em=Em)
    except Exception as e:
        print("  fit fail:", e); return None


def voigt(E, A, E0, sigma_G, gamma_L, c):
    """Voigt = Lorentzian(HWHM=gamma_L) ⊗ Gaussian(std=sigma_G), plus a constant background."""
    return A * voigt_profile(E - E0, sigma_G, gamma_L) + c


def fit_voigt(Ew, I, sigma_G, bg="const"):
    """Fix the Gaussian width sigma_G and let only the Lorentzian (lifetime) width vary in the fit.
    sigma_G = FWHM_Gauss / (2√(2ln2)). Returns the same interface as fit_lorentz:
    G = Lorentzian FWHM = 2·gamma_L (lifetime width), tau = HBAR/G.
    bg="const" adds a constant background; bg="quad" adds a2·t²+a1·t+a0 (t=E−mean(E), centered)
    for a joint peak+background fit on raw data. Gating (on G) stays consistent with fit_lorentz."""
    need = 7 if bg == "quad" else 5
    if Ew.size < need:
        print("  voigt skip: <%d points" % need); return None
    peak = float(I.max() - np.median(I)); E0 = Ew[int(np.argmax(I))]
    A0 = peak / max(float(voigt_profile(0.0, sigma_G, 2.0)), 1e-12)   # normalize to the peak-height scale
    Em = float(np.mean(Ew))
    try:
        if bg == "quad":
            # robust seeds against a sloped raw background (see fit_lorentz)
            base = I[0] + (I[-1] - I[0]) * (Ew - Ew[0]) / (Ew[-1] - Ew[0] + 1e-9)
            e0q = Ew[int(np.argmax(I - base))]
            A0q = float((I - base).max()) / max(float(voigt_profile(0.0, sigma_G, 2.0)), 1e-12)
            def _model(E, A, e0, g, a2, a1, a0):
                t = E - Em
                return A * voigt_profile(E - e0, sigma_G, g) + a2 * t * t + a1 * t + a0
            a1_0 = (I[-1] - I[0]) / (Ew[-1] - Ew[0] + 1e-9)
            popt, pcov = curve_fit(_model, Ew, I, p0=[A0q, e0q, 2.0, 0.0, a1_0, float(np.median(I))],
                                   bounds=([0, Ew.min(), 0.15, -np.inf, -np.inf, -np.inf],
                                           [np.inf, Ew.max(), 25, np.inf, np.inf, np.inf]), maxfev=30000)
            pred = _model(Ew, *popt)
        else:
            popt, pcov = curve_fit(lambda E, A, e0, g, c: voigt(E, A, e0, sigma_G, g, c),
                                   Ew, I, p0=[A0, E0, 2.0, np.median(I)],
                                   bounds=([0, Ew.min(), 0.15, -np.inf],      # gamma∈[0.15,25] → G∈[0.3,50]
                                           [np.inf, Ew.max(), 25, np.inf]), maxfev=20000)
            pred = voigt(Ew, popt[0], popt[1], sigma_G, popt[2], popt[3])
        gamma = abs(popt[2]); G = 2.0 * gamma                            # Lorentzian FWHM
        sG = 2.0 * float(np.sqrt(abs(pcov[2, 2])))
        R2 = 1 - np.sum((I - pred) ** 2) / (np.sum((I - I.mean()) ** 2) + 1e-30)
        return dict(G=G, sG=sG, E0=popt[1], tau=HBAR / G, stau=HBAR / G * sG / max(G, 1e-9),
                    R2=R2, popt=popt, model="voigt", sigma_G=float(sigma_G), bg=bg, Em=Em)
    except Exception as e:
        print("  voigt fit fail:", e); return None


def eval_fit(E, fit):
    """Reconstruct the fitted model curve at E from a fit dict — peak (Lorentzian or Voigt)
    plus its background (constant or quadratic). Handles the variable popt length."""
    E = np.asarray(E, float); popt = fit["popt"]
    if fit.get("model") == "voigt":
        peak = popt[0] * voigt_profile(E - popt[1], fit["sigma_G"], popt[2])
    else:
        G = popt[2]; peak = popt[0] * (G / 2) ** 2 / ((E - popt[1]) ** 2 + (G / 2) ** 2)
    if fit.get("bg") == "quad":
        t = E - fit["Em"]; return peak + popt[3] * t * t + popt[4] * t + popt[5]
    return peak + popt[3]


def eval_fit_with_G(E, fit, Gnew):
    """Same peak position/amplitude and same background, but the peak width forced to Γ=Gnew
    (used to draw the ensemble median-Γ comparison curve)."""
    E = np.asarray(E, float); popt = fit["popt"]
    if fit.get("model") == "voigt":
        peak = popt[0] * voigt_profile(E - popt[1], fit["sigma_G"], Gnew / 2.0)
    else:
        peak = popt[0] * (Gnew / 2) ** 2 / ((E - popt[1]) ** 2 + (Gnew / 2) ** 2)
    if fit.get("bg") == "quad":
        t = E - fit["Em"]; return peak + popt[3] * t * t + popt[4] * t + popt[5]
    return peak + popt[3]


# ---------------------------------------------------------------------------
# Ensemble estimation: apply small jitter + stretch to the hand-drawn box, build a family of neighboring boxes, fit each one,
# use the quality-gated median Γ as a robust point estimate, and the 16/84 percentiles for the uncertainty band.
# Key: never take the single max-R² box (that would overfit and introduce a ~12% biased shift); R² is used only as a lower-bound gate.
# Uses only HP-filtered data and a single Lorentzian.
# ---------------------------------------------------------------------------

# fixed keys of gate_breakdown (in SPEC order)
_GATE_KEYS = ("degenerate", "few_points", "fit_none", "G_railed",
              "peak_outside", "G_wider_than_box", "R2_floor", "sG_blewup")


def _empty_result(status, gate_breakdown=None, n=400, seed=0):
    """Empty result with all numeric fields set to nan and arrays left empty (for no_valid_fit / invalid input)."""
    if gate_breakdown is None:
        gate_breakdown = {k: 0 for k in _GATE_KEYS}
    nan = float("nan")
    return dict(
        tau=nan, G=nan, tau_lo=nan, tau_hi=nan, G_lo=nan, G_hi=nan,
        sigma_tau=nan, sigma_G_sys=nan, sigma_G_stat=nan,
        tau_argmaxR2=nan, G_argmaxR2=nan, R2_median=nan, tau_anchor=nan,
        n_kept=0, n=int(n), frac_kept=0.0,
        gate_breakdown=gate_breakdown,
        status=status, flags=["low_survivors"], seed=int(seed),
        tau_samples=np.array([]), G_samples=np.array([]), R2_samples=np.array([]),
    )


def _trimmed_mean(a, frac):
    """Symmetric trimmed mean: drop a fraction frac from each end."""
    a = np.sort(np.asarray(a, dtype=float))
    n = a.size
    if n == 0:
        return float("nan")
    k = int(np.floor(frac * n))
    if 2 * k >= n:
        return float(np.median(a))
    return float(a[k:n - k].mean())


def ensemble_tau(D, x, E, roi, n=400, seed=0, model="lorentz", gaussian_fwhm=None, bg="const"):
    """Apply small jitter + stretch to the hand-drawn ROI to build a family of neighboring boxes, fit each with a single Lorentzian,
    and after gating use the median Γ to give a robust τ=HBAR/median(Γ) and a 16/84 percentile band.

    D : HP-filtered LDOS array d[pos, E]; x : position (nm); E : energy (meV);
    roi=(x0,x1,E0,E1); n : family size; seed : fed to np.random.default_rng.
    Returns dict (keys see SPEC). The point estimate is always the median of the gated set, never the max-R².
    """
    # --- step 1: input validation ---
    x0, x1, E0, E1 = roi
    if x1 < x0:
        x0, x1 = x1, x0
    if E1 < E0:
        E0, E1 = E1, E0
    Wx = x1 - x0
    WE = E1 - E0
    cx = (x0 + x1) / 2.0
    cE = (E0 + E1) / 2.0
    x = np.asarray(x, dtype=float)
    E = np.asarray(E, dtype=float)
    if x.size < 2 or E.size < 2:
        return _empty_result("no_valid_fit", n=n, seed=seed)
    dx = abs(x[1] - x[0])
    dE = abs(E[1] - E[0])
    if Wx <= 0 or WE <= 0:
        return _empty_result("no_valid_fit", n=n, seed=seed)
    xlo, xhi = x.min(), x.max()
    Elo, Ehi = E.min(), E.max()

    _sigma_G = None                          # for Voigt: the fixed Gaussian standard deviation
    if model == "voigt":
        if not gaussian_fwhm or gaussian_fwhm <= 0:
            return _empty_result("no_valid_fit", n=n, seed=seed)   # Voigt requires a Gaussian width
        _sigma_G = gaussian_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # --- step 2: jitter/stretch scales (all in physical units) ---
    rng = np.random.default_rng(seed)
    sigma_x = max(0.06 * Wx, 1.5 * dx)
    sigma_E = max(0.06 * WE, 1.5 * dE)
    S = 0.10            # stretch half-amplitude ±10%
    cap = 2.5           # truncation (in units of sigma)
    min_w = max(0.5 * Wx, 4 * dx)
    min_h = max(0.5 * WE, 4 * dE)

    # --- step 3: pre-draw all perturbations (fixed order, so (roi,n,seed) is bit-level reproducible) ---
    n = int(n)
    sx = np.empty(n); sE = np.empty(n)
    jL = np.empty(n); jR = np.empty(n)
    jB = np.empty(n); jT = np.empty(n)
    # member 0 = unperturbed anchor
    sx[0] = 1.0; sE[0] = 1.0
    jL[0] = jR[0] = jB[0] = jT[0] = 0.0
    for m in range(1, n):
        sx[m] = rng.uniform(1 - S, 1 + S)
        sE[m] = rng.uniform(1 - S, 1 + S)
        jL[m] = np.clip(rng.normal(0.0, sigma_x), -cap * sigma_x, cap * sigma_x)
        jR[m] = np.clip(rng.normal(0.0, sigma_x), -cap * sigma_x, cap * sigma_x)
        jB[m] = np.clip(rng.normal(0.0, sigma_E), -cap * sigma_E, cap * sigma_E)
        jT[m] = np.clip(rng.normal(0.0, sigma_E), -cap * sigma_E, cap * sigma_E)

    gate = {k: 0 for k in _GATE_KEYS}
    G_s, sG_s, E0_s, tau_s, R2_s = [], [], [], [], []
    tau_anchor = float("nan")
    clip_count = 0

    for m in range(n):
        # --- step 4: build edges and clip to the data range ---
        x0m = cx - Wx * sx[m] / 2.0 + jL[m]
        x1m = cx + Wx * sx[m] / 2.0 + jR[m]
        E0m = cE - WE * sE[m] / 2.0 + jB[m]
        E1m = cE + WE * sE[m] / 2.0 + jT[m]
        clipped = False
        if x0m < xlo: x0m = xlo; clipped = True
        if x1m > xhi: x1m = xhi; clipped = True
        if E0m < Elo: E0m = Elo; clipped = True
        if E1m > Ehi: E1m = Ehi; clipped = True
        if clipped:
            clip_count += 1

        # --- step 5: geometric gate (degenerate), deterministic skip without resampling ---
        wm = x1m - x0m
        hm = E1m - E0m
        if wm < min_w or hm < min_h:
            gate["degenerate"] += 1
            continue

        # --- step 6: integrate ---
        Ew, I = integrate_roi(D, x, E, (x0m, x1m, E0m, E1m))
        if Ew.size < 8:
            gate["few_points"] += 1
            continue

        # --- step 7: fit (Lorentzian, or Voigt with fixed Gaussian) ---
        r = fit_lorentz(Ew, I, bg=bg) if model == "lorentz" else fit_voigt(Ew, I, _sigma_G, bg=bg)
        if r is None:
            gate["fit_none"] += 1
            continue

        # --- step 8: physical gate (direction-neutral, no ranking) ---
        G = r["G"]
        if not (0.5 < G < 49.5):
            gate["G_railed"] += 1
            continue
        marg = 0.05 * hm
        if not (E0m + marg <= r["E0"] <= E1m - marg):
            gate["peak_outside"] += 1
            continue
        if G > 0.9 * hm:
            gate["G_wider_than_box"] += 1
            continue
        if not (r["sG"] / max(G, 1e-9) < 1.0):
            gate["sG_blewup"] += 1
            continue

        # --- step 9: R² lower-bound gate (applied last, trims only true garbage, never used for ranking/selection) ---
        if r["R2"] < 0.90:
            gate["R2_floor"] += 1
            continue

        # --- step 10: survivors ---
        G_s.append(G); sG_s.append(r["sG"]); E0_s.append(r["E0"])
        tau_s.append(r["tau"]); R2_s.append(r["R2"])
        if m == 0:
            tau_anchor = r["tau"]

    # --- step 11: determine status before statistics ---
    n_kept = len(G_s)
    threshold = max(0.30 * n, 30)
    if n_kept == 0:
        res = _empty_result("no_valid_fit", gate_breakdown=gate, n=n, seed=seed)
        return res
    status = "ok" if n_kept >= threshold else "unreliable"

    G_s = np.asarray(G_s, dtype=float)
    sG_s = np.asarray(sG_s, dtype=float)
    tau_s = np.asarray(tau_s, dtype=float)
    R2_s = np.asarray(R2_s, dtype=float)

    # --- step 14 (multimodal handled before computing the median, so it can restrict by cluster) ---
    flags = []
    MAD_tau = float(np.median(np.abs(tau_s - np.median(tau_s)))) if tau_s.size else 0.0
    cluster_mask = np.ones(tau_s.size, dtype=bool)
    if tau_s.size >= 4 and MAD_tau > 0:
        order = np.argsort(tau_s)
        ts = tau_s[order]
        gaps = np.diff(ts)
        gap_thresh = 3 * 1.4826 * MAD_tau
        big = np.where(gaps > gap_thresh)[0]
        if big.size > 0:
            # split at the first large gap into two clusters, requiring both clusters to be non-trivial
            split = big[0]
            left_idx = order[:split + 1]
            right_idx = order[split + 1:]
            if left_idx.size >= 2 and right_idx.size >= 2:
                flags.append("multimodal")
                # pick the cluster containing tau_anchor; otherwise the cluster closest to G_med
                G_med_all = float(np.median(G_s))
                tau_med_all = HBAR / G_med_all
                left_lo, left_hi = ts[0], ts[split]
                right_lo, right_hi = ts[split + 1], ts[-1]
                pick_left = None
                if np.isfinite(tau_anchor):
                    if left_lo <= tau_anchor <= left_hi:
                        pick_left = True
                    elif right_lo <= tau_anchor <= right_hi:
                        pick_left = False
                if pick_left is None:
                    # the cluster center closest to tau_med_all
                    dl = abs(np.median(tau_s[left_idx]) - tau_med_all)
                    dr = abs(np.median(tau_s[right_idx]) - tau_med_all)
                    pick_left = dl <= dr
                cluster_mask = np.zeros(tau_s.size, dtype=bool)
                cluster_mask[left_idx if pick_left else right_idx] = True

    G_c = G_s[cluster_mask]
    sG_c = sG_s[cluster_mask]
    tau_c = tau_s[cluster_mask]
    R2_c = R2_s[cluster_mask]

    # --- step 12: statistics (computed on the directly-fitted variable Γ; median and HBAR/Γ commute exactly) ---
    G_med = float(np.median(G_c))
    G16 = float(np.percentile(G_c, 16))
    G84 = float(np.percentile(G_c, 84))
    tau = HBAR / G_med
    tau_lo = HBAR / G84          # large Γ -> small τ
    tau_hi = HBAR / G16
    sigma_G_sys = (G84 - G16) / 2.0
    sigma_G_stat = float(np.median(sG_c))
    sigma_tau = tau * np.sqrt(sigma_G_sys ** 2 + sigma_G_stat ** 2) / G_med
    R2_median = float(np.median(R2_c))

    # --- step 13: transparency fields (only to show the overfitting gap, never to replace the headline) ---
    imax = int(np.argmax(R2_c))
    G_argmaxR2 = float(G_c[imax])
    tau_argmaxR2 = float(tau_c[imax])

    # --- step 14: remaining flags ---
    G_med_full = float(np.median(G_s))
    MAD_G = float(np.median(np.abs(G_s - G_med_full)))
    if abs(G_med_full - _trimmed_mean(G_s, 0.20)) > 1.4826 * MAD_G:
        flags.append("asymmetric")
    if clip_count > 0.5 * n:
        flags.append("near_edge")
    if status != "ok":
        flags.append("low_survivors")

    return dict(
        tau=float(tau), G=G_med, tau_lo=float(tau_lo), tau_hi=float(tau_hi),
        G_lo=G16, G_hi=G84,
        sigma_tau=float(sigma_tau), sigma_G_sys=float(sigma_G_sys),
        sigma_G_stat=float(sigma_G_stat),
        tau_argmaxR2=tau_argmaxR2, G_argmaxR2=G_argmaxR2,
        R2_median=R2_median, tau_anchor=float(tau_anchor),
        n_kept=int(n_kept), n=int(n), frac_kept=float(n_kept) / n,
        gate_breakdown=gate, status=status, flags=flags, seed=int(seed),
        tau_samples=tau_s, G_samples=G_s, R2_samples=R2_s,
    )


def select_rois(D, x, E, filt, title):
    """Pop up a figure, drag the mouse to draw several ROIs (lime box + number), return the ROI list on close."""
    rois = []
    fig, ax = plt.subplots(figsize=(9, 8))
    imshow_map(ax, D, x, E, filt, title)

    def onselect(ec, er):
        x0, x1 = sorted([ec.xdata, er.xdata]); E0, E1 = sorted([ec.ydata, er.ydata])
        rois.append((x0, x1, E0, E1))
        ax.add_patch(Rectangle((x0, E0), x1 - x0, E1 - E0, fill=False, ec="lime", lw=2))
        ax.text(x0, E1, str(len(rois)), color="lime", fontsize=13, fontweight="bold", va="bottom")
        fig.canvas.draw_idle()
        print("  ROI %d: x[%.0f,%.0f]nm  E[%.0f,%.0f]meV" % (len(rois), x0, x1, E0, E1))

    # RS must stay referenced while plt.show() blocks (a local variable is enough), otherwise it stops working.
    RS = RectangleSelector(ax, onselect, useblit=True, button=[1],
                           minspanx=2, minspany=2, spancoords="data", interactive=True)
    plt.show()
    del RS
    return rois


def process_axis(D, x, E, rois, ensemble=True):
    """Integrate + fit (+ ensemble estimate) each ROI of one axis in turn.
    Returns [{i, roi, Ew, I, fit, ens}]. When ensemble=True (HP data only),
    each ROI additionally computes ensemble_tau; wrapped in try/except so a single bad ROI does not affect the whole batch.
    """
    out = []
    for i, roi in enumerate(rois, 1):
        Ew, I = integrate_roi(D, x, E, roi)
        fit = fit_lorentz(Ew, I)
        ens = None
        if ensemble:
            try:
                ens = ensemble_tau(D, x, E, roi, n=400, seed=0)
            except Exception as e:
                print("  ensemble fail (ROI %d):" % i, e); ens = None
        out.append(dict(i=i, roi=roi, Ew=Ew, I=I, fit=fit, ens=ens))
    return out


def grid_dims(n):
    if n <= 1:
        return 1, 1
    ncols = 2 if n <= 4 else (3 if n <= 9 else 4)
    return int(np.ceil(n / ncols)), ncols


def save_fits_figure(results, axis, data, filt):
    """One large figure: fits of all ROIs of this axis, one subplot per ROI (E on the x-axis, ∫LDOS on the y-axis)."""
    n = len(results)
    if n == 0:
        print("  (%s has no ROI, skipping the fits figure)" % axis); return None
    nrows, ncols = grid_dims(n)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                            constrained_layout=True, squeeze=False)
    for k, res in enumerate(results):
        ax = axs[k // ncols][k % ncols]
        Ew, I, r, roi = res["Ew"], res["I"], res["fit"], res["roi"]
        ens = res.get("ens")
        ax.plot(Ew, I, "k.", ms=4, label="∫ over position")
        if r:
            Ef = np.linspace(Ew.min(), Ew.max(), 300)
            ax.plot(Ef, eval_fit(Ef, r), "r-", lw=1.7, label="single-box fit")
            # same fitted model but with the ensemble median Γ (peak width replaced), thin dashed for comparison
            if ens is not None and ens.get("status") != "no_valid_fit" \
                    and np.isfinite(ens.get("G", float("nan"))):
                ax.plot(Ef, eval_fit_with_G(Ef, r, ens["G"]), "--", color="0.35",
                        lw=1.3, label="median-Γ (ensemble)")
            ax.set_title("ROI %d   Γ=%.1f±%.1f meV\nτ=%.0f±%.0f fs   R²=%.2f"
                         % (res["i"], r["G"], r["sG"], r["tau"], r["stau"], r["R2"]),
                         fontsize=9)
        else:
            ax.set_title("ROI %d   fit fail" % res["i"], fontsize=9, color="red")
        ax.text(0.02, 0.97, "x[%.0f,%.0f]nm\nE[%.0f,%.0f]meV" % roi,
                transform=ax.transAxes, fontsize=7, va="top", color="0.4")
        ax.set_xlabel("E (meV)"); ax.set_ylabel("∫ LDOS dx (a.u.)")
        # ensemble result: second-line title (color by status) + small histogram in the top-right corner
        if ens is not None:
            st = ens.get("status", "no_valid_fit")
            col = {"ok": "0.45", "unreliable": "darkorange"}.get(st, "red")
            if st == "no_valid_fit" or not np.isfinite(ens.get("tau", float("nan"))):
                ax.text(0.5, -0.30, "τ_ens: unreliable (n_kept=%d, %s)"
                        % (ens.get("n_kept", 0), st),
                        transform=ax.transAxes, fontsize=8, color=col,
                        ha="center", va="top")
            else:
                ax.text(0.5, -0.30,
                        "τ_ens=%.0f ±%.0f fs  (jitter[%.0f,%.0f], n=%d, %s)"
                        % (ens["tau"], ens["sigma_tau"], ens["tau_lo"],
                           ens["tau_hi"], ens["n_kept"], st),
                        transform=ax.transAxes, fontsize=8, color=col,
                        ha="center", va="top")
                ts = np.asarray(ens.get("tau_samples", []))
                if ts.size >= 5:
                    iax = ax.inset_axes([0.62, 0.60, 0.35, 0.35])
                    iax.hist(ts, bins=20, color="0.6")
                    iax.axvline(ens["tau"], color="r", lw=1.2)          # median
                    iax.axvline(ens["tau_lo"], color="r", ls=":", lw=0.8)  # 16% jitter
                    iax.axvline(ens["tau_hi"], color="r", ls=":", lw=0.8)  # 84% jitter
                    iax.set_xticks([]); iax.set_yticks([])
                    iax.set_title("τ jitter dist", fontsize=6, color="0.4")
    for k in range(n, nrows * ncols):                # hide extra empty cells
        axs[k // ncols][k % ncols].axis("off")
    fig.suptitle("%s  %s  fits  (filter=%s)  —  %d ROI" % (data.upper(), axis, filt, n))
    p = os.path.join(HERE, "fits_%s_%s_%s.png" % (axis, data, filt))
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    return p


def save_overview(maps, results_by_axis, data, filt):
    """One figure: long (major) / short (minor) axis LDOS side by side, with all ROIs drawn and τ labeled in green inside each box."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, axis in zip(axs, ("major", "minor")):
        D, x, E = maps[axis]
        imshow_map(ax, D, x, E, filt, "%s  %s (%s)" % (data.upper(), axis, filt))
        for res in results_by_axis[axis]:
            x0, x1, E0, E1 = res["roi"]; r = res["fit"]
            ens = res.get("ens")
            ax.add_patch(Rectangle((x0, E0), x1 - x0, E1 - E0, fill=False, ec="lime", lw=1.8))
            # prefer the ensemble estimate (median + asymmetric band); no_valid_fit shows a red note;
            # fall back to the single-box τ when there is no ens.
            tau_col = "lime"
            if ens is not None and ens.get("status") != "no_valid_fit" \
                    and np.isfinite(ens.get("tau", float("nan"))):
                tau_txt = ("τ=%.0f±%.0f fs"
                           % (ens["tau"], ens["sigma_tau"]))
            elif ens is not None:
                tau_txt = "τ: unreliable"; tau_col = "red"
            else:
                tau_txt = ("τ=%.0f fs" % r["tau"]) if r else "fit fail"
            ax.text((x0 + x1) / 2, (E0 + E1) / 2, tau_txt, color=tau_col, fontsize=8,
                    ha="center", va="center", fontweight="bold")
            ax.text(x0, E1, str(res["i"]), color="lime", fontsize=10, va="bottom",
                    fontweight="bold")
    fig.suptitle("LDOS + ROI + τ   (data=%s, filter=%s)" % (data, filt))
    p = os.path.join(HERE, "ldos_overview_%s_%s.png" % (data, filt))
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    return p


def save_tau_summary(results_by_axis, data, filt):
    """Summary error-bar figure: the τ of each ROI and its uncertainty, all peaks at a glance.
    point = median-Γ ensemble estimate; thick error bar = ±σ_τ (box-selection ⊕ fit, orthogonal, well calibrated);
    light thin band = the 16/84 band of pure box-selection jitter (viewed alone it underestimates the total error, only for comparison).
    status=unreliable points are colored orange; no_valid_fit is annotated in red at the bottom.
    """
    axes_with = [ax for ax in ("major", "minor") if results_by_axis.get(ax)]
    if not axes_with:
        return None
    fig, axs = plt.subplots(1, len(axes_with),
                            figsize=(1.5 + 5.5 * len(axes_with), 5),
                            squeeze=False, constrained_layout=True)
    for col, axis in enumerate(axes_with):
        ax = axs[0][col]
        xs, tau, sig, blo, bhi, orange, labels, bad = [], [], [], [], [], [], [], []
        for j, res in enumerate(results_by_axis[axis]):
            ens = res.get("ens"); i = res["i"]
            ok = (ens is not None and ens.get("status") != "no_valid_fit"
                  and np.isfinite(ens.get("tau", float("nan"))))
            if not ok:
                bad.append(i); continue
            xs.append(j); tau.append(ens["tau"]); sig.append(ens["sigma_tau"])
            blo.append(ens["tau"] - ens["tau_lo"]); bhi.append(ens["tau_hi"] - ens["tau"])
            orange.append(ens.get("status") == "unreliable"); labels.append(str(i))
        if len(xs):
            xa = np.array(xs, dtype=float); ta = np.array(tau)
            ax.errorbar(xa, ta, yerr=[blo, bhi], fmt="none", ecolor="0.78",
                        elinewidth=5, capsize=0, zorder=1, label="box-jitter 16/84")
            ax.errorbar(xa, ta, yerr=sig, fmt="o", ms=5, mfc="C0", mec="k",
                        ecolor="0.2", elinewidth=1.3, capsize=3, zorder=2,
                        label="±σ_τ (sys⊕stat)")
            om = np.array(orange)
            if om.any():
                ax.plot(xa[om], ta[om], "o", mfc="darkorange", mec="k", ms=6,
                        zorder=3, label="unreliable")
            ax.set_xticks(xa); ax.set_xticklabels(labels)
            if col == 0:
                ax.legend(fontsize=7, loc="best")
        if bad:
            ax.text(0.5, 0.02, "no valid fit: ROI " + ",".join(map(str, bad)),
                    transform=ax.transAxes, color="red", fontsize=8,
                    ha="center", va="bottom")
        ax.set_xlabel("ROI"); ax.set_ylabel("τ (fs)")
        ax.set_title("%s  %s (%s)" % (data.upper(), axis, filt), fontsize=10)
    fig.suptitle("Trapping time per ROI — point=median-Γ ensemble, bar=±σ_τ, "
                 "shaded=box-jitter 16/84   (data=%s, filter=%s)" % (data, filt))
    p = os.path.join(HERE, "tau_summary_%s_%s.png" % (data, filt))
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    return p


def write_csv(rows):
    if not rows:
        print("No fit results to write (all failed?)."); return
    csvp = os.path.join(HERE, "roi_results.csv")
    head = ["roi", "axis", "filter", "data", "x0", "x1", "E0", "E1",
            "E0_fit", "Gamma_meV", "tau_fs", "sigtau_fs", "R2",
            "tau_ens_fs", "tau_lo_fs", "tau_hi_fs", "sigtau_ens_fs",
            "Gamma_ens_meV", "n_kept", "frac_kept", "tau_argmaxR2_fs",
            "status_ens"]
    # old header column count mismatch → archive, to avoid ragged rows (do not delete data, rename to keep it)
    if os.path.exists(csvp):
        with open(csvp) as f:
            first = f.readline().rstrip("\n")
        if first and first.count(",") + 1 != len(head):
            bak = os.path.join(HERE, "roi_results_legacy.csv"); k = 1
            while os.path.exists(bak):
                k += 1; bak = os.path.join(HERE, "roi_results_legacy_%d.csv" % k)
            os.rename(csvp, bak)
            print("old CSV column count (%d) does not match the new header (%d), archived as %s"
                  % (first.count(",") + 1, len(head), bak))
    new = not os.path.exists(csvp)
    with open(csvp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(head)
        w.writerows(rows)
    print("\nresults ->", csvp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["exp", "sim", "ext2"], default="exp",
                    help="choose dataset: exp/sim, or ext2 (extended_sample2: reads ext2_major.txt/ext2_minor.txt)")
    ap.add_argument("--filter", choices=["raw", "hp"], default="hp", help="whether to high-pass filter")
    ap.add_argument("--tag", default="",
                    help="output tag: when non-empty, all outputs are renamed {data}_{tag} (fits/overview/tau_summary figures + "
                         "the data column of roi_results.csv). Used for a second round of box-selection on the same dataset (e.g. --tag center "
                         "for center boxes) without overwriting the previous round. Input data is still read according to --data.")
    args = ap.parse_args()

    # input is always read per --data ({data}_{axis}.txt); outputs/CSV are distinguished by label, so multiple box-selection rounds do not overwrite each other.
    label = args.data + ("_" + args.tag if args.tag else "")

    # preview (EXP|SIM comparison, both axes) to help you decide --filter.
    # only exp/sim have a counterpart to compare; a standalone dataset (e.g. ext2) has no sim counterpart → skip,
    # decide --filter directly by looking at its own extended_sample2/extended_sample2_*_rawVSfilt.png.
    if args.data in ("exp", "sim"):
        save_previews("major"); save_previews("minor")

    # load only the selected dataset's major + minor
    maps = load_dataset(args.data, args.filter)

    # sequential box-selection: first major then close window, then minor then close window
    rois_by_axis = {}
    for axis in ("major", "minor"):
        D, x, E = maps[axis]
        print("\n=== draw %s ROIs (data=%s, filter=%s) — drag boxes, multiple allowed, close window when done ==="
              % (axis, label, args.filter))
        rois_by_axis[axis] = select_rois(
            D, x, E, args.filter,
            "%s %s (%s) — drag boxes to draw ROIs (multiple allowed), close window when done → then draw the other axis"
            % (label.upper(), axis, args.filter))
        print("  %s has %d ROIs in total" % (axis, len(rois_by_axis[axis])))

    if not any(rois_by_axis.values()):
        print("No ROI drawn, exiting."); return

    # integrate + fit per axis (ensemble estimate only for HP data, hard constraint)
    do_ens = (args.filter == "hp")
    results_by_axis = {axis: process_axis(*maps[axis], rois_by_axis[axis],
                                          ensemble=do_ens)
                       for axis in ("major", "minor")}

    # three figures
    save_fits_figure(results_by_axis["major"], "major", label, args.filter)
    save_fits_figure(results_by_axis["minor"], "minor", label, args.filter)
    save_overview(maps, results_by_axis, label, args.filter)
    save_tau_summary(results_by_axis, label, args.filter)

    # csv + terminal summary
    def _rnd(v, nd):
        """nan-tolerant rounding: nan -> '' writes empty."""
        try:
            if v is None or not np.isfinite(v):
                return ""
        except TypeError:
            return ""
        return round(v, nd) if nd > 0 else round(v)

    rows = []
    for axis in ("major", "minor"):
        for res in results_by_axis[axis]:
            r = res["fit"]
            if not r:
                continue
            roi = res["roi"]
            ens = res.get("ens")
            row = [res["i"], axis, args.filter, label,
                   round(roi[0]), round(roi[1]), round(roi[2]), round(roi[3]),
                   round(r["E0"], 1), round(r["G"], 2), round(r["tau"], 0),
                   round(r["stau"], 0), round(r["R2"], 3)]
            if ens is not None:
                row += [_rnd(ens["tau"], 0), _rnd(ens["tau_lo"], 0),
                        _rnd(ens["tau_hi"], 0), _rnd(ens["sigma_tau"], 0),
                        _rnd(ens["G"], 2), ens["n_kept"], _rnd(ens["frac_kept"], 2),
                        _rnd(ens["tau_argmaxR2"], 0), ens["status"]]
            else:
                row += ["", "", "", "", "", "", "", "", ""]
            rows.append(row)
    write_csv(rows)
    for axis in ("major", "minor"):
        for res in results_by_axis[axis]:
            r = res["fit"]
            ens = res.get("ens")
            if r:
                line = ("  %-5s ROI%d: Γ=%.1f meV  τ=%.0f fs  R²=%.2f"
                        % (axis, res["i"], r["G"], r["tau"], r["R2"]))
                if ens is not None and ens.get("status") != "no_valid_fit" \
                        and np.isfinite(ens.get("tau", float("nan"))):
                    line += ("   τ_ens=%.0f [%.0f,%.0f] fs (%s)"
                             % (ens["tau"], ens["tau_lo"], ens["tau_hi"],
                                ens["status"]))
                elif ens is not None:
                    line += "   τ_ens: unreliable (%s)" % ens.get("status")
                print(line)
            else:
                print("  %-5s ROI%d: fit fail" % (axis, res["i"]))


if __name__ == "__main__":
    main()
