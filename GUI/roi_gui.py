#!/usr/bin/env python3
"""GUI/roi_gui.py — Interactive panel: lifetime (ROI-τ) fitting + potential-picking quadratic fit, switchable between two modes.

Placed in the GUI/ subfolder; via sys.path it reuses the parent directory's trapping_roi (integration/fitting/ensemble/plotting/
load_any/Voigt) and fit_potential (quadratic fit + errors + joint). On startup it loads major/minor from
GUI/sample_data/ by default; you can also import major/minor separately by path from the panel (txt / pxp auto-detected).

Panel options:
  · filter: hp (default, high-pass envelope subtraction σ=6.5nm×8.5meV) / raw (original dI/dV, unfiltered).
  · fit: Lorentzian (default) / Voigt (fixed Gaussian width, fits only the lifetime Lorentzian width; enter the Gaussian FWHM in the box).
    —— To get a "clean lifetime with thermal + instrumental broadening removed": choose raw + Voigt, and set the Gaussian FWHM to √((3.53kT)²+(√3·V_mod)²).
Two modes (mode radio):
  · ROI-τ: left-drag a box → live integrated estimate τ=ħ/Γ (±σ_τ); right-click inside a box to delete it.
  · potential: left-click points → live fit V(s)=a(s-s0)²+V0 (with errors); right-click removes the nearest point; major/minor combine into V(x,y).

Export: each Save creates a timestamped folder under the output root directory (csv + figures), without overwriting. The root defaults to
GUI/output; it can be set via --outdir or the panel's out box / `...` button. Axes with uncalibrated coordinates cannot be exported.

Runs locally and interactively (requires a graphical display). Panel text is in English to avoid missing-glyph boxes with Chinese fonts; terminal messages are in English.
Usage:
  python roi_gui.py
  python roi_gui.py --major <path> --minor <path> [--outdir <dir>]
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Button, RadioButtons, TextBox

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import trapping_roi as tr          # noqa: E402
import fit_potential as fp         # noqa: E402

LIVE_N = 400                       # number of ensemble members per box (~0.6s)
SAMPLE_DIR = os.path.join(_HERE, "sample_data")
OUTPUT_DIR = os.path.join(_HERE, "output")
DEFAULT_GAUSS = 2.94               # default Gaussian FWHM (meV): T=4.5K, V_mod=1.5mV peak


def _cmap_clim(d):
    """Auto-select colormap by data sign: bipolar (hp, background subtracted) → symmetric RdBu_r; strictly positive (raw) → inferno."""
    p1 = float(np.percentile(d, 1)); p99 = float(np.percentile(d, 99))
    if p1 < 0 and p99 > 0 and abs(p1) > 0.2 * abs(p99):
        v = max(abs(p1), abs(p99))
        return "RdBu_r", -v, v
    return "inferno", p1, p99


def _filt_tag(d):
    return "hp" if _cmap_clim(d)[0] == "RdBu_r" else "raw"


class Panel:
    def __init__(self, major_path=None, minor_path=None, outdir=None):
        self.mode = "roi"                        # "roi" | "pot"
        self.filter_on = True                    # True=hp, False=raw
        self.fitmode = "lorentz"                 # "lorentz" | "voigt"
        self.bg_mode = "const"                   # "const" | "quad" (joint quadratic background, for raw)
        self.gauss_fwhm = DEFAULT_GAUSS          # used by Voigt: Gaussian FWHM (meV)
        self._init_outdir = outdir or OUTPUT_DIR
        self.axis = "major"
        self.maps_raw = {}                       # axis -> (d_raw, x, E)  original import
        self.maps_hp = {}                        # axis -> (d_hp,  x, E)  after hp filter
        self.src = {}
        self.cal = {}
        self.boxes = {"major": [], "minor": []}  # per box: dict(i, roi, Ew, I, fit, ens)
        self.points = {"major": [], "minor": []}
        self.pot_fit = {"major": None, "minor": None}
        self.overlay = []
        self._load(("major", major_path or self._default("major")),
                   ("minor", minor_path or self._default("minor")))
        self._build()

    # ---------- data ----------
    def _default(self, axis):
        if not os.path.isdir(SAMPLE_DIR):
            return None
        cands = [f for f in sorted(os.listdir(SAMPLE_DIR))
                 if f.lower().endswith((".txt", ".pxp")) and axis in f.lower()]
        pref = [f for f in cands if f.startswith("ext2_")]
        pick = pref or cands
        return os.path.join(SAMPLE_DIR, pick[0]) if pick else None

    def _load(self, *axis_path_pairs):
        for axis, path in axis_path_pairs:
            if not path:                                    # no default data: placeholder empty plot
                z = (np.zeros((2, 2)), np.array([0.0, 1.0]), np.array([0.0, 1.0]))
                self.maps_raw[axis] = z; self.maps_hp[axis] = z
                self.src[axis] = "(no data - use Load)"; self.cal[axis] = False
                self.boxes[axis] = []; self.points[axis] = []; self.pot_fit[axis] = None
                print("  %-5s: no default data, please import with Load" % axis)
                continue
            ldos, x, E, meta = tr.load_any(path, return_meta=True)
            self.maps_raw[axis] = (ldos, x, E)                       # original
            self.maps_hp[axis] = (tr.hp_filter(ldos, x, E), x, E)    # high-pass
            self.src[axis] = os.path.basename(path)
            self.cal[axis] = bool(meta["x_cal"] and meta["E_cal"])
            self.boxes[axis] = []; self.points[axis] = []; self.pot_fit[axis] = None
            print("  loaded %-5s <- %s  %s  x[%.0f,%.0f] E[%.0f,%.0f] cal=%s"
                  % (axis, self.src[axis], ldos.shape, x.min(), x.max(), E.min(), E.max(),
                     self.cal[axis]))

    def _D(self, axis):
        """This axis's data under the current filter (hp or raw)."""
        return self.maps_hp[axis] if self.filter_on else self.maps_raw[axis]

    def _sigmaG(self):
        return self.gauss_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # ---------- fitting ----------
    def _fit_roi_axis(self, axis, roi):
        d, x, E = self._D(axis)
        Ew, I = tr.integrate_roi(d, x, E, roi)
        bg = self.bg_mode
        if self.fitmode == "voigt":
            fit = tr.fit_voigt(Ew, I, self._sigmaG(), bg=bg)
            gfw = self.gauss_fwhm
        else:
            fit = tr.fit_lorentz(Ew, I, bg=bg); gfw = None
        try:
            ens = tr.ensemble_tau(d, x, E, roi, n=LIVE_N, seed=0,
                                  model=self.fitmode, gaussian_fwhm=gfw, bg=bg)
        except Exception as e:
            print("  ensemble fail:", e); ens = None
        return dict(roi=roi, Ew=Ew, I=I, fit=fit, ens=ens)

    def _fit_roi(self, roi):
        return self._fit_roi_axis(self.axis, roi)

    def add_box(self, roi):
        x0, x1 = sorted(roi[:2]); E0, E1 = sorted(roi[2:])
        b = self._fit_roi((x0, x1, E0, E1))
        self.boxes[self.axis].append(b); self._renumber()
        return b

    def _renumber(self):
        for i, b in enumerate(self.boxes[self.axis], 1):
            b["i"] = i

    def _recompute_all(self):
        """filter / fit / background / Gaussian width changed → recompute all boxes with the new settings."""
        n = sum(len(self.boxes[a]) for a in ("major", "minor"))
        if n == 0:
            return
        self.info.set_text("recomputing %d boxes ..." % n)
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        for axis in ("major", "minor"):
            for b in self.boxes[axis]:
                nb = self._fit_roi_axis(axis, b["roi"]); i = b["i"]
                b.clear(); b.update(nb); b["i"] = i

    def _fit_pot(self):
        pts = self.points[self.axis]
        self.pot_fit[self.axis] = (fp.fit_quadratic([p[0] for p in pts], [p[1] for p in pts])
                                   if len(pts) >= 3 else None)

    # ---------- base image / overlay ----------
    def _update_image(self):
        d, x, E = self._D(self.axis)
        cmap, lo, hi = _cmap_clim(d)
        self.im.set_data(d.T); self.im.set_cmap(cmap); self.im.set_clim(lo, hi)
        self.im.set_extent([x.min(), x.max(), E.min(), E.max()])
        self.ax.set_xlim(x.min(), x.max()); self.ax.set_ylim(E.min(), E.max())
        self.ax.set_autoscale_on(False)
        mtag = "ROI-tau" if self.mode == "roi" else "Potential"
        fkey = "hp" if self.filter_on else "raw"
        fit = self.fitmode if self.mode == "roi" else "-"
        bg = self.bg_mode if self.mode == "roi" else "-"
        cal = "" if self.cal[self.axis] else "  [UNCALIBRATED]"
        self.ax.set_title("%s  [%s | %s | %s+%s]  src=%s%s"
                          % (self.axis, mtag, fkey, fit, bg, self.src[self.axis], cal), fontsize=9)

    def _clear_overlay(self):
        for a in self.overlay:
            try:
                a.remove()
            except Exception:
                pass
        self.overlay = []

    def _redraw(self):
        self._clear_overlay()
        (self._draw_boxes if self.mode == "roi" else self._draw_points)()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _draw_boxes(self):
        for b in self.boxes[self.axis]:
            x0, x1, E0, E1 = b["roi"]
            r = Rectangle((x0, E0), x1 - x0, E1 - E0, fill=False, ec="lime", lw=1.8)
            self.ax.add_patch(r); self.overlay.append(r)
            ens = b["ens"]
            if ens is not None and ens.get("status") != "no_valid_fit" \
                    and np.isfinite(ens.get("tau", np.nan)):
                lab = "%d: t=%.0f±%.0f" % (b["i"], ens["tau"], ens["sigma_tau"])
                col = "lime" if ens["status"] == "ok" else "orange"
            else:
                lab = "%d: no fit" % b["i"]; col = "red"
            t = self.ax.text((x0 + x1) / 2, (E0 + E1) / 2, lab, color=col, fontsize=8,
                             ha="center", va="center", fontweight="bold",
                             bbox=dict(boxstyle="round", fc="black", alpha=0.45, ec="none"))
            self.overlay.append(t)

    def _draw_points(self):
        pts = self.points[self.axis]
        _, x, E = self._D(self.axis)
        if pts:
            self.overlay.append(self.ax.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                                             mfc="cyan", mec="k", ms=6, zorder=5)[0])
        f = self.pot_fit[self.axis]
        if f is not None:
            xf = np.linspace(x.min(), x.max(), 300)
            yf = f["a"] * xf ** 2 + f["b"] * xf + f["c"]
            self.overlay.append(self.ax.plot(xf, yf, "-", color="lime", lw=2.0, zorder=4)[0])
            if np.isfinite(f["x0"]):
                self.overlay.append(self.ax.plot([f["x0"]], [f["V0"]], "*", color="yellow",
                                                 mec="k", ms=15, zorder=6)[0])
        self.ax.set_xlim(x.min(), x.max()); self.ax.set_ylim(E.min(), E.max())

    def _update_info(self):
        fkey = "hp" if self.filter_on else "raw"
        L = ["mode:%s  axis:%s" % (self.mode, self.axis),
             "filter:%s  fit:%s  bg:%s" % (fkey, self.fitmode, self.bg_mode),
             "src :%s" % self.src[self.axis]]
        if self.fitmode == "voigt":
            L.append("gauss FWHM=%.2f meV" % self.gauss_fwhm)
            if self.filter_on:
                L.append("(!) Voigt on hp: use raw")
        if not self.cal[self.axis]:
            L.append("** UNCALIBRATED **")
        L.append("")
        if self.mode == "roi":
            L.append("boxes: %d" % len(self.boxes[self.axis]))
            for b in self.boxes[self.axis]:
                ens = b["ens"]
                if ens is not None and np.isfinite(ens.get("tau", np.nan)) \
                        and ens.get("status") != "no_valid_fit":
                    s = "t=%.0f±%.0f %s" % (ens["tau"], ens["sigma_tau"], ens["status"])
                else:
                    s = "no fit"
                L.append("#%d %s" % (b["i"], s))
        else:
            f = self.pot_fit[self.axis]
            L.append("points: %d" % len(self.points[self.axis]))
            if f is not None:
                L += ["V(s)=a(s-s0)^2+V0",
                      " a =%.4g +/-%.2g" % (f["a"], f.get("sa", float("nan"))),
                      " s0=%.2f +/-%.2f" % (f["x0"], f.get("sx0", float("nan"))),
                      " V0=%.2f +/-%.2f" % (f["V0"], f.get("sV0", float("nan"))),
                      " R2=%.4f" % f["R2"]]
            else:
                L.append("(need >=3 pts)")
            fj, fn = self.pot_fit["major"], self.pot_fit["minor"]
            if fj is not None and fn is not None:
                cj = fp.combine_axes(fj, fn)
                L += ["", "JOINT V(x,y):",
                      " a_x=%.4g a_y=%.4g" % (cj["ax"], cj["ay"]),
                      " aniso a_x/a_y=%.2f" % cj["aniso_ratio"],
                      " V0=%.2f (dV0=%.2f)" % (cj["V0"], cj["V0_diff"])]
        self.info.set_text("\n".join(L))

    # ---------- events ----------
    def _set_rs(self, on):
        try:
            self.rs.set_active(on)
        except Exception:
            try:
                self.rs.active = on
            except Exception:
                pass

    def _on_mode(self, label):
        self.mode = "roi" if label.startswith("ROI") else "pot"
        self._set_rs(self.mode == "roi")
        self._update_image(); self._redraw()

    def _on_axis(self, label):
        self.axis = label
        self._update_image(); self._redraw()

    def _on_filter(self, label):
        self.filter_on = (label == "hp")
        self._update_image(); self._recompute_all(); self._redraw()

    def _on_fit(self, label):
        self.fitmode = "voigt" if label.lower().startswith("v") else "lorentz"
        self._update_image(); self._recompute_all(); self._redraw()

    def _on_bg(self, label):
        self.bg_mode = "quad" if label.lower().startswith("q") else "const"
        self._update_image(); self._recompute_all(); self._redraw()

    def _on_gauss(self, text):
        try:
            v = float(text)
            if v > 0:
                self.gauss_fwhm = v
                if self.fitmode == "voigt":
                    self._recompute_all()
                self._redraw()
        except ValueError:
            pass

    def _on_select(self, ec, er):
        if None in (ec.xdata, er.xdata, ec.ydata, er.ydata):
            return
        self.ax.set_title("computing tau ..."); self.fig.canvas.draw(); self.fig.canvas.flush_events()
        b = self.add_box((ec.xdata, er.xdata, ec.ydata, er.ydata))
        lab = "no fit" if (b["ens"] is None or not np.isfinite(b["ens"].get("tau", np.nan))) \
            else "tau=%.0f" % b["ens"]["tau"]
        print("  + roi %s #%d  %s" % (self.axis, b["i"], lab))
        self._update_image(); self._redraw()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.mode == "roi":
            if event.button == 3:
                cand = []
                for b in self.boxes[self.axis]:
                    x0, x1, E0, E1 = b["roi"]
                    if x0 <= event.xdata <= x1 and E0 <= event.ydata <= E1:
                        cand.append(((x1 - x0) * (E1 - E0), b))
                if cand:
                    self.boxes[self.axis].remove(min(cand, key=lambda t: t[0])[1])
                    self._renumber(); self._redraw()
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        pts = self.points[self.axis]
        if event.button == 1:
            pts.append((event.xdata, event.ydata))
        elif event.button == 3 and pts:
            _, x, E = self._D(self.axis)
            sx = (x.max() - x.min()) or 1.0
            sE = (E.max() - E.min()) or 1.0
            j = min(range(len(pts)), key=lambda k:
                    ((pts[k][0] - event.xdata) / sx) ** 2 + ((pts[k][1] - event.ydata) / sE) ** 2)
            pts.pop(j)
        else:
            return
        self._fit_pot(); self._redraw()

    def _undo(self, _):
        if self.mode == "roi":
            if self.boxes[self.axis]:
                self.boxes[self.axis].pop(); self._renumber(); self._redraw()
        else:
            if self.points[self.axis]:
                self.points[self.axis].pop(); self._fit_pot(); self._redraw()

    def _clear(self, _):
        if self.mode == "roi":
            self.boxes[self.axis] = []
        else:
            self.points[self.axis] = []; self.pot_fit[self.axis] = None
        self._redraw()

    # ---------- import ----------
    def _choose_path(self, kind, prompt, loc):
        """Native file/folder chooser, cross-platform. macOS uses osascript
        (tkinter clashes with the macosx backend); Windows/Linux use a tkinter
        dialog. kind = 'file' | 'dir'. Returns a path, or '' if cancelled."""
        if sys.platform == "darwin":
            import subprocess
            verb = "folder" if kind == "dir" else "file"
            script = ('POSIX path of (choose %s with prompt "%s" '
                      'default location (POSIX file "%s"))' % (verb, prompt, loc))
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""
        # Windows / Linux: tkinter dialog (reuse the TkAgg backend window if present)
        import tkinter as tk
        from tkinter import filedialog
        parent = getattr(getattr(self.fig.canvas, "manager", None), "window", None)
        own = None
        if parent is None:                       # non-Tk backend: use a hidden root
            own = tk.Tk(); own.withdraw()
        kw = dict(title=prompt, initialdir=(loc or None))
        if parent is not None:
            kw["parent"] = parent
        try:
            p = (filedialog.askdirectory(**kw) if kind == "dir"
                 else filedialog.askopenfilename(**kw))
        finally:
            if own is not None:
                own.destroy()
        return p or ""

    def _pick(self, axis):
        try:
            loc = SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else _HERE
            p = self._choose_path("file", "Load %s dI/dV file" % axis, loc)
            if not p:
                return
            self._load((axis, p)); self._update_image(); self._redraw()
        except Exception as e:
            self.info.set_text("file dialog failed:\n%s\nuse path box" % e)
            self.fig.canvas.draw_idle()

    def _load_path(self, _):
        p = (self.path_box.text or "").strip()
        if not p:
            return
        try:
            self._load((self.axis, p)); self._update_image(); self._redraw()
        except Exception as e:
            self.info.set_text("load fail:\n%s" % e); self.fig.canvas.draw_idle()

    def _pick_outdir(self, _):
        try:
            cur = os.path.expanduser((self.outdir_box.text or "").strip())
            loc = cur if (cur and os.path.isdir(cur)) else (SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else _HERE)
            p = self._choose_path("dir", "Choose output folder", loc)
            if p:
                self.outdir_box.set_val(p)
        except Exception as e:
            self.info.set_text("folder dialog failed:\n%s\ntype path in out box" % e)
            self.fig.canvas.draw_idle()

    # ---------- export ----------
    def _outdir(self, prefix):
        root = os.path.expanduser((self.outdir_box.text or "").strip() or OUTPUT_DIR)
        tag = (self.tag_box.text or "").strip() or "run"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(root, "%s_%s_%s" % (prefix, tag, stamp))
        os.makedirs(d, exist_ok=True)
        return d

    def _save(self, _):
        if self.mode == "pot":
            return self._save_pot()
        maps = {ax: self._D(ax) for ax in ("major", "minor")}
        rb = {ax: [dict(i=b["i"], roi=b["roi"], Ew=b["Ew"], I=b["I"], fit=b["fit"], ens=b["ens"])
                   for b in self.boxes[ax]] for ax in ("major", "minor")}
        if not any(rb.values()):
            self.info.set_text("no boxes to save"); self.fig.canvas.draw_idle(); return
        outdir = self._outdir("tau")
        ftag = _filt_tag(maps["major"][0])
        old = tr.HERE; tr.HERE = outdir
        try:
            tr.save_fits_figure(rb["major"], "major", "run", ftag)
            tr.save_fits_figure(rb["minor"], "minor", "run", ftag)
            tr.save_overview(maps, rb, "run", ftag)
            tr.save_tau_summary(rb, "run", ftag)
        finally:
            tr.HERE = old
        self._write_tau_table(os.path.join(outdir, "tau_table.csv"))
        self.info.set_text("saved ->\n%s" % outdir)
        self.fig.canvas.draw_idle()
        print("  saved tau run ->", outdir)

    def _write_tau_table(self, path):
        fkey = "hp" if self.filter_on else "raw"
        gfw = self.gauss_fwhm if self.fitmode == "voigt" else ""
        head = ["axis", "roi", "x0_nm", "x1_nm", "E0_mV", "E1_mV", "E_state_mV",
                "Gamma_meV", "tau_fs", "sigma_tau_fs", "tau_lo_fs", "tau_hi_fs", "R2", "status",
                "filter", "fit_model", "background", "gauss_fwhm_meV"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(head)
            for axis in ("major", "minor"):
                for b in self.boxes[axis]:
                    ens = b["ens"]; fh = b.get("fit"); roi = b["roi"]
                    ok = (ens is not None and ens.get("status") != "no_valid_fit"
                          and np.isfinite(ens.get("tau", np.nan)))
                    if not ok:
                        continue
                    w.writerow([axis, b["i"], round(roi[0]), round(roi[1]), round(roi[2]), round(roi[3]),
                                round(fh["E0"], 1) if fh else "",
                                round(ens["G"], 2), round(ens["tau"]), round(ens["sigma_tau"]),
                                round(ens["tau_lo"]), round(ens["tau_hi"]),
                                round(fh["R2"], 3) if fh else "", ens["status"],
                                fkey, self.fitmode, self.bg_mode, gfw])
        print("  wrote", path)

    def _save_pot(self):
        outdir = self._outdir("pot")
        old = fp.HERE; fp.HERE = outdir
        saved = []
        try:
            for axis in ("major", "minor"):
                pts = self.points[axis]; f = self.pot_fit[axis]
                if len(pts) < 3 or f is None:
                    continue
                if not self.cal[axis]:
                    print("  skip %s: UNCALIBRATED" % axis); continue
                d, x, E = self._D(axis)
                fp.save_outputs(axis, "run", _filt_tag(d), [p[0] for p in pts], [p[1] for p in pts], f, d, x, E)
                saved.append(axis)
        finally:
            fp.HERE = old
        fj, fn = self.pot_fit["major"], self.pot_fit["minor"]
        joint = fj is not None and fn is not None and self.cal["major"] and self.cal["minor"]
        if joint:
            self._write_joint(os.path.join(outdir, "potential_joint.csv"))
        if not saved and not joint:
            self.info.set_text("need >=3 calibrated pts to save"); self.fig.canvas.draw_idle(); return
        self.info.set_text("saved ->\n%s" % outdir)
        self.fig.canvas.draw_idle()
        print("  saved pot run ->", outdir)

    def _write_joint(self, path):
        cj = fp.combine_axes(self.pot_fit["major"], self.pot_fit["minor"])

        def _s(v):
            return round(v, 4) if isinstance(v, float) and np.isfinite(v) else ""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quantity", "value", "sigma", "unit"])
            w.writerow(["a_x (major)", _s(cj["ax"]), _s(cj["sax"]), "meV/nm^2"])
            w.writerow(["a_y (minor)", _s(cj["ay"]), _s(cj["say"]), "meV/nm^2"])
            w.writerow(["x0", _s(cj["x0"]), _s(cj["sx0"]), "nm"])
            w.writerow(["y0", _s(cj["y0"]), _s(cj["sy0"]), "nm"])
            w.writerow(["V0", _s(cj["V0"]), _s(cj["sV0"]), "meV"])
            w.writerow(["aniso a_x/a_y", _s(cj["aniso_ratio"]), "", ""])
            w.writerow(["V0_major", _s(cj["V0_major"]), "", "meV"])
            w.writerow(["V0_minor", _s(cj["V0_minor"]), "", "meV"])
            w.writerow(["V0_diff", _s(cj["V0_diff"]), "", "meV"])
            w.writerow(["formula", "V(x,y)=a_x(x-x0)^2+a_y(y-y0)^2+V0", "", ""])
        print("  wrote", path)

    # ---------- UI ----------
    def _build(self):
        self.fig = plt.figure(figsize=(14.5, 8))
        try:
            self.fig.canvas.manager.set_window_title("ROI-tau / Potential panel")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.30, 0.08, 0.66, 0.86])
        d, x, E = self._D(self.axis)
        cmap, lo, hi = _cmap_clim(d)
        self.im = self.ax.imshow(d.T, aspect="auto", origin="lower", cmap=cmap,
                                 vmin=lo, vmax=hi, extent=[x.min(), x.max(), E.min(), E.max()])
        self.ax.set_xlabel("position (nm)"); self.ax.set_ylabel("bias (mV)")
        self._update_image()
        self.fig.text(0.02, 0.965, "ROI: L-drag add / R-click del    Pot: L-click / R-click", fontsize=8)
        axm = self.fig.add_axes([0.02, 0.882, 0.24, 0.058]); axm.set_title("mode", fontsize=9)
        self.r_mode = RadioButtons(axm, ("ROI-tau", "Potential"), active=0)
        self.r_mode.on_clicked(self._on_mode)
        axa = self.fig.add_axes([0.02, 0.798, 0.115, 0.056]); axa.set_title("axis", fontsize=8)
        self.r_axis = RadioButtons(axa, ("major", "minor"), active=0); self.r_axis.on_clicked(self._on_axis)
        axf = self.fig.add_axes([0.145, 0.798, 0.115, 0.056]); axf.set_title("filter", fontsize=8)
        self.r_filt = RadioButtons(axf, ("hp", "raw"), active=0); self.r_filt.on_clicked(self._on_filter)
        axfit = self.fig.add_axes([0.02, 0.716, 0.115, 0.056]); axfit.set_title("fit", fontsize=8)
        self.r_fit = RadioButtons(axfit, ("Lorentz", "Voigt"), active=0); self.r_fit.on_clicked(self._on_fit)
        axbg = self.fig.add_axes([0.145, 0.716, 0.115, 0.056]); axbg.set_title("background", fontsize=8)
        self.r_bg = RadioButtons(axbg, ("const", "quad"), active=0); self.r_bg.on_clicked(self._on_bg)
        self.gauss_box = TextBox(self.fig.add_axes([0.075, 0.672, 0.185, 0.030]), "G ",
                                 initial="%.2f" % self.gauss_fwhm)
        self.gauss_box.on_submit(self._on_gauss)
        self.b_lmaj = Button(self.fig.add_axes([0.02, 0.628, 0.115, 0.034]), "Load major")
        self.b_lmaj.on_clicked(lambda e: self._pick("major"))
        self.b_lmin = Button(self.fig.add_axes([0.145, 0.628, 0.115, 0.034]), "Load minor")
        self.b_lmin.on_clicked(lambda e: self._pick("minor"))
        self.path_box = TextBox(self.fig.add_axes([0.075, 0.586, 0.145, 0.032]), "path ", initial="")
        self.b_path = Button(self.fig.add_axes([0.225, 0.586, 0.035, 0.032]), ">")
        self.b_path.on_clicked(self._load_path)
        self.b_undo = Button(self.fig.add_axes([0.02, 0.544, 0.24, 0.034]), "Undo last")
        self.b_undo.on_clicked(self._undo)
        self.b_clear = Button(self.fig.add_axes([0.02, 0.502, 0.24, 0.034]), "Clear axis")
        self.b_clear.on_clicked(self._clear)
        self.b_save = Button(self.fig.add_axes([0.02, 0.460, 0.24, 0.034]), "Save / Export")
        self.b_save.on_clicked(self._save)
        self.tag_box = TextBox(self.fig.add_axes([0.075, 0.420, 0.185, 0.030]), "tag ", initial="run")
        self.outdir_box = TextBox(self.fig.add_axes([0.075, 0.382, 0.145, 0.030]), "out ",
                                  initial=self._init_outdir)
        self.b_outdir = Button(self.fig.add_axes([0.225, 0.382, 0.035, 0.030]), "...")
        self.b_outdir.on_clicked(self._pick_outdir)
        axi = self.fig.add_axes([0.02, 0.03, 0.26, 0.33]); axi.axis("off")
        self.info = axi.text(0, 1, "", va="top", ha="left", fontsize=7.5, family="monospace")
        self.rs = RectangleSelector(self.ax, self._on_select, useblit=False, button=[1],
                                    minspanx=2, minspany=2, spancoords="data", interactive=False)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._redraw()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--major", default=None, help="major file path (default: *_major in sample_data)")
    ap.add_argument("--minor", default=None, help="minor file path (default: *_minor in sample_data)")
    ap.add_argument("--outdir", default=None, help="output root directory (default GUI/output); can also be changed in the panel out box")
    args = ap.parse_args()
    print("=== ROI-tau / Potential panel ===")
    # Must hold Panel in a variable: matplotlib callbacks are weak references; without a strong reference it gets GC'd → all interaction breaks.
    panel = Panel(args.major, args.minor, outdir=args.outdir)
    plt.show()
    del panel


if __name__ == "__main__":
    main()
