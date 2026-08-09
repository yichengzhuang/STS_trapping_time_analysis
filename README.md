# Trapping-time extraction from LDOS maps

An interactive tool for measuring the trapping time (lifetime) of quasi-bound
electronic states from scanning-tunnelling-microscopy local density of states
(LDOS) maps.

A spectral feature is framed by hand on an LDOS map `L(x, E)`; the intensity inside
the box is summed along the position axis to form an energy distribution `I(E)`,
which is fitted with a Lorentzian or Voigt line shape. The trapping time follows from
the fitted linewidth `Γ` via `τ = ħ/Γ`, and an uncertainty is estimated from an
ensemble of nearby windows.

## Features

- **Interactive panel** (matplotlib) with two modes:
  - *Lifetime* — drag a box around a peak, integrate along position, and read a live
    `τ ± σ_τ` for each box.
  - *Potential* — click points along a band and fit a quadratic
    `V(s) = a(s − s₀)² + V₀` with parameter errors; the two axes combine into a
    2-D anisotropic potential.
- **Line shape** — Lorentzian, or Voigt with a fixed Gaussian width for the
  instrumental and thermal broadening (the Gaussian FWHM is entered on the panel).
- **Background** — a constant, or a quadratic polynomial fitted jointly with the
  peak (useful on raw data, where the peak sits on a sloped background).
- **Filtering** — raw data, or high-pass background removal.
- **Uncertainty** — an ensemble of perturbed windows gives a median linewidth and a
  percentile band.
- **Input formats** — labelled text maps, plain `dI/dV` matrices, and Igor `.pxp`.
- **Output** — a timestamped folder per run containing a results table (CSV) and
  figures.

## How it works

Each quasi-bound state appears as a peak in the LDOS map `L(x, E)`. To turn that
peak into a lifetime:

1. **Box the peak** on the map.
2. **Integrate along position** — the intensity inside the box is summed over the
   position axis, collapsing the 2-D patch into a 1-D energy distribution `I(E)`.
3. **Fit the line shape** of `I(E)` (a peak on a background):
   - *Lorentzian* — the line shape of a state with a finite lifetime; its full
     width at half maximum `Γ` is the quantity we want.
   - *Voigt* — a Lorentzian convolved with a fixed Gaussian that represents the
     thermal and instrumental broadening, so the fitted `Γ` is the intrinsic
     (deconvolved) width. The Gaussian FWHM is entered in the **G** box.
   - *background* — constant, or a quadratic polynomial fitted together with the
     peak. On raw data the peak usually sits on a sloped, slowly varying
     background; the quadratic option absorbs it instead of letting it broaden
     the peak.
4. **Convert to a lifetime** through `τ = ħ / Γ`  (`ħ = 658.2 meV·fs`).
5. **Estimate the uncertainty** — the fit is repeated over a few hundred slightly
   shifted and stretched windows around the one you drew; the reported `τ` is the
   median of that ensemble and the error bar is its spread, so the result does not
   hinge on a single hand-placed box.

The optional high-pass filter (`hp`) subtracts a broadened copy of the map to
remove a slowly varying background, which can make weak features easier to see
before boxing them.

## Installation

```bash
pip install numpy scipy matplotlib igor2
```

Python 3.9 or newer. `igor2` is only needed to read `.pxp` files.

## Usage

```bash
cd GUI
python roi_gui.py
```

At startup the panel loads any `*_major` / `*_minor` files found in
`GUI/sample_data/`. Otherwise, use the **Load major** / **Load minor** buttons, or
type a path in the box, to import your own data. The two maps and an output directory
can also be given on the command line:

```bash
python roi_gui.py --major path/to/major.txt --minor path/to/minor.txt --outdir results/
```

### Controls

| Control | Action |
|---|---|
| **mode** | switch between lifetime and potential fitting |
| **axis** | select the major or minor map |
| **filter** | high-pass (`hp`) or `raw` |
| **fit** | Lorentzian or Voigt (the Gaussian FWHM is set in the **G** box) |
| **background** | constant, or a quadratic polynomial fitted jointly with the peak (for raw data) |
| left-drag | *(lifetime)* draw a region and read its `τ` |
| left-click | *(potential)* add a point |
| right-click | delete the region or point under the cursor |
| **Undo / Clear** | remove the last item, or all items, on the current axis |
| **Save / Export** | write a timestamped folder to the output directory |

Each *Save* writes to its own folder, so runs never overwrite one another.

### A typical session

1. Launch the panel — the major / minor maps load automatically, or use **Load
   major** / **Load minor** (or the path box) to import your own.
2. Choose **filter** (`raw` / `hp`), **fit** (Lorentzian / Voigt) and
   **background** (const / quad).
3. In *lifetime* mode, drag a box around a peak; the trapping time `τ ± σ` appears
   inside the box at once. Add as many boxes as you like, and right-click a box to
   delete it.
4. Click **Save / Export** to write the results table (CSV — one row per box with
   `Γ`, `τ` and its error band) and the fit figures to a timestamped folder.

## Input format

A text map is a matrix of `dI/dV` values with position along rows and energy (bias)
along columns. Two conventions are recognised automatically:

- **Labelled** — row 0 holds the energy axis, column 0 the position axis, with `NaN`
  in the corner.
- **Plain matrix** — the values only, with the axis ranges given in a header comment
  (`# ... pos_nm=... bias_mV=...`).

`.pxp` files are read with `igor2`; the LDOS is taken from the `didv_matrix` wave and
the position/energy axes from its scaling.

## Output

**Save / Export** writes a new timestamped folder under the output directory (runs
never overwrite one another). In *lifetime* mode the folder contains:

- `tau_table.csv` — one row per box, with the columns listed below.
- `fits_<axis>_run_<filter>.png` — one panel per box: the integrated `I(E)`, the
  fitted curve, `Γ`, `τ`, `R²`, and the ensemble `τ` histogram.
- `ldos_overview_run_<filter>.png` — the major and minor maps with every box drawn
  and its `τ` labelled inside.
- `tau_summary_run_<filter>.png` — `τ` for each box with error bars (point =
  ensemble median, bar = ±σ_τ, shaded = the box-jitter 16/84 band).

`tau_table.csv` columns:

| Column | Meaning |
|---|---|
| `axis` | `major` / `minor` |
| `roi` | box number on that axis |
| `x0_nm`, `x1_nm`, `E0_mV`, `E1_mV` | box edges (position, energy) |
| `E_state_mV` | fitted peak centre |
| `Gamma_meV` | fitted linewidth `Γ` |
| `tau_fs` | trapping time `τ = ħ / Γ` |
| `sigma_tau_fs` | combined uncertainty on `τ` |
| `tau_lo_fs`, `tau_hi_fs` | 16 / 84 percentile band of `τ` |
| `R2` | fit quality |
| `status` | `ok` / `unreliable` / `no_valid_fit` |
| `filter` | `raw` / `hp` |
| `fit_model` | `lorentz` / `voigt` |
| `background` | `const` / `quad` |
| `gauss_fwhm_meV` | Voigt Gaussian FWHM (blank for Lorentzian) |

In *potential* mode a Save instead writes `potential_fit_<axis>_run.png` (map +
picked points + parabola + parameters), `potential_points_<axis>_run.csv` (the
points you clicked, so a fit can be reproduced), `potential_fits.csv` (the parabola
parameters `a`, `s₀`, `V₀`, `d²V/ds²`, `R²`), and — when both axes are fitted —
`potential_joint.csv` (the combined 2-D potential `V(x, y)`).

## Repository layout

| Path | Contents |
|---|---|
| `GUI/roi_gui.py` | the interactive panel |
| `trapping_roi.py` | LDOS I/O, integration, Lorentzian / Voigt fitting, ensemble uncertainty |
| `fit_potential.py` | quadratic potential fitting |

`GUI/roi_gui.py` imports `trapping_roi.py` and `fit_potential.py` from the parent
directory, so keep the two core modules one level above `GUI/`.

## License

Released under the MIT License — see [LICENSE](LICENSE).
