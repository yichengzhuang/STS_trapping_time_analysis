#!/usr/bin/env python3
"""matrix_to_paperdata.py — convert a "pure matrix" dI/dV txt into the paper_data format that trapping_roi.py accepts.

Input (the kind you export from .pxp):
  Line 1 is a comment `# ... | pos_nm=A..B (Npos) | bias_mV=C..D (Nbias)`,
  followed by an Npos×Nbias pure intensity matrix (rows=position, cols=bias), with no coordinate axes.

Output (paper_data format, directly readable by trapping_roi.py's load_txt):
  row 0 = [NaN, E_0..E_{Nbias-1}] (energy/bias, meV);
  col 0 = [NaN, x_0..x_{Npos-1}] (position, nm);
  inner block = LDOS(pos, E).

The coordinate axes are parsed from the comment's `pos_nm=A..B (N)` / `bias_mV=C..D (N)`, then rebuilt with linspace
(verified: consistent with the sfA/sfB scaling of didv_matrix in the .pxp —— pos ±125nm, bias ±60mV, both linear).

Usage:
  python matrix_to_paperdata.py  # convert the two sets configured in JOBS below
"""
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (source matrix file, output paper_data file) —— output name must match trapping_roi.py's {data}_{axis}.txt
JOBS = [
    ("extended_sample2/LS_250617.204809-0T_didv_major.txt", "ext2_major.txt"),
    ("extended_sample2/LS_250617.020028-0T_didv_minor.txt", "ext2_minor.txt"),
]

_FLOAT = r"(-?\d+(?:\.\d+)?)"
_POS_RE = re.compile(r"pos_nm=" + _FLOAT + r"\.\." + _FLOAT + r"\s*\((\d+)\)")
_BIAS_RE = re.compile(r"bias_mV=" + _FLOAT + r"\.\." + _FLOAT + r"\s*\((\d+)\)")


def parse_axes_from_header(header):
    """Parse (pos_lo, pos_hi, npos), (bias_lo, bias_hi, nbias) from the comment line. Raise an error if not found."""
    p = _POS_RE.search(header)
    b = _BIAS_RE.search(header)
    if not p or not b:
        raise ValueError("No pos_nm=.. / bias_mV=.. axis info found in the comment:\n  %s" % header.strip())
    pos = (float(p.group(1)), float(p.group(2)), int(p.group(3)))
    bias = (float(b.group(1)), float(b.group(2)), int(b.group(3)))
    return pos, bias


def convert(src, dst):
    src_p = src if os.path.isabs(src) else os.path.join(HERE, src)
    dst_p = dst if os.path.isabs(dst) else os.path.join(HERE, dst)

    with open(src_p) as f:
        header = f.readline()
    if not header.lstrip().startswith("#"):
        raise ValueError("The first line is not a comment header, cannot parse coordinate axes: %s" % src_p)
    (pos_lo, pos_hi, npos), (bias_lo, bias_hi, nbias) = parse_axes_from_header(header)

    M = np.loadtxt(src_p, comments="#")          # (npos, nbias) pure matrix
    if M.shape != (npos, nbias):
        raise ValueError("Matrix shape %s does not match the (%d, %d) declared in the comment: %s"
                         % (M.shape, npos, nbias, src_p))

    x = np.linspace(pos_lo, pos_hi, npos)        # position (nm)
    E = np.linspace(bias_lo, bias_hi, nbias)     # energy/bias (meV)

    out = np.full((npos + 1, nbias + 1), np.nan) # leave the top-left corner as NaN
    out[0, 1:] = E                                # row 0 = energy axis
    out[1:, 0] = x                                # col 0 = position axis
    out[1:, 1:] = M                               # inner block = LDOS(pos, E)

    hdr = ("%s\nrow0=[NaN,bias(mV)...] col0=[NaN,pos(nm)...] inner=dI/dV(pos,bias) | "
           "shape %dx%d | pos %.2f..%.2f nm | bias %.2f..%.2f mV"
           % (os.path.basename(src), npos, nbias, pos_lo, pos_hi, bias_lo, bias_hi))
    np.savetxt(dst_p, out, fmt="%.7g", header=hdr, comments="# ")
    print("OK  %-45s -> %s  (%dx%d, x[%.0f,%.0f]nm E[%.0f,%.0f]mV)"
          % (src, dst, npos, nbias, pos_lo, pos_hi, bias_lo, bias_hi))
    return dst_p


def main():
    for src, dst in JOBS:
        convert(src, dst)


if __name__ == "__main__":
    main()
