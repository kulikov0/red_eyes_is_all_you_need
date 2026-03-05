"""
Generate ISPA PWL coefficients for fp16 GELU - https://www.mdpi.com/2079-9292/14/9/1825

EPSS (Error Peak Splitting Strategy) breakpoint search on erf(t) over [0, 3.5].
16 non-uniform segments via iterative split at local error maxima.
Per-segment least-squares fit: erf(t) ~ a_i * t + b_i
Convert to x-space: fold x/sqrt(2) into breakpoints and slopes.
Quantize to fp16. Write mem/gelu_pwl.hex.

Output format: 15 breakpoints + 16 x {slope, intercept} = 47 fp16 values
"""

import os
import math
import numpy as np
from scipy.special import erf
from scipy.signal import argrelextrema

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "mem", "gelu_pwl.hex")
VH_OUT = os.path.join(PROJ, "rtl", "gelu_pwl_coeffs.vh")

N_SEGMENTS = 16
T_MAX = 3.5
N_SAMPLE = 50000


def float_to_fp16_bits(f):
    return int(np.float16(f).view(np.uint16))


def fp16_bits_to_float(bits):
    return float(np.uint16(bits).view(np.float16))


# Fit one segment, return (slope, intercept, max_error)
def fit_segment(t_all, erf_all, lo, hi):
    m = (t_all >= lo) & (t_all <= hi)
    ts, es = t_all[m], erf_all[m]
    if len(ts) < 2:
        return 0.0, math.erf(lo), 0.0
    A = np.column_stack([ts, np.ones_like(ts)])
    r = np.linalg.lstsq(A, es, rcond=None)
    a, b = float(r[0][0]), float(r[0][1])
    errs = np.abs(es - (a * ts + b))
    return a, b, float(errs.max())


# Find local maxima of error within a segment (interior only)
def find_error_peaks(t_all, erf_all, lo, hi, order=50):
    m = (t_all >= lo) & (t_all <= hi)
    ts, es = t_all[m], erf_all[m]
    if len(ts) < 3:
        return []
    A = np.column_stack([ts, np.ones_like(ts)])
    r = np.linalg.lstsq(A, es, rcond=None)
    a, b = r[0]
    errs = np.abs(es - (a * ts + b))
    peaks = argrelextrema(errs, np.greater, order=order)[0]
    return [float(ts[p]) for p in peaks]


# EPSS: iteratively insert breakpoints at local error maxima
def epss_breakpoints(t_all, erf_all):
    bp = [0.0, T_MAX]
    rnd = 0

    while len(bp) - 1 < N_SEGMENTS:
        n_segs = len(bp) - 1
        new_points = []

        for i in range(n_segs):
            peaks = find_error_peaks(t_all, erf_all, bp[i], bp[i + 1])
            _, _, max_err = fit_segment(t_all, erf_all, bp[i], bp[i + 1])
            if peaks:
                print(f"  Round {rnd} seg[{i}] [{bp[i]:.4f}, {bp[i+1]:.4f}]"
                      f" max_err={max_err:.6e} peaks={[f'{p:.4f}' for p in peaks]}")
                new_points.extend(peaks)
            else:
                print(f"  Round {rnd} seg[{i}] [{bp[i]:.4f}, {bp[i+1]:.4f}]"
                      f" max_err={max_err:.6e} (no interior peaks)")

        if not new_points:
            print(f"  No more peaks found, stopping at {n_segs} segments")
            break

        # Would adding all peaks exceed target? Take only the ones from
        # the segments with the largest errors
        room = N_SEGMENTS - n_segs
        if len(new_points) > room:
            # Rank by error at each peak position
            peak_errs = []
            for pt in new_points:
                # Find which segment this peak belongs to
                for i in range(n_segs):
                    if bp[i] <= pt <= bp[i + 1]:
                        _, _, err = fit_segment(t_all, erf_all, bp[i], bp[i + 1])
                        peak_errs.append((err, pt))
                        break
            peak_errs.sort(reverse=True)
            new_points = [pt for _, pt in peak_errs[:room]]

        bp = sorted(set(bp + new_points))
        rnd += 1

    n_segs = len(bp) - 1
    max_e = 0
    for i in range(n_segs):
        _, _, err = fit_segment(t_all, erf_all, bp[i], bp[i + 1])
        max_e = max(max_e, err)
    print(f"  Final: {n_segs} segments, max segment error = {max_e:.6e}")
    return bp


# Convert from t-space to x-space (x = t * sqrt(2))
def convert_to_x_space(breakpoints, slopes, intercepts):
    sqrt2 = math.sqrt(2.0)
    breaks_x = [t * sqrt2 for t in breakpoints]
    # erf(t) ~ a_t * t + b_t, with t = x/sqrt(2)
    # erf(x/sqrt2) ~ (a_t/sqrt2) * x + b_t
    slopes_x = [a / sqrt2 for a in slopes]
    return breaks_x, slopes_x, list(intercepts)


if __name__ == "__main__":
    print(f"Sampling erf(t) on [0, {T_MAX}] with {N_SAMPLE} points...")
    t_all = np.linspace(0, T_MAX, N_SAMPLE)
    erf_all = erf(t_all)

    print(f"EPSS breakpoint search ({N_SEGMENTS} segments)...")
    breakpoints = epss_breakpoints(t_all, erf_all)

    # Final fit
    slopes = []
    intercepts = []
    print(f"\nSegment fits (t-space):")
    for i in range(len(breakpoints) - 1):
        a, b, err = fit_segment(t_all, erf_all, breakpoints[i], breakpoints[i + 1])
        slopes.append(a)
        intercepts.append(b)
        print(f"  seg[{i:2d}] [{breakpoints[i]:.4f}, {breakpoints[i+1]:.4f}]"
              f" a={a:+.8f} b={b:+.8f} max_err={err:.6e}")

    # Convert to x-space
    breaks_x, slopes_x, intercepts_x = convert_to_x_space(breakpoints, slopes, intercepts)
    inner_breaks = breaks_x[1:-1]
    sat_x = breaks_x[-1]

    print(f"\nBreakpoints (x-space, {len(inner_breaks)} inner):")
    for i, bx in enumerate(inner_breaks):
        bits = float_to_fp16_bits(bx)
        print(f"  break[{i:2d}] = {bx:8.5f} -> fp16 0x{bits:04x} ({fp16_bits_to_float(bits):8.5f})")

    print(f"\nCoefficients (x-space):")
    for i in range(len(slopes)):
        a_bits = float_to_fp16_bits(slopes_x[i])
        b_bits = float_to_fp16_bits(intercepts_x[i])
        print(f"  seg[{i:2d}] slope=0x{a_bits:04x} ({fp16_bits_to_float(a_bits):+.6f})"
              f"  icept=0x{b_bits:04x} ({fp16_bits_to_float(b_bits):+.6f})")

    # Error stats with fp16-quantized coefficients
    inner_bits = [float_to_fp16_bits(b) for b in inner_breaks]
    slope_bits = [float_to_fp16_bits(s) for s in slopes_x]
    icept_bits = [float_to_fp16_bits(c) for c in intercepts_x]
    sat_bits = float_to_fp16_bits(sat_x)

    x_test = np.linspace(-8, 8, 50000)
    ideal_gelu = x_test * 0.5 * (1.0 + erf(x_test / math.sqrt(2.0)))

    pwl_gelu = np.zeros_like(x_test)
    for idx, x in enumerate(x_test):
        abs_x_f16 = np.float16(abs(x))
        if abs_x_f16 >= fp16_bits_to_float(sat_bits):
            erf_val = np.float16(1.0)
        else:
            seg = 0
            for k in range(len(inner_bits)):
                if abs_x_f16 >= fp16_bits_to_float(inner_bits[k]):
                    seg = k + 1
            a_f = fp16_bits_to_float(slope_bits[seg])
            b_f = fp16_bits_to_float(icept_bits[seg])
            erf_val = np.float16(np.float16(a_f) * abs_x_f16 + np.float16(b_f))
        if x < 0:
            erf_val = -erf_val
        half_x = np.float16(np.float16(x) * np.float16(0.5))
        pwl_gelu[idx] = float(np.float16(half_x * np.float16(np.float16(1.0) + erf_val)))

    errs = np.abs(pwl_gelu - ideal_gelu)
    print(f"\nFP16 PWL GELU vs ideal over [-8, 8] ({len(x_test)} points):")
    print(f"  MSE      = {np.mean(errs**2):.6e}")
    print(f"  Max err  = {np.max(errs):.6e}")
    print(f"  Mean err = {np.mean(errs):.6e}")

    # Write hex file
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("// GELU PWL coefficients: 15 breakpoints + 16 x {slope, intercept}\n")
        f.write(f"// Generated by gen_gelu_pwl_coeffs.py (EPSS {len(slopes)} segments)\n")
        f.write(f"// Saturation: |x| >= {sat_x:.4f} (0x{sat_bits:04x}) -> erf = 1.0\n")
        f.write("// Section 1: 15 inner breakpoints (fp16, x-space)\n")
        for b in inner_breaks:
            f.write(f"{float_to_fp16_bits(b):04x}\n")
        f.write("// Section 2: 16 x {slope, intercept} (fp16 pairs, x-space)\n")
        for i in range(len(slopes)):
            f.write(f"{float_to_fp16_bits(slopes_x[i]):04x}\n")
            nl = "\n" if i < len(slopes) - 1 else ""
            f.write(f"{float_to_fp16_bits(intercepts_x[i]):04x}{nl}")

    print(f"\nWrote {OUT} (47 fp16 values)")

    # Write Verilog include file
    with open(VH_OUT, "w") as f:
        f.write("// Auto-generated by gen_gelu_pwl_coeffs.py - DO NOT EDIT\n")
        f.write(f"// EPSS {len(slopes)} segments, saturation |x| >= {sat_x:.4f}\n\n")
        f.write(f"localparam [15:0] SAT_THRESH = 16'h{sat_bits:04x};\n\n")
        for i, bx in enumerate(inner_breaks):
            f.write(f"localparam [15:0] BREAK_{i:02d} = 16'h{float_to_fp16_bits(bx):04x};"
                    f" // {bx:.5f}\n")
        f.write("\n")
        for i in range(len(slopes)):
            f.write(f"localparam [15:0] SLOPE_{i:02d} = 16'h{float_to_fp16_bits(slopes_x[i]):04x};"
                    f" // {slopes_x[i]:+.6f}\n")
        f.write("\n")
        for i in range(len(slopes)):
            nl = "\n" if i < len(slopes) - 1 else ""
            f.write(f"localparam [15:0] ICEPT_{i:02d} = 16'h{float_to_fp16_bits(intercepts_x[i]):04x};"
                    f" // {intercepts_x[i]:+.6f}{nl}")

    print(f"Wrote {VH_OUT}")