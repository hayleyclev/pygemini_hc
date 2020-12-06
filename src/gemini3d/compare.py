"""
compare simulation outputs to verify model performance
"""

from pathlib import Path
import numpy as np
import logging
import argparse
from datetime import datetime
import typing as T
import sys

from .readdata import (
    read_config,
    loadframe,
    datetime_range,
    read_precip,
    read_Efield,
    read_state,
)

from .find import get_frame_filename

try:
    from .plotdiff import plotdiff
    from matplotlib.pyplot import show
except ImportError:
    plotdiff = show = None

TOL = {
    "rtol": 1e-5,
    "rtolN": 1e-5,
    "rtolT": 1e-5,
    "rtolJ": 1e-5,
    "rtolV": 1e-5,
    "atol": 1e-8,
    "atolN": 1e9,
    "atolT": 100,
    "atolJ": 1e-7,
    "atolV": 50,
}


def cli():

    p = argparse.ArgumentParser(description="Compare simulation file outputs and inputs")
    p.add_argument("outdir", help="directory to compare")
    p.add_argument("refdir", help="reference directory")
    p.add_argument("-p", "--plot", help="make plots of differences", action="store_true")
    p.add_argument("-only", help="only check in or out", choices=["in", "out"])
    p.add_argument(
        "-file_format",
        help="specify file format to read from output dir",
        choices=["h5", "nc", "raw"],
    )
    P = p.parse_args()

    errs = compare_all(P.outdir, P.refdir, TOL, P.plot, P.file_format, P.only)

    if errs:
        for e, v in errs.items():
            print(f"{e} has {v} errors", file=sys.stderr)
        if P.plot and show is not None:
            show()
        raise SystemExit(f"FAIL: compare {P.outdir}")

    print(f"OK: Gemini comparison {P.outdir} {P.refdir}")


def compare_all(
    outdir: Path,
    refdir: Path,
    tol: T.Dict[str, float] = TOL,
    plot: bool = True,
    file_format: str = None,
    only: str = None,
) -> T.Dict[str, int]:
    """
    compare two directories across time steps
    """
    outdir = Path(outdir).expanduser()
    refdir = Path(refdir).expanduser()

    if outdir.samefile(refdir):
        raise OSError(f"reference and output are the same directory: {outdir}")

    # %% READ IN THE SIMULATION INFORMATION
    params = read_config(outdir)
    if not params:
        raise FileNotFoundError(f"{outdir} does not appear to contain config.nml")
    # %% TIMES OF INTEREST
    t0 = params["t0"]
    times = datetime_range(t0, t0 + params["tdur"], params["dtout"])
    if len(times) <= 1:
        raise ValueError(
            f"{outdir} simulation did not run long enough, must run for more than one time step"
        )

    errs = {}
    if not only or only == "out":
        e = compare_output(outdir, refdir, tol, times, file_format, plot)
        if e:
            errs["out"] = e

    if not only or only == "in":
        e = compare_input(outdir, refdir, tol, times, file_format, plot)
        if e:
            errs["in"] = e

    return errs


def compare_input(
    outdir: Path,
    refdir: Path,
    tol: T.Dict[str, float],
    times: T.Sequence[datetime],
    file_format: str,
    plot: bool = True,
) -> int:

    if len(times) == 0:
        raise ValueError("Must have at least one time to compare")

    ref_params = read_config(refdir)
    if not ref_params:
        raise FileNotFoundError(f"{refdir} does not appear to contain config.nml")
    ref_indir = refdir / ref_params["indat_file"].parts[-2]
    ref = read_state(ref_indir / ref_params["indat_file"].name)

    new_params = read_config(outdir)
    if not new_params:
        raise FileNotFoundError(f"{outdir} does not appear to contain config.nml")
    new_indir = outdir / new_params["indat_file"].parts[-2]
    new = read_state(new_indir / new_params["indat_file"].name)

    if not file_format:
        file_format = new_params["indat_file"].suffix[1:]

    errs = 0
    # %% initial conditions
    names = ("ns", "Ts", "vs")
    itols = ("N", "T", "V")

    for k, j in zip(names, itols):
        b = ref[k]
        a = new[k]

        assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

        if not np.allclose(a, b, 0.1 * tol[f"rtol{j}"], 0.1 * tol[f"atol{j}"]):
            errs += 1
            logging.error(f"{k}  {err_pct(a, b):.1f} %")

            if plot and plotdiff is not None:
                plotdiff(a, b, k, times[0], outdir, refdir)

    if "precdir" in new_params:
        prec_errs = compare_precip(
            new_indir, new_params, ref_indir, ref_params, tol, times, plot, file_format
        )
        errs += prec_errs

    if "E0dir" in new_params:
        efield_errs = compare_Efield(
            new_indir, new_params, ref_indir, ref_params, tol, times, plot, file_format
        )
        errs += efield_errs

    return errs


def err_pct(a: np.ndarray, b: np.ndarray) -> float:
    """ compute maximum error percent """

    return (abs(a - b).max() / abs(b).max()) * 100


def compare_precip(
    new_indir: Path,
    new_params: T.Dict[str, T.Any],
    ref_indir: Path,
    ref_params: T.Dict[str, T.Any],
    tol: T.Dict[str, float],
    times: T.Sequence[datetime],
    plot: bool,
    file_format: str,
) -> int:

    prec_errs = 0
    prec_path = new_indir / new_params["precdir"].name

    # often we reuse precipitation inputs without copying over files
    for t in times:
        ref = read_precip(
            get_frame_filename(ref_indir / ref_params["precdir"].name, t), file_format
        )
        new = read_precip(get_frame_filename(prec_path, t), file_format)

        for k in ref.keys():
            b = np.atleast_1d(ref[k])
            a = np.atleast_1d(new[k])

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, tol["rtol"], tol["atol"]):
                prec_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, new_indir.parent, ref_indir.parent)
            if prec_errs == 0:
                print(f"OK: {k}  {prec_path}")

    return prec_errs


def compare_Efield(
    new_indir: Path,
    new_params: T.Dict[str, T.Any],
    ref_indir: Path,
    ref_params: T.Dict[str, T.Any],
    tol: T.Dict[str, float],
    times: T.Sequence[datetime],
    plot: bool,
    file_format: str,
) -> int:

    efield_errs = 0
    efield_path = new_indir / new_params["E0dir"].name
    # often we reuse Efield inputs without copying over files
    for t in times:
        ref = read_Efield(get_frame_filename(ref_indir / ref_params["E0dir"].name, t), file_format)
        new = read_Efield(get_frame_filename(efield_path, t), file_format)
        for k in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            b = ref[k][1]
            a = new[k][1]

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, tol["rtol"], tol["atol"]):
                efield_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, new_indir.parent, ref_indir.parent)

    if efield_errs == 0:
        print(f"OK: Efield {efield_path}")

    return efield_errs


def compare_output(
    outdir: Path,
    refdir: Path,
    tol: T.Dict[str, float],
    times: T.Sequence[datetime],
    file_format: str = None,
    plot: bool = True,
) -> int:
    """compare simulation outputs"""

    ref: T.Dict[str, T.Any] = {}
    errs = 0
    a: np.ndarray = None

    for i, t in enumerate(times):
        st = f"UTsec {t}"
        A = loadframe(outdir, t, file_format)
        B = loadframe(refdir, t)
        # don't specify file_format for reference,
        # so that one reference file format can check multiple "out" format

        names = ["ne", "v1", "v2", "v3", "Ti", "Te", "J1", "J2", "J3"]
        itols = ["N", "V", "V", "V", "T", "T", "J", "J", "J"]

        for k, j in zip(names, itols):
            a = A[k][1]
            b = B[k][1]

            assert a.shape == b.shape, f"{k} time {i} {t}: shape: ref {b.shape} != data {a.shape}"

            if not np.allclose(a, b, tol[f"rtol{j}"], tol[f"atol{j}"]):
                errs += 1
                logging.error(f"{k} {st}   {err_pct(a, b):.1f}")
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, outdir, refdir)
        # %% assert time steps have unique output (earth always rotating...)
        if i > 1:
            names = ["ne", "v1", "v2", "v3"]
            itols = ["N", "V", "V", "V"]
            for k, j in zip(names, itols):
                if np.allclose(ref[k][1], a, 0.0001 * tol[f"rtol{j}"], 0.0001 * tol[f"atol{j}"]):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        if i == 3:
            for k in ("Ti", "Te"):
                if np.allclose(ref[k][1], A[k][1], 0.01 * tol["rtolT"], 0.1 * tol["atolT"]):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        if i == 2:
            for k in ("J1", "J2", "J3"):
                if np.allclose(ref[k][1], a, 0.01 * tol["rtolJ"], 0.1 * tol["atolJ"]):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        ref.update(A)

    return errs


if __name__ == "__main__":
    cli()
