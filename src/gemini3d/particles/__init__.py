from __future__ import annotations
import typing as T
import numpy as np
import xarray

from .. import write
from .grid import precip_grid
from .gaussian2d import gaussian2d


def particles_BCs(cfg: dict[str, T.Any], xg: dict[str, T.Any]):
    """ write particle precipitation to disk """

    pg = precip_grid(cfg, xg)

    # %% CREATE PRECIPITATION INPUT DATA
    # Q: energy flux [mW m^-2]
    # E0: characteristic energy [eV]

    # did user specify on/off time? if not, assume always on.
    t0 = pg.time[0].data

    if "precip_startsec" in cfg:
        t = t0 + np.timedelta64(cfg["precip_startsec"])
        i_on = abs(pg.time - t).argmin().item()
    else:
        i_on = 0

    if "precip_endsec" in cfg:
        t = t0 + np.timedelta64(cfg["precip_endsec"])
        i_off = abs(pg.time - t).argmin().item()
    else:
        i_off = pg.time.size

    # NOTE: in future, E0 could be made time-dependent in config.nml as 1D array
    for i in range(i_on, i_off):
        pg["Q"][i, :, :] = gaussian2d(pg, cfg["Qprecip"], cfg["Qprecip_background"])
        pg["E0"][i, :, :] = cfg["E0precip"]

    # %% CONVERT THE ENERGY TO EV
    # E0 = max(E0,0.100);
    # E0 = E0*1e3;

    # %% SAVE to files
    # LEAVE THE SPATIAL AND TEMPORAL INTERPOLATION TO THE
    # FORTRAN CODE IN CASE DIFFERENT GRIDS NEED TO BE TRIED.
    # THE EFIELD DATA DO NOT NEED TO BE SMOOTHED.

    write.precip(pg, cfg["precdir"], cfg["file_format"])