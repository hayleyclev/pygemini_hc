"""
these test that PyGemini generates inputs that match expectations
"""

import logging
import importlib.resources
import pytest
import os

import gemini3d
import gemini3d.web
import gemini3d.grid
from gemini3d.compare import compare_all
import gemini3d.read
import gemini3d.model
import gemini3d.find
import gemini3d.msis


@pytest.mark.skipif(gemini3d.msis.get_msis_exe() is None, reason="msis_setup not available")
@pytest.mark.parametrize(
    "name,file_format",
    [
        ("mini2dew_eq", "h5"),
        ("mini2dew_fang", "h5"),
        ("mini2dew_glow", "h5"),
        ("mini2dns_eq", "h5"),
        ("mini2dns_fang", "h5"),
        ("mini2dns_glow", "h5"),
        ("mini3d_eq", "h5"),
        ("mini3d_fang", "h5"),
        ("mini3d_glow", "h5"),
    ],
)
def test_runner(name, file_format, tmp_path, monkeypatch):

    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    out_dir = tmp_path
    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    # setup new test data
    params = gemini3d.read.config(test_dir)

    params["file_format"] = file_format
    params["out_dir"] = out_dir

    for k in {"indat_file", "indat_size", "indat_grid"}:
        params[k] = params[k].with_suffix("." + file_format)

    # patch eq_dir to use reference data
    if "eq_dir" in params:
        eq_dir = test_dir.parent / params["eq_dir"].name
        if eq_dir.is_dir():
            print(f"Using {eq_dir} for equilibrium data")
        params["eq_dir"] = eq_dir

    # %% generate initial condition files
    gemini3d.model.setup(params, out_dir)

    # %% check generated files
    errs = compare_all(
        params["out_dir"], ref_dir=test_dir, only="in", plot=False, file_format=file_format
    )

    if errs:
        for err, v in errs.items():
            logging.error(f"compare:{err}: {v} errors")
        raise ValueError(f"compare_input: new generated inputs do not match reference for: {name}")