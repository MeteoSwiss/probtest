"""
Tests for configurable fof tolerance variables (--fof-tolerance-vars):
tolerance writes one column per variable, check reads the variables from the
tolerance file columns, fof-compare applies its tolerance to all of them.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner

from engine.check import check
from engine.fof_compare import fof_compare
from engine.tolerance import tolerance

E_O_PERTURBATIONS = [1e-3, -2e-3, 5e-4]


@pytest.fixture(name="fof_ensemble")
def fixture_fof_ensemble(tmp_dir, sample_dataset_fof):
    """Reference plus members in which veri_data and e_o are perturbed."""
    ens_dir = os.path.join(tmp_dir, "fof_tol_vars")
    os.makedirs(ens_dir, exist_ok=True)

    sample_dataset_fof.to_netcdf(os.path.join(ens_dir, "fofAIREP_.nc"))
    for member_id, eps in enumerate(E_O_PERTURBATIONS, start=1):
        ds = sample_dataset_fof.copy(deep=True)
        ds["veri_data"] = (("d_body",), ds["veri_data"].values * (1 + eps))
        ds["e_o"] = (("d_body",), ds["e_o"].values * (1 + eps))
        ds.to_netcdf(os.path.join(ens_dir, f"fofAIREP_{member_id}.nc"))

    # a "current" run with e_o slightly changed, within the ensemble spread
    cur = sample_dataset_fof.copy(deep=True)
    cur["e_o"] = (("d_body",), cur["e_o"].values * (1 + 1e-4))
    cur_file = os.path.join(ens_dir, "fofAIREP_cur.nc")
    cur.to_netcdf(cur_file)

    return ens_dir, cur_file


def run_tolerance(ens_dir, tol_file, tol_vars=None):
    args = [
        "--ensemble-files",
        os.path.join(ens_dir, "fof{fof_type}_{member_id}.nc"),
        "--tolerance-files",
        tol_file,
        "--member-ids",
        ",".join(str(i) for i in range(1, len(E_O_PERTURBATIONS) + 1)),
        "--fof-types",
        "AIREP",
    ]
    if tol_vars is not None:
        args += ["--fof-tolerance-vars", tol_vars]
    return CliRunner().invoke(tolerance, args)


def run_check(ref_file, cur_file, tol_file):
    return CliRunner().invoke(
        check,
        [
            "--reference-files",
            ref_file,
            "--current-files",
            cur_file,
            "--tolerance-files",
            tol_file,
            "--factor",
            "1.0",
            "--fof-types",
            "",
        ],
    )


def test_tolerance_writes_one_column_per_variable(fof_ensemble, sample_dataset_fof):
    ens_dir, _ = fof_ensemble
    tol_file = os.path.join(ens_dir, "tol_AIREP.csv")

    result = run_tolerance(ens_dir, os.path.join(ens_dir, "tol_{fof_type}.csv"))
    assert result.exit_code == 0, result.output
    assert list(pd.read_csv(tol_file, index_col=0).columns) == ["veri_data"]

    result = run_tolerance(
        ens_dir, os.path.join(ens_dir, "tol_{fof_type}.csv"), "veri_data,e_o"
    )
    assert result.exit_code == 0, result.output
    df_tol = pd.read_csv(tol_file, index_col=0)
    assert list(df_tol.columns) == ["veri_data", "e_o"]
    assert len(df_tol) == sample_dataset_fof.sizes["d_body"]

    # e_o tolerance is the maximum relative difference over the members
    e_o = np.sort(sample_dataset_fof["e_o"].values)  # rows are sorted, not original
    expected = np.max(
        [np.abs(e_o * eps) / (1.0 + np.abs(e_o)) for eps in E_O_PERTURBATIONS],
        axis=0,
    )
    np.testing.assert_allclose(np.sort(df_tol["e_o"].values), np.sort(expected))


def test_tolerance_rejects_sort_key(fof_ensemble):
    ens_dir, _ = fof_ensemble
    result = run_tolerance(
        ens_dir, os.path.join(ens_dir, "tol_{fof_type}.csv"), "veri_data,level"
    )
    assert isinstance(result.exception, ValueError)
    assert "sort keys" in str(result.exception)


def test_check_reads_tolerance_vars_from_csv(fof_ensemble):
    ens_dir, cur_file = fof_ensemble
    ref_file = os.path.join(ens_dir, "fofAIREP_.nc")
    tol_pattern = os.path.join(ens_dir, "tol_{fof_type}.csv")
    tol_file = os.path.join(ens_dir, "tol_AIREP.csv")

    # veri_data only: the changed e_o is compared exactly and fails
    run_tolerance(ens_dir, tol_pattern)
    assert run_check(ref_file, cur_file, tol_file).exit_code == 1

    # veri_data and e_o: the e_o change is within the tolerance
    run_tolerance(ens_dir, tol_pattern, "veri_data,e_o")
    assert run_check(ref_file, cur_file, tol_file).exit_code == 0


@pytest.mark.parametrize("header", ["0", "tolerance"])
def test_check_legacy_single_column_tolerance(fof_ensemble, header):
    """Old tolerance files with one unnamed column still apply to veri_data."""
    ens_dir, _ = fof_ensemble
    ref_file = os.path.join(ens_dir, "fofAIREP_.nc")
    member_file = os.path.join(ens_dir, "fofAIREP_1.nc")
    tol_file = os.path.join(ens_dir, f"tol_legacy_{header}.csv")

    # members differ in e_o too; build a run with only veri_data changed
    cur_file = os.path.join(ens_dir, "fofAIREP_veri_only.nc")
    with xr.open_dataset(member_file) as member, xr.open_dataset(ref_file) as ref:
        ref = ref.load()
        ref["veri_data"] = member["veri_data"].load()
    ref.to_netcdf(cur_file)

    n_rows = int(ref.attrs["n_body"])
    pd.DataFrame({header: np.full(n_rows, 0.1)}).to_csv(tol_file)
    assert run_check(ref_file, cur_file, tol_file).exit_code == 0

    pd.DataFrame({header: np.full(n_rows, 0.0)}).to_csv(tol_file)
    assert run_check(ref_file, cur_file, tol_file).exit_code == 1


def run_fof_compare(file1, file2, extra_args):
    return CliRunner().invoke(
        fof_compare,
        [
            "--file1",
            file1.replace("AIREP", "{fof_type}"),
            "--file2",
            file2.replace("AIREP", "{fof_type}"),
            "--fof-types",
            "AIREP",
            "--tolerance",
            "1e-3",
            *extra_args,
        ],
    )


def test_fof_compare_tolerance_vars(fof_ensemble, monkeypatch, caplog):
    ens_dir, cur_file = fof_ensemble
    ref_file = os.path.join(ens_dir, "fofAIREP_.nc")
    monkeypatch.chdir(ens_dir)

    with caplog.at_level(logging.INFO):
        result = run_fof_compare(ref_file, cur_file, [])
    assert result.exit_code == 0, result.output
    assert "Files are NOT consistent!" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        result = run_fof_compare(
            ref_file, cur_file, ["--fof-tolerance-vars", "veri_data,e_o"]
        )
    assert result.exit_code == 0, result.output
    assert "Files are consistent!" in caplog.text


@pytest.mark.parametrize(
    "tol_vars, rules, match",
    [
        ("veri_data,level", "{}", "sort keys"),
        ("veri_data,state", '{"state": [1, 5]}', "--rules"),
        ("veri_data,not_a_var", "{}", "not found"),
    ],
)
def test_fof_compare_invalid_tolerance_vars(
    fof_ensemble, monkeypatch, tol_vars, rules, match
):
    ens_dir, cur_file = fof_ensemble
    ref_file = os.path.join(ens_dir, "fofAIREP_.nc")
    monkeypatch.chdir(ens_dir)

    result = run_fof_compare(
        ref_file, cur_file, ["--fof-tolerance-vars", tol_vars, "--rules", rules]
    )
    assert isinstance(result.exception, ValueError)
    assert match in str(result.exception)
