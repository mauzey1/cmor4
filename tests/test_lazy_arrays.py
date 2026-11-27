"""Lazy (dask) arrays must stay lazy through dataset assembly.

``build_dataset`` is shared by ``cmorize()`` and ``DatasetWriter``. On
``close()``, the writer validates the metadata, then replaces the data and the
z-factor values with lazy arrays read from its staging store, and only then calls
``build_dataset``. So ``build_dataset`` must not compute the data or the
z-factor values. Any refactor of ``_assembly`` must keep this test passing.
"""

from __future__ import annotations

from dataclasses import replace

import dask
import dask.array as da
import numpy as np
import pytest

from cmor import _assembly
from dataset_helpers import dataset_info, horizontal_axes, time_axis
from table_helpers import cmip7_project


def _refuse_to_compute(*args, **kwargs):
    raise AssertionError("build_dataset computed a lazy dask array")


@pytest.fixture(scope="module")
def project():
    return cmip7_project()


def test_lazy_data_and_zfactor_stay_lazy_through_build_dataset(project, tmp_path):
    info = project.dataset_info(dataset_info(tmp_path))
    variable = project.variable("tnhusscpbl_tavg-al-hxy-u", table_id="atmos")
    axes = [
        time_axis(project, values=[15.0, 45.0], bounds=[[0.0, 30.0], [30.0, 60.0]]),
        project.axis(
            "standard_hybrid_sigma",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        *horizontal_axes(project),
    ]
    ps = np.full((2, 2, 2), 99000.0, dtype="f4")
    zfactors = [
        project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
        project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
        project.zfactor("p0", values=100000.0),
        project.zfactor("ps", values=ps),
    ]
    data = da.ones((2, 2, 2, 2), dtype="f4", chunks=1)

    # Validation reads concrete values, as in DatasetWriter.close().
    ctx = _assembly.validate_metadata(info, variable, axes, zfactors)
    lazy_ps = da.from_array(ps, chunks=1)
    ctx = replace(
        ctx,
        zfactors=tuple(
            zf.updated(values=lazy_ps) if zf.name == "ps" else zf for zf in ctx.zfactors
        ),
    )

    with dask.config.set(scheduler=_refuse_to_compute):
        ds = _assembly.build_dataset(ctx, data)

    var_name = variable.names()[0]
    assert dask.is_dask_collection(ds[var_name].data)
    assert dask.is_dask_collection(ds["ps"].data)
    np.testing.assert_array_equal(ds[var_name].values, np.ones((2, 2, 2, 2)))
    np.testing.assert_array_equal(ds["ps"].values, ps)
