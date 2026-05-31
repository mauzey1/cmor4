from __future__ import annotations

from pathlib import Path

import cmor4

PROJECT_TABLE_ROOT = Path(__file__).resolve().parents[1] / "project_tables"
CMIP7_TABLE_ROOT = PROJECT_TABLE_ROOT / "cmip7-cmor-tables"
DRCDP_TABLE_ROOT = PROJECT_TABLE_ROOT / "DRCDP"
OBS4MIPS_TABLE_ROOT = PROJECT_TABLE_ROOT / "obs4MIPs-cmor-tables"


def cmip7_project(*variable_tables: str) -> cmor4.ProjectTables:
    tables = variable_tables or (
        "tables/CMIP7_atmos.json",
        "tables/CMIP7_land.json",
        "tables/CMIP7_ocean.json",
        "tables/CMIP7_seaIce.json",
    )
    return cmor4.ProjectTables.from_directory(
        CMIP7_TABLE_ROOT,
        cv_file="tables-cvs/cmor-cvs.json",
        variable_tables=tables,
    )


def drcdp_project(*variable_tables: str) -> cmor4.ProjectTables:
    tables = variable_tables or (
        "Tables/DRCDP_AP1hr.json",
        "Tables/DRCDP_APday.json",
    )
    return cmor4.ProjectTables.from_directory(
        DRCDP_TABLE_ROOT,
        cv_file="Tables/DRCDP_CV.json",
        variable_tables=tables,
    )


def obs4mips_project(*variable_tables: str) -> cmor4.ProjectTables:
    tables = variable_tables or (
        "Tables/obs4MIPs_Amon.json",
        "Tables/obs4MIPs_A1hrPt.json",
    )
    return cmor4.ProjectTables.from_directory(
        OBS4MIPS_TABLE_ROOT,
        cv_file="Tables/obs4MIPs_CV.json",
        variable_tables=tables,
    )
