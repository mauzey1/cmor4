# Independent Review: CF-1.13 vs WCRP CMIP7 Compliance Checker

**Date**: 2026-06-26  
**Reviewed file**: `CF_1.13_vs_COMPLIANCE_ANALYSIS.md`  
**Checker inspected**: `cc-plugin-wcrp` 2.3.2, `wcrp_cmip7:1.0`  
**Scope**: Whether the report's assessments are justified when the checker behavior is compared with CF-1.13, and where the issue is instead a CMIP7 profile/table/checker concern.

## Summary

I agree with the central distinction in the reviewed report: several WCRP CMIP7 checker failures are not CF-1.13 failures. CF-1.13 is generally less prescriptive than the CMIP7 checker about project metadata, variable registry values, exact missing-value constants, and monthly timestamp conventions.

I do not agree with all of the report's labels and evidence. The checker is often stricter than CF because it is a CMIP7 profile checker, not just a CF checker. Some findings called "violates CF" should be described as "outside CF" or "stricter than CF." The time-squareness finding also needs correction: the installed checker does not compare against the supplied bounds midpoint; it reconstructs an expected CMIP monthly time axis from filename start, frequency, table_id, and calendar. With `calendar="360_day"` and `frequency="mon"`, it expects 30-day calendar months, so `15.0` and `45.0`, not `15.5` and `45.5`.

## Evidence Checked

- CF-1.13 official text:
  - Time `calendar`: Section 4.4.3 says it is recommended that a time coordinate variable should have a `calendar` attribute rather than relying on the default.
  - Cell boundaries: Section 7.1 says that if boundaries are provided, each gridpoint should lie somewhere within or upon its own cell boundaries.
  - `cell_measures`: Section 7.2 says the attribute may be defined; for rectangular lat/lon grids, supplying cell areas is unnecessary when they can be calculated from bounds.
  - `long_name`: Section 3.2 says `long_name` is optional and ad hoc; `standard_name` is the CF-controlled identifier.
  - Missing values: Section 2.5.1 discusses `_FillValue` and `missing_value`, but does not require CMIP's exact `1.0e20` value or a particular printed precision.
- Installed checker behavior:
  - `plugins/cmip7/config/wcrp/coordinate_variables.toml` requires `time:calendar` and enables `TIME001`.
  - `checks/time_checks/check_time_squareness.py` reconstructs expected times from filename start and frequency, using calendar-aware month increments.
  - `plugins/cmip7/config/wcrp/geophysical_variable.toml` requires `cell_measures`, `long_name`, `_FillValue`, and `missing_value`, and validates some fields against ESGVOC registry terms.
  - `checks/consistency_checks/check_experiment_consistency.py` records `parent_experiment_id` and `sub_experiment_id` as missing before it checks whether the experiment actually needs them.
- Current local compliance run:
  - `./venv/bin/pytest tests/test_cmip7_examples.py --run-compliance -k test_cmip7_compliance -q`
  - Result: all 7 compliance tests fail.
  - Common current failures: missing `cell_measures`, missing `parent_experiment_id` and `sub_experiment_id`, source vocabulary check for `DUMMY-MODEL`, and `TIME001` for time-bearing files.
  - Current `TIME001` message for Example 1: expected `15.000000`, got `15.500000` at index 0.

## Assessment By Issue

### Calendar Attribute

The reviewed report is mostly right. CF-1.13 recommends `calendar`; it does not require it. The CMIP7 checker requires it at high severity through the WCRP plugin configuration.

This is not a CF-1.13 requirement, but it is a defensible CMIP7 profile rule. The report's wording that CF-1.13 "strengthens" the recommendation is broadly fair, though "recommended" remains non-mandatory.

Verdict: agree, but classify as stricter CMIP7 profile behavior, not a checker bug.

### Time Squareness / Monthly Timestamps

The reviewed report is only partly right. It is correct that CF-1.13 does not require a coordinate value to be the arithmetic midpoint of its bounds. It is also correct that `15.5` and `45.5` lie within the supplied bounds `[[0, 31], [31, 60]]`, so this specific data is not rejected by CF-1.13 on the basis of Section 7.1.

However, the report mischaracterizes the installed checker's calculation. The checker is not computing `(lower + upper) / 2` from the supplied bounds and getting the wrong answer. It builds a theoretical axis from the filename start, `frequency="mon"`, table_id, and `calendar="360_day"`. In a 360-day calendar, a calendar month is 30 days, so the expected midpoint of the first monthly interval is 15.0. The local checker output confirms this.

So the issue is not "incorrect arithmetic." It is a CMIP profile assumption that monthly mean time coordinates should align with calendar month intervals. The test data uses 31-day/29-day bounds while declaring `360_day`; that can be CF-valid, but it is not the monthly 360-day CMIP convention the checker is enforcing.

Verdict: disagree with the "checker math bug" framing. Agree that this is not a CF-1.13 violation. For CMIP7 compliance, either the test data should use `time=[15.0, 45.0]`, `time_bnds=[[0, 30], [30, 60]]`, or the checker should document and scope the profile rule clearly.

### `cell_measures`

The report is right about CF-1.13: `cell_measures` is optional, and CF explicitly says grid-cell areas are unnecessary for rectangular lat/lon grids when calculable from bounds.

The report is too strong when it says the checker "violates CF." A stricter profile is allowed to require extra metadata. The better criticism is consistency: the main CMIP7 variable tables contain empty `cell_measures` fields, while the companion `CMIP7_cell_measures.json` is mostly populated and the CMIP7 README says model groups should use that companion file. CMOR4 currently loads the main variable table metadata, so it does not automatically add those companion values.

Verdict: agree this is not a CF-1.13 requirement. Treat as a CMIP7 table/integration/profile issue, not a pure checker bug.

### `missing_value` Precision

The report is right that CF-1.13 does not specify the CMIP checker's exact `1.0e20` value or printed precision. The checker requires `_FillValue` and `missing_value` to equal `1.0e20` through CMIP7 config.

This is stricter than CF but reasonable for a project profile. One implementation detail matters: the checker compares constants as strings after reading the attribute, which can make type representation visible in failures.

Verdict: agree with the report's practical conclusion: this is a CMIP7 reproducibility/profile requirement, not a CF requirement.

### `parent_experiment_id` and `sub_experiment_id`

The reviewed report is right that these are not CF-1.13 concepts. They are CMIP/ESGVOC project metadata.

The installed checker appears buggy here. `check_experiment_consistency.py` unconditionally attempts to read both attributes and reports them missing, even when the experiment may not require them. That is independent of CF-1.13 and should be fixed in the checker logic.

Verdict: agree this is a checker bug, but the argument should be based on CMIP7 CV semantics, not CF.

### `long_name` Validation

The report is right that CF-1.13 does not validate `long_name` against a registry. CF explicitly describes `long_name` usage as ad hoc and optional.

For CMIP7, however, the checker deliberately uses ESGVOC registry terms. That can be a legitimate project-profile rule. The local failures for `tas` and `hfls` look more like registry lookup ambiguity or branded-variable/variable-id mismatch than a CF problem.

Verdict: agree this is not a CF-1.13 requirement. Do not call it a CF checker bug. Call it a CMIP7 registry/profile issue unless the checker is demonstrably resolving the wrong term.

## Corrections To The Reviewed Report

1. The current time-squareness failure is `expected 15.000000, got 15.500000` at index 0, not `expected 45.0, got 45.5` at index 1.
2. The time-squareness check is calendar/frequency reconstruction, not arithmetic midpoint-of-provided-bounds validation.
3. `CMIP7_cell_measures.json` is not empty. It contains 1,974 entries, 1,744 of them nonblank in this checkout. The empty values are in many main variable-table `cell_measures` fields; the companion file is intended to reintroduce this metadata.
4. "Violates CF" is too strong for several checker rules. A project profile can be stricter than CF. The real question is whether CMIP7 documents those stricter rules and whether the checker applies them consistently.
5. The broad claim that CMOR4 is "fully CF-1.13 compliant" is not proven by the analysis. For example, local tests currently treat `utc` and `tai` calendars as invalid because `cftime` does not support them, while CF-1.13 defines those calendar values.

## Recommended Actions

1. Update the reviewed report's time-squareness section to distinguish CF validity from CMIP monthly-axis expectations.
2. For CMIP7 example data using `calendar="360_day"` and `frequency="mon"`, use 30-day monthly bounds and midpoints if the goal is to pass the WCRP checker.
3. Decide whether CMOR4 should ingest `CMIP7_cell_measures.json` and `CMIP7_long_name_overrides.json` as part of CMIP7 table preparation, since the CMIP7 examples explicitly apply those companion files.
4. Report the unconditional `parent_experiment_id` / `sub_experiment_id` missing-attribute behavior to the checker maintainers.
5. Treat `long_name` failures as CMIP7 registry resolution/profile issues, not CF-1.13 failures.

## Bottom Line

I agree with the report's main conclusion that several checker failures are outside CF-1.13. I disagree with the stronger claim that the checker is broadly "violating CF" or doing wrong midpoint arithmetic. The more accurate framing is:

- CF-1.13: permissive on calendar presence, bounds midpoint placement, `cell_measures`, `long_name`, project metadata, and missing-value constants.
- CMIP7 checker: stricter project profile, with some legitimate extra requirements.
- Actual checker bugs or likely bugs: unconditional parent/sub-experiment reporting, and possibly registry resolution for `long_name`.
- CMOR4/test-data issues to consider for checker compliance: 360-day monthly time bounds/midpoints and companion table handling for `cell_measures`.
