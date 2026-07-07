# DatasetWriter Implementation Summary

This directory contains two complementary implementation plans:

## 1. DATASETWRITER_PLAN.md

Complete implementation plan for adding incremental write capability to CMOR4.

**Key Features:**
- Incremental time-series writes (write one chunk at a time)
- Append mode (extend existing files with new time records)
- Preserve mode (reuse metadata across multiple output files)
- Memory-bounded operation using Zarr staging

**Implementation Status:**
- **Phase 0:** Validation pipeline refactoring - ✅ **COMPLETED AND MERGED**
- **Phase 1:** Core DatasetWriter with Zarr staging (3-4 days) - READY TO START
- **Phase 2:** Append mode (2-3 days)
- **Phase 3:** Preserve mode and zfactors (2-3 days)
- **Phase 4:** Integration and documentation (2-3 days)

**Remaining Effort:** 10-13 days (Phase 0 complete)

## 2. VALIDATION_REFACTOR_PLAN.md

Detailed analysis of validation code overlap and refactoring strategy.

**Status:** ✅ **COMPLETED AND MERGED**

**Original Problem:** 
- `create_dataset()` had ~400 lines of validation logic
- `DatasetWriter` would need ~60% of the same validation
- Without refactoring, this would create code duplication and maintenance burden

**Solution Implemented:**
- ✅ Extracted validation into `src/cmor4/utils/validation.py`
- ✅ Extracted construction into `src/cmor4/utils/construction.py`
- ✅ Replaced underscore files (`_axis_validation.py`, `_variable_validation.py`) with utils modules
- ✅ Both `cmorize()` and `DatasetWriter` can now reuse the same validation pipeline

**Benefits Achieved:**
1. ✅ Eliminated ~400 lines of code duplication
2. ✅ Ensures validation consistency between APIs
3. ✅ Created reusable infrastructure for future features
4. ✅ Makes codebase more maintainable and testable
5. ✅ No breaking changes - all existing tests pass

## Relationship Between Plans

```
VALIDATION_REFACTOR_PLAN.md
    ↓
    Provides detailed analysis and justification
    ↓
DATASETWRITER_PLAN.md - Phase 0
    ↓
    Implements the refactoring
    ↓
DATASETWRITER_PLAN.md - Phases 1-4
    ↓
    Uses refactored validation for DatasetWriter
```

## Quick Start

1. Read `VALIDATION_REFACTOR_PLAN.md` to understand the validation architecture (already implemented)
2. Read `DATASETWRITER_PLAN.md` for the complete implementation plan
3. ~~Start with Phase 0 (validation refactoring)~~ ✅ Phase 0 complete
4. Proceed to Phases 1-4 (DatasetWriter implementation) - **START HERE**

## Key Decisions

- ✅ Use Zarr-backed staging (skip in-memory collector)
- ✅ Use xarray for NetCDF finalization (~20-70 MB overhead, simpler code)
- ✅ Auto-detect time axis (check for `axis="T"` or name matching "time*")
- ✅ Auto-cleanup staging directory on success, keep on error
- ✅ Refactor validation pipeline before implementing DatasetWriter

## Files to Create/Modify

**Phase 0:** ✅ COMPLETED
- ✅ Created: `src/cmor4/utils/` directory
- ✅ Created: `src/cmor4/utils/__init__.py`
- ✅ Created: `src/cmor4/utils/validation.py`
- ✅ Created: `src/cmor4/utils/construction.py`
- ✅ Renamed: `src/cmor4/core.py` → `src/cmor4/dataset.py`
- ✅ Modified: `src/cmor4/dataset.py` (refactored to use utils)
- ✅ Removed: `src/cmor4/_axis_validation.py` (replaced by utils)
- ✅ Removed: `src/cmor4/_variable_validation.py` (replaced by utils)

**Phases 1-4:** READY TO START
- New: `src/cmor4/writer.py`
- New: `src/cmor4/utils/writer_helpers.py`
- Modify: `src/cmor4/__init__.py`
- Modify: `pyproject.toml`
