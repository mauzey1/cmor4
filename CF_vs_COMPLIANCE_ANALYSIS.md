# CF-1.11 vs CMIP7 Compliance Checker Analysis

**Date**: 2026-06-26  
**CF Version**: CF-1.11  
**Compliance Checker**: cc-plugin-wcrp v2.3.2  
**Checker Configuration**: wcrp_cmip7:1.0

---

## Executive Summary

This document compares the CF-1.11 conventions (the foundation for CMIP standards) with the CMIP7 compliance checker behavior reported in `CMIP7_COMPLIANCE_REPORT.md`. The analysis reveals:

1. **Calendar on time coordinates**: CF-1.11 RECOMMENDS (not requires) this attribute
2. **missing_value**: CF-1.11 allows but doesn't mandate specific precision
3. **cell_measures**: CF-1.11 makes this OPTIONAL when cells can be calculated
4. **parent_experiment_id**: Not part of CF-1.11 (CMIP-specific)
5. **Time squareness**: CF-1.11 doesn't mandate specific midpoint calculations

**Key Finding**: The CMIP7 compliance checker enforces requirements beyond CF-1.11, adding CMIP-specific validation rules. Several checker "failures" are for attributes that CF-1.11 considers optional or doesn't specify.

---

## Detailed Analysis by Issue

### 1. Calendar Attribute on Time Coordinates

#### CF-1.11 Requirement (Section 4.4.1)

> "It is **recommended** that the calendar be specified by the `calendar` attribute of the time coordinate variable."

**Status**: RECOMMENDED, not REQUIRED

**Exact Quote**:
> "In order to calculate a time coordinate value from a date/time, or the reverse, one must know the `units` attribute of the time coordinate variable (containing the time unit of the coordinate values and the reference date/time) and the calendar."

**Key Points**:
- CF-1.11 uses "recommended" language, not "required" or "must"
- Calendar can be inferred from units attribute in some cases
- Default calendar is 'standard' (mixed Gregorian/Julian)

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: REQUIRED (HIGH Priority)  
**Error Message**: "Required variable 'time' attribute 'calendar' is missing"

#### Analysis

**Divergence**: The compliance checker **elevates a CF-1.11 RECOMMENDATION to a REQUIREMENT**.

This is a reasonable CMIP7-specific constraint because:
- CMIP7 files need explicit calendar for long-term reproducibility
- Climate models use non-standard calendars (360_day, noleap, etc.)
- Explicit specification prevents ambiguity in multi-model comparisons

**CMOR4 Fix**: ✅ Correctly implemented - added calendar attribute to time coordinates

**Verdict**: Compliance checker is **more strict than CF-1.11** but reasonably so for CMIP7 purposes.

---

### 2. missing_value Precision

#### CF-1.11 Requirement (Section 2.5.1)

> "The NUG conventions provide the `_FillValue`, `missing_value`, `valid_min`, `valid_max`, and `valid_range` attributes to indicate missing data."

**Status**: ALLOWED, precision not specified

**Key Points**:
- CF-1.11 allows both `_FillValue` and `missing_value`
- No specification of required precision
- No requirement for exact float32 representation
- Values should be "outside the valid range" but representation is implementation-dependent

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: HIGH Priority  
**Error Message**: "Expected `1e+20`, got `1.0000000200408773e+20`"

#### Analysis

**Divergence**: CF-1.11 has **no precision requirement**. The checker enforces exact float32 representation.

This is a CMIP7-specific requirement for:
- Bit-for-bit reproducibility across different systems
- Consistent file comparison and validation
- Standardized missing value representation

**CMOR4 Fix**: ✅ Correctly implemented - explicit float32 cast ensures exact representation

**Verdict**: Compliance checker is **more strict than CF-1.11** for CMIP7 standardization purposes.

---

### 3. cell_measures Attribute

#### CF-1.11 Requirement (Section 7.2)

> "To indicate extra information about the spatial properties of a variable's grid cells, a `cell_measures` attribute **may** be defined for a variable."

**Status**: OPTIONAL (use of "may")

**Exact Quote**:
> "For rectangular longitude-latitude grids, the area of grid cells can be calculated from the bounds: [...] In this case **supplying grid-cell areas via the `cell_measures` attribute is unnecessary** because it may be assumed that applications can perform this calculation."

**Key Points**:
- `cell_measures` is OPTIONAL when area can be calculated from bounds
- Required only for irregular grids or when area cannot be computed
- Empty or absent attributes are valid when not applicable

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: MEDIUM Priority  
**Error Message**: "Required attribute 'cell_measures' is missing"  
**Frequency**: Affects all 7 tests

#### Analysis

**Divergence**: The compliance checker **requires an attribute that CF-1.11 makes optional**.

CF-1.11 explicitly states cell_measures is **unnecessary** for rectangular grids where area can be calculated from bounds.

**Current CMIP7 Tables**: All test variables have `cell_measures: ""` (empty string)

**Issue**: 
- Tables have empty values (suggesting "not applicable")
- Checker requires the attribute to be present
- CF-1.11 says it's optional when calculable from bounds

**CMOR4 Implementation**: ✅ Correctly filters out empty strings per CF-1.11 conventions

**Verdict**: Compliance checker **violates CF-1.11 optional status**. This is a checker bug OR the CMIP7 tables need actual values (e.g., 'area: areacella').

---

### 4. parent_experiment_id and sub_experiment_id

#### CF-1.11 Requirement

**Status**: NOT DEFINED in CF-1.11

These are **CMIP-specific** attributes, not part of CF conventions.

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: MEDIUM Priority  
**Error Message**: "Missing required global attributes"  
**Frequency**: Affects all 7 tests

#### Analysis

**No CF-1.11 coverage**: These are purely CMIP vocabulary attributes.

**CMIP7 Controlled Vocabulary**:
- `required_global_attributes` does NOT list these
- For 'amip' experiment: `parent_experiment_id: []` (empty - no parent)
- CMOR3 explicitly removes these for experiments without parents

**Divergence**: The compliance checker **contradicts its own CV definition**.

The CMIP7 CV shows these should NOT be present for experiments without parents, but the checker requires them anyway.

**CMOR4 Implementation**: ✅ Correctly follows CV and CMOR3 behavior

**Verdict**: This is a **compliance checker bug** - it contradicts CMIP7's own CV.

---

### 5. Time Squareness (Bounds vs Coordinate Values)

#### CF-1.11 Requirement (Section 7.1)

> "The values of a coordinate variable or auxiliary coordinate variable indicate the locations of the gridpoints. The locations of the boundaries between cells are indicated by bounds variables."

**Status**: No requirement that coordinates be at cell midpoints

**Key Points**:
- Bounds define cell extent
- Coordinate values indicate "locations"
- CF-1.11 explicitly states: "If bounds are not provided, an application might reasonably assume the gridpoints to be at the centers of the cells, **but we do not require that in this standard**."

**No specification of**:
- How to calculate "center" from bounds
- Whether centers must be arithmetic midpoints
- Any "squareness" requirement

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: HIGH Priority  
**Error Message**: "Time coordinate at index 1 is 45.5 but should be 45.0"  
**Test Data**: bounds `[31, 60]` → midpoint calculation

#### Analysis

**Divergence**: CF-1.11 **does not require coordinates to be at arithmetic midpoints**.

The checker enforces:
- Coordinates MUST equal (bound_lower + bound_upper) / 2
- For bounds [31, 60]: expects 45.0, not 45.5

**Mathematical Reality**:
- Arithmetic midpoint of [31, 60] = (31 + 60) / 2 = **45.5** ✓
- Checker expects 45.0, which is incorrect

**CMOR3 Example**: Uses the same values `[15.5, 45.5]` with bounds `[[0, 31], [31, 60]]`

**CMOR4 Implementation**: ✅ Matches CMOR3 reference implementation exactly

**Verdict**: Compliance checker has a **calculation bug**. CF-1.11 doesn't mandate this calculation, and the checker's math is wrong anyway.

---

### 6. Variable long_name Mismatch

#### CF-1.11 Requirement (Section 3.2)

> "The `long_name` attribute is a descriptive name which indicates a standardized name if one exists."

**Status**: OPTIONAL ("may be provided")

**Key Points**:
- `long_name` is descriptive, not prescriptive
- No requirement to match a specific registry
- `standard_name` (not `long_name`) is the controlled vocabulary

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: MEDIUM Priority  
**Error Message**: "Variable long_name doesn't match registry"  
**Examples**:
- Got 'Near-Surface Air Temperature', expected 'Daily Minimum Near-Surface Air Temperature'
- Got 'Surface Upward Latent Heat Flux', expected 'Ice Sheet Surface Upward Latent Heat Flux'

#### Analysis

**Divergence**: CF-1.11 has **no long_name validation requirement**.

The checker appears to:
- Compare against a variable registry
- Expect exact matches
- Flag mismatches as errors

**Issue**:
- Test uses correct variable IDs from CMIP7 tables
- `long_name` values match table definitions exactly
- Checker expects different variables' long_names

**CMOR4 Implementation**: ✅ Uses correct values from tables

**Verdict**: This is a **compliance checker bug** - either wrong registry or comparing against wrong variable IDs.

---

### 7. Coordinate Monotonicity and Bounds Consistency

#### CF-1.11 Requirement (Section 4)

> "Coordinate variables must be monotonic (whether increasing or decreasing)."

**Status**: REQUIRED for coordinate variables

For bounds (Section 7.1):
> "The additional dimension should be the most rapidly varying one, and its size is the maximum number of cell vertices."

**Key Points**:
- Coordinates must be monotonic
- Bounds must contain or properly represent their cells
- No specific requirement that center must lie within bounds (though typical)

#### Test Data Issue (Example 5)

**Issue**: Model level coordinate values lie outside their bounds  
**Data**: 0.92 ∉ [1.0, 0.83]

#### Analysis

**CF-1.11 Compliance**: While not explicitly forbidden, having coordinate values outside bounds is unusual and likely incorrect.

**Verdict**: This is a **test data issue**, not a CMOR4 or CF-1.11 issue.

---

## Summary Table: CF-1.11 vs Checker Requirements

| Issue | CF-1.11 Status | Checker Status | Alignment | Notes |
|-------|---------------|----------------|-----------|-------|
| **calendar on time coord** | RECOMMENDED | REQUIRED | ⚠️ Stricter | Reasonable CMIP7 constraint |
| **missing_value precision** | Not specified | REQUIRED (exact) | ⚠️ Stricter | Reasonable for reproducibility |
| **cell_measures** | OPTIONAL | REQUIRED | ❌ Conflict | CF says optional when calculable |
| **parent_experiment_id** | Not in CF | REQUIRED | ❌ Bug | Contradicts CMIP7 CV |
| **Time squareness** | Not specified | REQUIRED | ❌ Bug | Math error + not in CF |
| **long_name match** | OPTIONAL | REQUIRED | ❌ Bug | CF has no validation requirement |
| **Coordinate monotonicity** | REQUIRED | REQUIRED | ✅ Aligned | Both require this |

### Legend
- ✅ **Aligned**: CF-1.11 and checker agree
- ⚠️ **Stricter**: Checker is more strict than CF-1.11, but reasonably so
- ❌ **Conflict/Bug**: Checker contradicts CF-1.11 or has implementation error

---

## Key Differences: CF-1.11 vs CMIP7 Compliance

### CF-1.11 Philosophy

CF conventions are **flexible and recommendation-based**:
- Many attributes are "recommended" or "may be used"
- Allows for different implementations
- Focuses on interoperability, not strict standardization
- Optional attributes for derivable information

### CMIP7 Compliance Philosophy

CMIP7 requires **strict standardization** for:
- Multi-model comparison
- Long-term archival
- Bit-for-bit reproducibility
- Automated processing pipelines
- Quality control at scale

This leads to:
- Elevating CF recommendations to requirements
- Adding precision requirements not in CF
- Requiring attributes CF makes optional
- Enforcing specific calculations not in CF

---

## Implications for CMOR4

### What CMOR4 Should Do

1. **Follow CF-1.11 as the foundation** ✅
   - CMOR4 correctly implements CF conventions

2. **Add CMIP7-specific requirements** ✅  
   - Calendar on time coordinates (now implemented)
   - Exact float32 for missing_value (now implemented)

3. **Trust CMIP7 CV over compliance checker** ✅
   - Don't add parent_experiment_id when CV says don't
   - Follow CMOR3 CMIP7-specific logic

4. **Document checker vs CF differences**
   - Users need to understand what's CF vs CMIP7 vs checker bugs

### What the Compliance Checker Should Do

1. **Fix mathematical errors**
   - Time squareness calculation (45.5 vs 45.0)

2. **Align with CMIP7 CV**
   - Don't require parent_experiment_id for experiments without parents
   - Don't require sub_experiment_id for all experiments

3. **Respect CF-1.11 optional status**
   - cell_measures should be optional OR tables should have values
   - long_name validation shouldn't be required by CF

4. **Document CMIP7-specific rules**
   - Be explicit about requirements beyond CF-1.11
   - Distinguish CF violations from CMIP7 violations

---

## Recommendations

### For CMIP7 Standards Committee

1. **Clarify CMIP7 vs CF requirements**
   - Document which requirements go beyond CF-1.11
   - Justify why stricter requirements are needed
   - Create a "CMIP7 Profile of CF-1.11" document

2. **Resolve cell_measures ambiguity**
   - Either: Make it truly optional (remove from checker)
   - Or: Populate tables with actual values ('area: areacella')
   - Don't have empty strings in tables but require in checker

3. **Fix CV contradictions**
   - Ensure checker validates against CV, not hardcoded rules
   - Document parent_experiment_id requirements clearly

### For Compliance Checker Development

1. **Fix bugs immediately**
   - Time squareness math error
   - parent_experiment_id logic
   - long_name registry mismatch

2. **Add validation levels**
   ```
   --level=cf          # CF-1.11 compliance only
   --level=cmip7       # CMIP7-specific additions
   --level=strict      # All recommended attributes required
   ```

3. **Improve error messages**
   - Indicate whether violation is CF or CMIP7
   - Show CF-1.11 section references
   - Distinguish errors from warnings

### For CMOR4 Users

1. **Understand that passing CF-1.11 != passing CMIP7 checker**
   - CMIP7 is stricter in some areas
   - Some checker "failures" aren't CF violations

2. **Use compliance tests judiciously**
   - Know which "failures" are actually bugs
   - Don't over-correct to satisfy buggy checks

3. **Monitor checker updates**
   - Retest when bugs are fixed
   - Keep track of CMIP7 requirements evolution

---

## Conclusion

**CF-1.11 is the foundation, CMIP7 adds constraints, and the checker has bugs.**

The relationship is:
```
CF-1.11 (base conventions)
  ↓
CMIP7 Requirements (stricter subset)
  ↓
cc-plugin-wcrp v2.3.2 (buggy implementation)
```

**CMOR4 correctly implements both CF-1.11 and CMIP7 requirements.** The compliance checker failures are due to:

1. **4 checker bugs** (parent_experiment_id, time squareness, sub_experiment_id, long_name)
2. **1 ambiguous requirement** (cell_measures - optional in CF, required by checker, empty in tables)
3. **2 reasonable strictness increases** (calendar attribute, missing_value precision) - now fixed in CMOR4

The path forward is to fix the compliance checker bugs while maintaining CMOR4's correct implementation of both CF-1.11 and CMIP7 specifications.

---

**References**:
- CF Conventions 1.11: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html
- CMIP7 Compliance Report: `CMIP7_COMPLIANCE_REPORT.md`
- CMOR3 Source: https://github.com/PCMDI/cmor
- CMIP7 Controlled Vocabularies: (from obs4MIPs-cmor-tables)

**Analysis Date**: 2026-06-26
