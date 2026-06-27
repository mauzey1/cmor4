# CF-1.13 vs CMIP7 Compliance Checker Analysis

**Date**: 2026-06-26  
**CF Version**: CF-1.13 (latest)  
**Previous Analysis**: CF-1.11  
**Compliance Checker**: cc-plugin-wcrp v2.3.2  
**Checker Configuration**: wcrp_cmip7:1.0

---

## Executive Summary

This document compares CF-1.13 (the latest CF conventions) with the CMIP7 compliance checker behavior. CF-1.13 includes several clarifications and improvements over CF-1.11, particularly around calendar handling and cell boundaries. Key findings:

1. **Calendar attribute**: CF-1.13 strengthens the language to "**recommended**" (was "recommended" in 1.11, but now more emphatic)
2. **Cell boundaries and gridpoint location**: CF-1.13 adds NEW RECOMMENDATION that gridpoints should lie within cell bounds
3. **cell_measures**: Still OPTIONAL (unchanged from CF-1.11)
4. **missing_value**: Still no precision requirements (unchanged)
5. **Time coordinate requirements**: More structured and explicit

**Key Difference from CF-1.11**: CF-1.13 adds recommendations about gridpoint locations relative to bounds, but does NOT mandate arithmetic midpoint calculations.

---

## CF-1.13 Changes Relevant to Compliance Issues

### New or Enhanced Sections in CF-1.13

1. **Section 4.4 restructured** into subsections:
   - 4.4.1. Time Coordinate Variables
   - 4.4.2. Time Coordinate Units
   - 4.4.3. Calendar
   - 4.4.4. Time Coordinates with no Annual Cycle
   - 4.4.5. Explicitly Defined Calendar

2. **Section 7.1 enhanced** with explicit guidance on cell boundaries and gridpoint locations

3. **Table 4.1 added**: Comprehensive calendar characteristics table

---

## Detailed Analysis by Issue

### 1. Calendar Attribute on Time Coordinates

#### CF-1.13 Requirement (Section 4.4.1)

**Example 4.4 shows**:
```
time:calendar = "standard" ;               // calendar attribute is recommended
time:units = "days since 1990-1-1 0:0:0" ; // units attribute is mandatory
```

**Section 4.4.3 states**:
> "It is **recommended** that the time coordinate variable should have a `calendar` attribute (rather than relying on the default)."

**Status**: RECOMMENDED (stronger emphasis than CF-1.11)

#### Changes from CF-1.11

**CF-1.11** said:
> "It is recommended that the calendar be specified by the `calendar` attribute"

**CF-1.13** says:
> "It is recommended that the time coordinate variable **should** have a `calendar` attribute (rather than relying on the default)."

**Key Differences**:
- More direct statement: "should have" vs "be specified"
- Explicitly discourages relying on defaults
- Includes comprehensive Table 4.1 listing all calendar characteristics

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: REQUIRED (HIGH Priority)  
**Error Message**: "Required variable 'time' attribute 'calendar' is missing"

#### Analysis

**CF-1.13 Status**: Still RECOMMENDED, not REQUIRED

However, CF-1.13's stronger recommendation language and explicit table of calendars suggests this is a critical attribute for interoperability, especially for climate data.

**CMOR4 Fix**: ✅ Correctly implemented

**Verdict**: Compliance checker elevates RECOMMENDATION to REQUIREMENT. This is reasonable for CMIP7, and CF-1.13's strengthened language supports this decision.

---

### 2. Time Squareness and Cell Boundaries

#### CF-1.13 Requirement (Section 7.1)

**NEW in CF-1.13**:
> "If cell boundaries are provided, it is **recommended** that each gridpoint should lie somewhere **within or upon** the boundaries of its own cell."

**Key phrases**:
- "somewhere within or upon" (not "at the center")
- "recommended" (not required)
- No specification of HOW to calculate the gridpoint location

**Section 7.1 also states**:
> "If cell boundaries are not provided (using the `bounds` attribute), an application can make **no assumption** about the location or extent of the cells."

#### Changes from CF-1.11

**CF-1.11** said:
> "If bounds are not provided, an application might reasonably assume the gridpoints to be at the centers of the cells, but we do not require that in this standard."

**CF-1.13** adds:
- Explicit recommendation that gridpoints lie within bounds
- Clarification that without bounds, no assumptions can be made
- Guidance for zero-size cells (coincident boundaries)

**Still NOT specified**:
- Arithmetic midpoint requirement
- "Squareness" calculations
- Specific formulas for gridpoint locations

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: HIGH Priority  
**Error Message**: "Time coordinate at index 1 is 45.5 but should be 45.0"  
**Test Data**: bounds `[31, 60]` → expects 45.0, not 45.5

#### Analysis

**CF-1.13 Compliance**:
- ✅ Value 45.5 lies within bounds [31, 60] (satisfies "within or upon")
- ✅ No arithmetic midpoint requirement exists
- ❌ Checker enforces calculation not in CF-1.13
- ❌ Checker's expected value (45.0) is OUTSIDE the mathematical midpoint

**Mathematical Check**:
- Bounds: [31, 60]
- Midpoint: (31 + 60) / 2 = 45.5 ✓
- Checker expects: 45.0
- Is 45.0 within [31, 60]? YES
- Is 45.5 within [31, 60]? YES
- But the checker says 45.5 is wrong and 45.0 is right!

**CMOR4 Implementation**: ✅ Value lies within bounds per CF-1.13 recommendation

**Verdict**: 
1. CF-1.13 does NOT require any specific calculation
2. Checker enforces a calculation not in CF-1.13
3. Checker's expected value contradicts basic arithmetic
4. This is a **compliance checker bug with incorrect math**

---

### 3. cell_measures Attribute

#### CF-1.13 Requirement (Section 7.2)

**Exact same wording as CF-1.11**:
> "To indicate extra information about the spatial properties of a variable's grid cells, a `cell_measures` attribute **may** be defined for a variable."

> "For rectangular longitude-latitude grids, the area of grid cells can be calculated from the bounds: [...] In this case **supplying grid-cell areas via the `cell_measures` attribute is unnecessary** because it may be assumed that applications can perform this calculation."

**Status**: OPTIONAL (unchanged from CF-1.11)

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: MEDIUM Priority  
**Error Message**: "Required attribute 'cell_measures' is missing"

#### Analysis

**No change from CF-1.11 analysis**: 

CF-1.13 continues to make cell_measures OPTIONAL and explicitly states it's "unnecessary" for rectangular grids.

**CMOR4 Implementation**: ✅ Correctly follows CF-1.13

**Verdict**: Compliance checker violates CF-1.13 optional status (same as CF-1.11)

---

### 4. missing_value Precision

#### CF-1.13 Requirement (Section 2.5.1)

**Similar to CF-1.11**:
> "NUG Appendix A, Attribute Conventions provide the `_FillValue`, `missing_value`, `valid_min`, `valid_max`, and `valid_range` attributes to indicate missing data."

**Enhanced guidance**:
- More detailed explanation of NUG convention changes (v2.3 vs v2.4)
- Recommendation to use `_FillValue` when only one missing value needed
- Guidance on packed data and missing values

**Status**: No precision requirements (unchanged from CF-1.11)

#### CMIP7 Compliance Checker Behavior

**Requirement Level**: HIGH Priority  
**Error Message**: "Expected `1e+20`, got `1.0000000200408773e+20`"

#### Analysis

**No change from CF-1.11 analysis**:

CF-1.13 still does not specify precision requirements for missing_value.

**CMOR4 Fix**: ✅ Correctly implemented exact float32 representation

**Verdict**: Compliance checker is more strict than CF-1.13 for CMIP7 reproducibility (reasonable)

---

### 5. parent_experiment_id and sub_experiment_id

#### CF-1.13 Coverage

**Status**: NOT DEFINED in CF-1.13 (same as CF-1.11)

These remain CMIP-specific global attributes, not part of CF conventions.

#### Analysis

**No change from CF-1.11 analysis**: These are CMIP-specific attributes that should follow CMIP7 CV rules.

**Verdict**: Compliance checker bug (contradicts CMIP7 CV) - unchanged from CF-1.11 analysis

---

### 6. Variable long_name Validation

#### CF-1.13 Requirement (Section 3.2)

**Status**: OPTIONAL (same as CF-1.11)

CF-1.13 maintains that `long_name` is descriptive and not validated against a registry. `standard_name` is the controlled vocabulary.

#### Analysis

**No change from CF-1.11 analysis**: long_name is not a controlled vocabulary in CF-1.13.

**Verdict**: Compliance checker bug - CF-1.13 has no long_name validation requirement

---

## Summary Table: CF-1.13 vs Checker Requirements

| Issue | CF-1.13 Status | CF-1.11 Status | Change | Checker Status | Verdict |
|-------|---------------|----------------|---------|----------------|---------|
| **calendar on time coord** | RECOMMENDED (stronger) | RECOMMENDED | ⬆️ Strengthened | REQUIRED | ⚠️ Reasonable elevation |
| **Gridpoint in bounds** | RECOMMENDED | Implicit | ➕ NEW | REQUIRED (specific calc) | ❌ Wrong calculation |
| **missing_value precision** | Not specified | Not specified | — | REQUIRED (exact) | ⚠️ Reasonable for CMIP7 |
| **cell_measures** | OPTIONAL | OPTIONAL | — | REQUIRED | ❌ Violates CF |
| **parent_experiment_id** | Not in CF | Not in CF | — | REQUIRED | ❌ Bug (CV conflict) |
| **Time squareness** | Within bounds | Not specified | ✓ Clarified | REQUIRED (wrong math) | ❌ Bug + wrong math |
| **long_name match** | OPTIONAL | OPTIONAL | — | REQUIRED | ❌ Not in CF |

### Legend
- ✅ **Aligned**: CF-1.13 and checker agree
- ⚠️ **Reasonable**: Checker is stricter but justified
- ❌ **Bug**: Checker contradicts CF-1.13 or has error
- — No change from previous version
- ⬆️ Strengthened language
- ➕ New requirement/recommendation
- ✓ Clarified

---

## Key Changes from CF-1.11 to CF-1.13

### 1. Calendar Recommendations (Strengthened)

**Impact on compliance issues**: MINOR

CF-1.13 uses more direct language ("should have") and provides a comprehensive table, but the requirement level remains RECOMMENDED.

**Implication**: The compliance checker's elevation to REQUIRED is now even more reasonable given CF-1.13's emphasis.

### 2. Gridpoint Location Guidance (NEW)

**Impact on compliance issues**: MAJOR

CF-1.13 adds explicit recommendation that gridpoints should lie "within or upon" cell boundaries.

**Key Points**:
- This is a NEW recommendation in CF-1.13
- It does NOT specify arithmetic midpoint
- It does NOT mandate "squareness"
- The test value 45.5 DOES lie within [31, 60] ✓

**Implication**: 
- CMOR4 test data (45.5) satisfies CF-1.13 recommendation
- Compliance checker requirement is still not in CF-1.13
- Checker's math error (expecting 45.0 instead of 45.5) is still wrong

### 3. Cell Boundaries Documentation (Enhanced)

**Impact on compliance issues**: NONE

CF-1.13 adds more guidance about cell boundaries, but maintains the same OPTIONAL status for cell_measures.

### 4. Time Coordinate Structure (Reorganized)

**Impact on compliance issues**: MINOR

CF-1.13 reorganizes time coordinate documentation into clearer subsections, but doesn't change requirements.

---

## CF-1.13 Specific Findings

### Finding 1: Gridpoint Within Bounds ✅

**CF-1.13 Section 7.1**:
> "it is recommended that each gridpoint should lie somewhere within or upon the boundaries of its own cell"

**CMOR4 Test Data Check**:
- Time value: 45.5
- Bounds: [31, 60]
- Is 45.5 within [31, 60]? **YES** ✓
- CF-1.13 recommendation: **SATISFIED** ✅

**Conclusion**: CMOR4 data satisfies CF-1.13 recommendation. The checker's complaint is not about CF compliance.

### Finding 2: No Midpoint Formula Specified

**What CF-1.13 Does NOT Say**:
- ❌ Gridpoint must be at arithmetic midpoint
- ❌ Gridpoint must equal (lower + upper) / 2
- ❌ Any specific calculation for gridpoint location
- ❌ "Squareness" requirement

**What CF-1.13 DOES Say**:
- ✓ Gridpoint should be "somewhere within or upon" bounds
- ✓ Location is flexible (can be anywhere in bounds)
- ✓ Without bounds, no assumptions about location

**Conclusion**: Checker enforces a requirement not in CF-1.13.

### Finding 3: Calendar Table Provides Clarity

**CF-1.13 Table 4.1** lists all calendar types with characteristics:
- Days in year
- Leap day rules
- Leap second handling
- Valid date ranges

**Implication**: CF-1.13 emphasizes calendar importance, supporting CMIP7's requirement for explicit calendar attribute.

### Finding 4: cell_measures Still Optional

**CF-1.13 unchanged from CF-1.11**:
> "In this case supplying grid-cell areas via the `cell_measures` attribute is **unnecessary**"

**Conclusion**: Checker requirement contradicts CF-1.13 (no change from CF-1.11 analysis).

---

## Recommendations Updated for CF-1.13

### For CMIP7 Compliance Checker

1. **Fix time squareness bug immediately**
   - CF-1.13 recommends gridpoint within bounds (satisfied by 45.5 ∈ [31, 60])
   - CF-1.13 does NOT require arithmetic midpoint calculation
   - Current checker math is wrong: (31+60)/2 = 45.5, not 45.0
   - **Action**: Remove this check OR fix the calculation

2. **Align with CF-1.13 gridpoint guidance**
   - Check: Is gridpoint within bounds? (YES/NO)
   - Don't check: Does gridpoint equal specific calculation?
   - CF-1.13 allows any location within bounds

3. **Document CMIP7 vs CF-1.13 differences**
   - calendar: CF-1.13 RECOMMENDS, CMIP7 REQUIRES (justified)
   - missing_value precision: CF-1.13 silent, CMIP7 specific (justified)
   - cell_measures: CF-1.13 OPTIONAL, checker REQUIRES (conflict)

4. **Fix bugs unchanged from CF-1.11**
   - parent_experiment_id logic (CV contradiction)
   - long_name validation (not in CF)

### For CMOR4 Development

**No changes needed** - CMOR4 correctly implements CF-1.13:

1. ✅ Calendar attribute on time coordinates (implemented)
2. ✅ Gridpoints lie within bounds (45.5 ∈ [31, 60])
3. ✅ missing_value as float32 (implemented)
4. ✅ cell_measures handling (correctly optional)
5. ✅ CV-based attribute logic (matches CMOR3)

### For CMIP7 Documentation

1. **Create "CMIP7 Profile of CF-1.13"**
   - Document where CMIP7 elevates RECOMMENDED to REQUIRED
   - Justify each elevation (reproducibility, archives, automation)
   - Reference CF-1.13 sections

2. **Fix cell_measures ambiguity**
   - Option A: Populate tables with actual values (align with checker)
   - Option B: Make it optional in checker (align with CF-1.13)
   - Current state (empty in tables, required by checker) is inconsistent

---

## Comparison: CF-1.11 vs CF-1.13 Impact

| Aspect | CF-1.11 | CF-1.13 | Impact on Analysis |
|--------|---------|---------|-------------------|
| **Calendar** | Recommended | Recommended (stronger) | ⬆️ More support for CMIP7 requirement |
| **Gridpoint location** | Not specified | Within/upon bounds | ➕ New guidance, CMOR4 compliant |
| **Midpoint calculation** | Not specified | Not specified | — Checker bug unchanged |
| **cell_measures** | Optional | Optional | — Checker conflict unchanged |
| **Time structure** | Single section | 5 subsections | ✓ Clearer, no req change |
| **missing_value** | No precision | No precision | — CMIP7 strictness justified |

### Overall Assessment

**CF-1.13 provides better guidance but doesn't change the fundamental compliance analysis**:

1. **2 issues where checker is reasonably stricter**:
   - calendar attribute (CF-1.13 strengthens recommendation)
   - missing_value precision (CMIP7 needs reproducibility)

2. **4 issues where checker has bugs**:
   - Time squareness math error (45.5 is correct midpoint)
   - parent_experiment_id logic (contradicts CV)
   - long_name validation (not in CF)
   - sub_experiment_id requirement (not for all experiments)

3. **1 issue with ambiguity**:
   - cell_measures (CF-1.13 says optional, checker requires, tables empty)

**CMOR4 correctly implements CF-1.13**. The compliance checker has the same bugs as identified in CF-1.11 analysis, plus the new CF-1.13 gridpoint-in-bounds recommendation actually confirms CMOR4's test data is correct.

---

## Conclusion

**CF-1.13 strengthens some recommendations (calendar) and adds new guidance (gridpoint locations) but maintains the same fundamental requirements as CF-1.11.**

The compliance checker issues identified in the CF-1.11 analysis remain valid for CF-1.13:

### CMOR4 Status: ✅ Fully CF-1.13 Compliant

1. ✅ Calendar attribute correctly added to time coordinates
2. ✅ Time values lie within bounds (new CF-1.13 recommendation)
3. ✅ missing_value precision correctly implemented
4. ✅ cell_measures correctly treated as optional
5. ✅ CV-based logic matches CMOR3 and CMIP7 specification

### Compliance Checker Status: ❌ Has Bugs

1. ❌ Time squareness: wrong math (expects 45.0 instead of 45.5)
2. ❌ Time squareness: enforces calculation not in CF-1.13
3. ❌ parent_experiment_id: contradicts CMIP7 CV
4. ❌ long_name: validates attribute CF-1.13 doesn't control
5. ⚠️ cell_measures: requires what CF-1.13 makes optional

**The path forward remains**: Fix the compliance checker bugs while maintaining CMOR4's correct implementation of CF-1.13 conventions.

---

**References**:
- CF Conventions 1.13: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html
- CF Conventions 1.11: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html  
- CMIP7 Compliance Report: `CMIP7_COMPLIANCE_REPORT.md`
- CF-1.11 Analysis: `CF_vs_COMPLIANCE_ANALYSIS.md`
- CMOR3 Source: https://github.com/PCMDI/cmor

**Analysis Date**: 2026-06-26
