# Benchmark Regression Workflow Testing Guide

## ✅ Local Validation Complete

The benchmark comparator has been tested successfully with three severity scenarios:

### Scenario Results

| Scenario | Severity | Triage | Status | Key Output |
|----------|----------|--------|--------|-----------|
| **BLOCKER** (HIGH) | high | BLOCKER | ❌ FAILED | `BENCHMARK_SEVERITY_RANK=3` |
| **REGRESSION** (MEDIUM) | medium | REGRESSION | ❌ FAILED | `BENCHMARK_SEVERITY_RANK=2` |
| **WATCH** (LOW) | low | WATCH | ✅ PASSED | `BENCHMARK_SEVERITY_RANK=1` |

All three scenarios generated correct:
- ✅ Human-readable delta tables with dP95(%) and dMem(%) columns
- ✅ Machine-readable status lines (BENCHMARK_PASSED, BENCHMARK_SEVERITY, etc.)
- ✅ JSON comparison output with severity classification
- ✅ Correct severity ranking for downstream job branching

## 🚀 Testing in GitHub Actions

### Option 1: Manual Workflow Dispatch (Easiest)

1. **Push test data to your repository:**
   ```bash
   git add artifacts/benchmark_history.json
   git commit -m "test: add sample benchmark history for workflow validation"
   git push origin main
   ```

2. **Trigger the workflow manually:**
   - Go to your GitHub repo → **Actions** tab
   - Select **Benchmark Regression Gate** workflow
   - Click **Run workflow** button
   - Use defaults (tests with 5% p95, 10% memory thresholds)
   - Click **Run workflow**

3. **What to expect:**
   - The workflow will run in ~30 seconds
   - Scroll to **compare** job to see the comparator output
   - Look for status lines like:
     ```
     BENCHMARK_PASSED=false
     BENCHMARK_SEVERITY=high
     BENCHMARK_TRIAGE=BLOCKER
     BENCHMARK_SEVERITY_RANK=3
     ```
   - Check job **Outputs** tab to verify 6 values are parsed:
     - `benchmark_passed`
     - `benchmark_severity`
     - `benchmark_triage`
     - `benchmark_severity_rank`
     - `benchmark_p95_exceeded`
     - `benchmark_peak_mem_exceeded`
   - Two downstream jobs will run:
     - **notify-regression** (if severity_rank == 2)
     - **blocker-escalation** (if severity_rank >= 3)

### Option 2: Test via Pull Request (More Realistic)

1. **Commit the sample benchmark history file:**
   ```bash
   git add artifacts/benchmark_history.json
   git commit -m "test: benchmark regression workflow"
   git push origin feature/workflow-test
   ```

2. **Create a PR on GitHub** from `feature/workflow-test` to `main`

3. **The workflow will auto-trigger** because the PR modifies `artifacts/benchmark_history.json`

4. **Verify the PR comment:**
   - Scroll down to Comments section
   - Look for the sticky benchmark summary with format:
     ```markdown
     ## [BLOCKER] Benchmark Regression Summary
     
     - Passed: **false**
     - Severity: **HIGH**
     - Triage: **BLOCKER**
     - Worst p95 regression: **17.68%**
     - Worst peak memory regression: **13.30%**
     
     | Variant | dP95 (ms) | dP95 (%) | dPeakMem (MB) | dPeakMem (%) |
     |---|---:|---:|---:|---:|
     | whisper-base | 4.5 | 3.9 | 8.5 | 1.66 |
     | whisper-small | 12.5 | 7.25 | 32.3 | 4.21 |
     | whisper-medium | 51.1 | 17.68 | 170.2 | 13.30 |
     ```
   - Comment should have hidden marker: `<!-- benchmark-regression-comment -->`
   - If you update the PR (e.g., fix the regression), the comment will auto-update

5. **Verify artifacts:**
   - Go to workflow run
   - Download `benchmark-comparison` artifact
   - Should contain:
     - `benchmark_comparison.json` (full delta details)
     - `benchmark_comparison.md` (markdown table)
     - `benchmark_compare_stdout.txt` (raw comparator output)

### Option 3: Test with Different Thresholds

Manually trigger the workflow with custom parameters:

1. Go to **Actions** → **Benchmark Regression Gate**
2. Click **Run workflow**
3. Customize inputs:
   - **baseline_index**: `-2` (default) or specify another index
   - **candidate_index**: `-1` (default) or specify another index
   - **max_p95_regression_pct**: `5` (default) or try stricter like `3`
   - **max_peak_mem_regression_pct**: `10` (default) or try stricter like `8`
   - **fail_on_regression**: `true` to enforce the gate, `false` to report only
4. Run and observe how severity changes based on thresholds

## ✅ Validation Checklist

After running the workflow, verify:

- [ ] **Comparator runs successfully** (compare job completes without errors)
- [ ] **Status lines are emitted** (look in "Compare benchmark runs" step output):
  - [ ] `BENCHMARK_PASSED=...`
  - [ ] `BENCHMARK_SEVERITY=...`
  - [ ] `BENCHMARK_TRIAGE=...`
  - [ ] `BENCHMARK_SEVERITY_RANK=...`
  - [ ] `BENCHMARK_P95_EXCEEDED=...`
  - [ ] `BENCHMARK_PEAK_MEM_EXCEEDED=...`
- [ ] **Job outputs are populated** (check "Outputs" tab of compare job)
- [ ] **Markdown summary is generated** (check "Build markdown regression summary" step)
- [ ] **PR comment is posted** (if triggered via PR) with `[TRIAGE]` prefix in header
- [ ] **Downstream jobs trigger correctly:**
  - [ ] notify-regression runs **only if** severity_rank == 2
  - [ ] blocker-escalation runs **only if** severity_rank >= 3
- [ ] **Artifacts are preserved** (benchmark-comparison artifact contains JSON, MD, stdout)

## 🔧 If Something Fails

### Workflow Doesn't Trigger on PR
- **Cause**: PR doesn't modify `artifacts/benchmark_history.json`
- **Fix**: Ensure your test branch commits the file: `git add artifacts/benchmark_history.json && git commit -m "test"`

### Status Lines Not Parsed
- **Cause**: Comparator output format changed or grep pattern is wrong
- **Check**: Look at "Compare benchmark runs" step → scroll all the way down for status lines
- **Fix**: Verify the output matches the pattern (exact `KEY=value` format with no extra whitespace)

### PR Comment Not Posted
- **Cause**: Workflow run not triggered by PR, OR permissions issue
- **Check**: Go to workflow run → "Comment benchmark summary on PR" step → view logs
- **Fix**: Manually trigger via `workflow_dispatch` to test comparator outside of PR flow

### Downstream Jobs Don't Run
- **Cause**: severity_rank output is empty or condition syntax is wrong
- **Check**: Verify compare job outputs are populated (3 outputs available)
- **Fix**: Check the "Outputs" tab in the compare job; if empty, see "Status Lines Not Parsed"

## 📋 Test Data Inventory

Three test files have been created in `artifacts/`:

1. **benchmark_history.json** (BLOCKER scenario)
   - Used by default manual dispatch
   - Contains 2 runs with high regressions
   - Expected output: `BENCHMARK_SEVERITY_RANK=3`

2. **benchmark_history_clean.json** (WATCH scenario)
   - Contains 3 runs with minimal regression
   - Expected output: `BENCHMARK_SEVERITY_RANK=1`, passed=true

3. **benchmark_history_regression.json** (REGRESSION scenario)
   - Contains 2 runs with moderate regression (exceeds p95 threshold)
   - Expected output: `BENCHMARK_SEVERITY_RANK=2`, passed=false

**To test each scenario**, modify the workflow dispatch input:
```yaml
history_json: artifacts/benchmark_history_clean.json  # for WATCH scenario
```

## 🎯 Next Steps

1. **Push test data** to your repo
2. **Run the workflow** via manual dispatch or PR
3. **Verify outputs** match the checklist above
4. **Adjust thresholds** as needed for your performance targets
5. **Remove test files** when satisfied:
   ```bash
   git rm artifacts/benchmark_history*.json artifacts/benchmark_comparison*.json
   git commit -m "test: clean up test benchmark data"
   git push
   ```

---

**Questions?** Check the workflow YAML at [.github/workflows/benchmark-regression.yml](.github/workflows/benchmark-regression.yml) for all step definitions and outputs mapping.
