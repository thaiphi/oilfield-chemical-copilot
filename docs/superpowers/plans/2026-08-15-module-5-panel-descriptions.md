# Module 5 Panel Descriptions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Module 5 Grafana panel a built-in information description that explains its purpose and displayed parameters.

**Architecture:** Grafana renders the panel-header information control from a panel's `description` property. Add static Markdown descriptions to the six tracked dashboard panels without altering queries, datasource access, stored telemetry, panel layout, or the synthetic seed. Extend the provisioning test to make the descriptions durable.

**Tech Stack:** Grafana dashboard JSON, Python 3.11, pytest, Ruff.

## Global Constraints

- Preserve all six existing panel IDs, titles, positions, queries, and visualization types.
- Explain only closed aggregate metrics; do not mention or expose prompt, answer, citation, source, identifier, or raw-error data.
- Use plain language and define each metric's unit, denominator, and UTC hourly basis where relevant.
- Do not change dashboard data, migration, seed, application, or monitoring persistence behavior.
- Do not lock Module 5, commit, or push.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `monitoring/grafana/dashboards/module5-monitoring.json` | Grafana panel descriptions displayed through the built-in information control. |
| `tests/monitoring/test_grafana_provisioning.py` | Regression contract for complete, parameter-aware panel descriptions. |

### Task 1: Test The Panel Information Contract

**Files:**
- Modify: `tests/monitoring/test_grafana_provisioning.py`

**Interfaces:**
- Consumes: Parsed JSON dashboard object with a `panels` list.
- Produces: A regression test requiring a non-empty `description` for every panel and metric terms specific to its panel ID.

- [x] **Step 1: Write the failing test**

```python
def test_every_panel_has_a_parameter_aware_description() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    required_terms = {
        1: ("request_count", "outcome", "UTC"),
        2: ("latency_total_ms", "latency_count", "milliseconds"),
        3: ("latency_minimum_ms", "latency_maximum_ms", "milliseconds"),
        4: ("outcome", "request_count", "percentage"),
        5: ("retrieval_mode", "request_count", "not_applicable"),
        6: ("helpful_rate", "feedback_count", "needs_work"),
    }
    for panel in dashboard["panels"]:
        description = panel.get("description", "")
        assert description
        assert all(term in description for term in required_terms[panel["id"]])
```

- [x] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py::test_every_panel_has_a_parameter_aware_description -q`
Expected: FAIL because panels have no `description` property.

### Task 2: Add Grafana Information Descriptions

**Files:**
- Modify: `monitoring/grafana/dashboards/module5-monitoring.json`

**Interfaces:**
- Consumes: Grafana `description` string on each panel.
- Produces: Built-in panel information controls with purpose, parameters, units, and scope.

- [x] **Step 1: Add the six descriptions**

Add a `description` property to each panel. Keep query and layout fields unchanged. Use exact parameter names required by Task 1 and explain the derived calculations in the description text.

- [x] **Step 2: Run focused monitoring tests**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py -q`
Expected: PASS.

- [x] **Step 3: Run lint and whitespace verification**

Run: `uv run ruff check tests/monitoring; git diff --check`
Expected: both commands succeed.

## Plan Self-Review

- Spec coverage: Task 1 defines durable descriptions; Task 2 adds all six information controls without changing metrics.
- Placeholder scan: every described test term and verification command is concrete.
- Scope: the plan changes only dashboard documentation metadata and its test.
