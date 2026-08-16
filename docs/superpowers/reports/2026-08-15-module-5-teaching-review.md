# Module 5 Teaching Review: Monitoring

## What You Built

The app now converts each completed interaction into a small operational measurement rather than storing the interaction itself. The dashboard reads those measurements and shows whether the system is answering, abstaining, failing configuration, or receiving helpful/needs-work feedback.

## Five Ideas To Know

1. **Metric versus event log:** An event log would keep one record per question and could expose private content. An hourly aggregate keeps only totals and latency summaries for a category, which is enough to monitor system behavior without retaining the conversation.
2. **Outcome versus feedback:** An outcome is what the system did, such as `rag_answered` or `scope_abstained`. Feedback is what a user thought of the completed response: `helpful` or `needs_work`. They answer different questions and are stored separately.
3. **Why Grafana needs a datasource:** Grafana does not contain the monitoring data. Its datasource definition tells it which limited database role may read the aggregate tables, and the dashboard definition tells it which charts to run. Both files are in Git so a reviewer can reproduce the visual result.
4. **Why synthetic telemetry matters:** A new reviewer has no real app traffic. The explicit seed provides deterministic aggregate data so every chart can be checked immediately. It is a demonstration of the monitoring path, not evidence of production performance.
5. **Why raw tables remain unused:** The older raw-content scaffold can retain fields that do not belong in this monitoring design. The runtime writes only to the dedicated hourly aggregate tables, and Grafana has no permission to read anything else.

## Concrete Example

The verified demo inserts 6 synthetic request events across the closed outcomes and 2 feedback events. Grafana groups those into the six panels. For example, the request-volume panel answers, “How many requests ended in each outcome this hour?” The helpful-rate panel answers, “Among submitted feedback this hour, what fraction was helpful?” Neither question requires the original prompt or answer.

## Practical Check

Run the documented monitoring stack and explicit demo seed, then open the dashboard. Confirm the synthetic UTC window shows six populated panels. Change the time picker to a current UTC period only when reviewing real local aggregate activity. Do not put source material, chat messages, or credentials into the dashboard.

## Scope Boundary

This module proves a local privacy-safe monitoring workflow. It does not prove chemistry correctness, treatment safety, production load handling, alerting, hosted operations, or deployment readiness.

## Lock Record

The live dashboard walkthrough and teaching review were completed with the user on 2026-08-16. The user explicitly approved the Module 5 lock on 2026-08-16.
