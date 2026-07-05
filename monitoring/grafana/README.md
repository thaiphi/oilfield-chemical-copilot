# Grafana Monitoring Plan

This folder is reserved for Grafana provisioning and dashboard JSON.

Planned dashboard panels:

- Conversation volume by hour and data mode.
- End-to-end answer latency and retrieval latency.
- Retrieved chunk count, retrieval method mix, and average retrieval score.
- Feedback counts and helpful-rate trend.
- Tool-call counts, latency, and error rate by tool.
- Ingestion run status from Kestra logs or future ingestion tables.

TODO: Add a PostgreSQL datasource provisioning file and dashboard JSON after the logging tables receive real data.
