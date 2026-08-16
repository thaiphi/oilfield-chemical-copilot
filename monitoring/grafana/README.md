# Module 5 Grafana Monitoring

This directory contains Git-tracked provisioning for the local, privacy-safe monitoring dashboard:

- `datasources/postgres.yml` connects Grafana through `grafana_reader`.
- `dashboards/module5-monitoring.json` defines the six-panel synthetic dashboard.
- `../bootstrap_grafana_role.py` grants the reader role `SELECT` only on the two hourly aggregate monitoring tables.
- `../seed_demo_metrics.py` inserts a fixed synthetic aggregate demo only when explicitly run.

The datasource and panels use only `monitoring_request_hourly` and `monitoring_feedback_hourly`. They never query raw conversation, source, retrieval, tool, or error data. Docker Compose binds Grafana to localhost and enables anonymous Viewer access so a reviewer can open the local dashboard without receiving credentials.
