FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY db ./db
COPY ingestion ./ingestion
COPY monitoring ./monitoring
COPY src ./src
COPY data/sample ./data/sample

RUN uv sync --locked --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
