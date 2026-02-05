

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1


ENV PYTHONUNBUFFERED=1

WORKDIR /app


ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

USER appuser


COPY app.py gradcam_visual.py loading_model.py final_pneumonia_model.keras ./

EXPOSE 8000


CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

