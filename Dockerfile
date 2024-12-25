FROM python:3.11-slim

ENV PATH="/root/.local/bin:$PATH" \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean

RUN poetry config virtualenvs.in-project true

WORKDIR /app

COPY . /app

RUN poetry install --no-interaction --no-ansi -vvv

ENTRYPOINT ["poetry", "run", "python", "app/inference.py"]
