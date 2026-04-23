FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY explainer.py /app/explainer.py
COPY detector.py /app/detector.py
COPY runtime/ /app/runtime/
COPY model_repo_seed/ /app/model_repo_seed/
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app:/app/runtime
ENV MLSERVER_PARALLEL_WORKERS=0

EXPOSE 8080 8081

CMD ["/app/entrypoint.sh"]
