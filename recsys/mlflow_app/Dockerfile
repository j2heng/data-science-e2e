FROM python:3.8.0

RUN pip install \
    mlflow==1.23.1 \
    psycopg2==2.9.1 \
    protobuf==3.20.1 \
    numpy==1.23.4 \
    SQLAlchemy==1.4.46 \
    && \
    mkdir /mlflow/

EXPOSE 5000

ENV USERNAME "recsys_app"
ENV PASSWORD "recsys_app"
ENV HOST "host.docker.internal"
ENV PORT "5432"
ENV DATABASE "mlflow"

CMD mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./artifacts --backend-store-uri postgresql://${USERNAME}:${PASSWORD}@${HOST}:${PORT}/${DATABASE}