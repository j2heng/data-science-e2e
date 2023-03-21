FROM postgres:latest

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install psycopg2-binary

EXPOSE 5432

ENV POSTGRES_USER recsys_app
ENV POSTGRES_PASSWORD recsys_app
ENV POSTGRES_DB mlflow

CMD ["postgres"]