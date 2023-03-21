#!/usr/bin/env bash

CMD=$1
shift

case $CMD in
    mlflow-build)
        docker build -f mlflow_app/Dockerfile -t recsys-mlflow-app:local .
    ;;
    mlflow-up)
        docker run --rm --name recsys-mlflow-app -p 5000:5000 recsys-mlflow-app:local
    ;;  
    postgresql-build)
        docker build -f postgresql_app/Dockerfile -t recsys-postgresql-app:local .
    ;;
    postgresql-up)
        docker run --rm --name recsys-postgresql-app -p 5432:5432 recsys-postgresql-app:local
    ;;   
    --help|-h|*)
        echo "$0 [command] ...

COMMANDS:
    mlflow-build: build mlflow image
    postgresql-build: build postgresql image

"
esac
