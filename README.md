# End-to-End Data Science Project with Kedro and MLflow

This project showcases the power of Kedro and MLflow for organizing effective data pipelines that are production-ready from the get-go. By leveraging Kedro, I was able to restructure the feature engineering and model training processes into pipelines that were testable and could run in parallel. 

I also incorporated MLflow into our project for tracking and experimentation. With MLflow, I was able to keep track of our experiments and iterations, making it easier to reproduce and build upon the results. This also allowed me to streamline the model selection process and more easily identify the best-performing models for deployment.

![alt text](https://github.com/j2heng/data-science-e2e/blob/main/Architecture%20Diagram.png?raw=true)
Architecture Diagram


## Installation
To install the necessary dependencies for this project, run:
```python
pip install -r requirements.txt
```

## Usage
To run the Kedro pipeline, execute the following command:
```python
kedro run --pipeline train_cf_model
```

This will start the model training process. However, the model tracking requires MLflow and PostgreSQL to be up and running. 
### Install Docker engine
See official docker documentation: https://docs.docker.com/get-docker/

### Setup postgresql server

A bash file `run-docker.sh` is created to help with the installation of mlflow and postgresql servers. 
```
./run-docker.sh postgresql-build
./run-docker.sh postgresql-up
```
The postgresql server will be automatically forwarded to your localhost port 5432. 

In case of `Permission denied`, run the following line to change the permission. 
```
chmod 755 run-docker.sh
```

### Start MLflow Server
```
./run-docker.sh mlflow-build
./run-docker.sh mlflow-up
```
You can then navigate to localhost:5000 in your web browser to view the MLflow tracking dashboard.





