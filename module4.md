# 4. Model Deployment

See [GihHub page](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/04-deployment) 
for this section.

## 4.1 Three ways of deploying a model

:movie_camera: [Youtube](https://www.youtube.com/watch?v=JMGe4yIoBRA&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=22).

### MLOps: Design, Train, Experiment, Deploy

TODO: Mettre ici des impressions d'écrans des diagrammes.

## 4.2 Web-services: Deploying models with Flask and Docker

:movie_camera: [Youtube](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23).

### Introduction

> [00:00](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=0s) Introduction

In previous section, we created a pickle file with these instructions.

```python
with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)
```

Now we can take this pickel file and deploy this as a web application.

Gregory recommends viewing the videos linked to a ML Zoomcamp module (https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment) before starting this section.
We will do something similar here in this section.



### Pipenv environment

> [01:41](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=101s) Pipenv environment

We will create a virtual environment with [pipenv](https://pipenv.pypa.io/en/latest/), and then 
we will put our model in a script, put this script into [Flask](https://flask.palletsprojects.com/en/) application, 
and finnaly we will package everything in a [Docker](https://www.docker.com/).

We already have a model [here](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/04-deployment/web-service).

We need to find out the exact version of scikit-learn that we use for creating this pickle file.
If we try to unpickle with a different version of scikit-learn, it might not work.

The command `pip freeze` shows all the libraries that we currently have installed.

```bash
pip freeze | grep scikit-learn
```

We should see this.

```txt
scikit-learn==1.0.2
scikit-learn-intelex==2021.20210714.170444
```

We need to install with `pipenv`.

```bash
pipenv install scikit-learn==1.0.2 flask --python=3.9
```

This command will create a virtual environment and install the specified packages.
Enter this environment with the following command.

```bash
pipenv shell
```

You could change the prompt for one shorter with this command.

```bash
PS1="> "
```

You must have a `Pipfile` file that contains this.

```txt
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
scikit-learn = "==1.0.2"
flask = "*"

[dev-packages]

[requires]
python_version = "3.9"
```

The `Pipfile.lock` file contains more precise information about the specific versions used of each 
of the libraries.

### Flask App

> [05:12](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=312s) Flask App

Below, the [`predict.py`](https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/04-deployment/web-service/predict.py) script.

```python
import pickle

from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

Now let's test that everything works.
See [`test.py`](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/04-deployment/web-service/test.py)

Test without Web request.

```python
import predict

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

features = predict.prepare_features(ride)
pred = predict.predict(features)
print(pred[0])
```

Run the following and you should get the prediction.

```bash
python test.py
# 26.43883355119793
```

Test with Web request has a Flask application.

```python
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
```

Run the following and you should get the prediction.

```bash
python test.py
# {'duration': 26.43883355119793}
```

### gunicorn

> [16:23](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=983s) gunicorn

We actually put our model in a Flask application and we can interact with it.
We can send ride information and get the prediction for this ride.

When we start our Flask application, we see this warning.

```txt
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
```

To fix this warning, we need to use a production server like [gunicorn](https://gunicorn.org/).

```bash
# Install.
pipenv install gunicorn

# Start the application
gunicorn --bind:0.0.0.0:9696 predict:app
```

Open another terminal window and run the same test.

```bash
python test.py
# {'duration': 26.43883355119793}
```

### Pipenv --dev

> [17:52](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=1072s) Pipenv --dev

The following commands do not work.

```bash
pipenv shell
python test.py
# ModuleNotFoundError: No module names 'requests'
```

We can install the requests library as a development dependency.
So only when we develop we will have this dependency but when we deploy this to production
this dependency will not be installed.

We can do this by using this command. Now, if we run the test, it will work.

```bash
pipenv install --dev requests
python test.py
# {'duration': 26.43883355119793}
```

### Docker

> [19:20](https://www.youtube.com/watch?v=D7wfMAdgdF8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=23&t=1160s) Docker

Just one last thing is packaging the app to a docker container.

Create a [`Dockerfile`](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/04-deployment/web-service/Dockerfile) with this content.

```txt
FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
```

To build an image from a Dockerfile, run this.

```bash
docker build -t ride-duration-prediction-service:v1 .
```

To create and run a new container from the image, just run this.

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

See [docker build](https://docs.docker.com/engine/reference/commandline/build/)
and [docker run](https://docs.docker.com/engine/reference/commandline/run/) for more 
information.


## 4.3 Web-services: Getting the models from the model registry (MLflow)

:movie_camera: [Youtube](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24).

### Introduction

> [00:00](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=0s) Introduction

### Goal and required set-up

> [00:51](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=51s) Goal and required set-up

In this section, we will talk about combining what we did in the previous lesson 
deploying a model with a web service through Flask with our model registry.



### Writing flask application with web-service mlflow

> [03:13](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=193s) Writing flask application with web-service mlflow

### Testing and improving application

> [07:03](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=423s) Testing and improving application

### Creating scikit-learn pipeline and logging it

> [10:10](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=610s) Creating scikit-learn pipeline and logging it

### Why & how to become independent from the tracking server

> [14:44](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=884s) Why & how to become independent from the tracking server

### Recap

> [18:27](https://www.youtube.com/watch?v=aewOpHSCkqI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=24&t=1107s) Recap

## 4.4 (Optional) Streaming: Deploying models with Kinesis and Lambda

:movie_camera: [Youtube](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25).

### Intro to Kinesis and Lambda


> [00:00](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=0s) Intro to Kinesis and Lambda

### Create Lambda

> [03:05](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=185s) Create Lambda

### Create a roll

> [04:32](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=272s) Create a roll

### Back to creating a Lambda function

> [06:46](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=406s) Back to creating a Lambda function

### Write Lambda function

> [07:30](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=450s) Write Lambda function

### Create Kinesis data stream

> [11:47](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=707s) Create Kinesis data stream

### Connect to Kinesis

> [14:03](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=843s) Connect to Kinesis

### Send a test event to the stream

> [15:33](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=933s) Send a test event to the stream

### Send an event about the ride

> [20:32](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=1232s) Send an event about the ride

### Create new kinesis stream for the predictions

> [24:52](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=1492s) Create new kinesis stream for the predictions

### Modify role to add write permission with new policy

> [30:30](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=1830s) Modify role to add write permission with new policy

### Read from the new stream and check that function works

> [33:38](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=2018s) Read from the new stream and check that function works

### Ad the model to the Lambda function

> [36:43](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=2203s) Ad the model to the Lambda function

### Package into a virtual Environment and creates docker

> [42:10](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=2530s) Package into a virtual Environment and creates docker

### Send test request

> [46:04](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=2764s) Send test request

### Publish it to ECR

> [53:06](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=3186s) Publish it to ECR

### Create new Lambda from Container Image

> [55:12](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=3312s) Create new Lambda from Container Image

### Recap

> [1:05:34](https://www.youtube.com/watch?v=TCqr9HNcrsI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=25&t=3934s) Recap

## 4.5 Batch: Preparing a scoring script

:movie_camera: [Youtube](https://www.youtube.com/watch?v=18Lbaaeigek&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=26).

### Introduction and goal

> [00:00](https://www.youtube.com/watch?v=18Lbaaeigek&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=26&t=0s) Introduction and goal

### Turn previous training notebook into applying model notebook

> [02:00](https://www.youtube.com/watch?v=18Lbaaeigek&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=26&t=120s) Turn previous training notebook into applying model notebook

### Turn notebook into a script

> [18:34](https://www.youtube.com/watch?v=18Lbaaeigek&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=26&t=1114s) Turn notebook into a script

### Tips for further improvements: Creating an environment and other options

> [24:16](https://www.youtube.com/watch?v=18Lbaaeigek&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=26&t=1456s) Tips for further improvements: Creating an environment and other options

## 4.6 MLOps Zoomcamp 4.6 - Batch: Scheduling batch scoring jobs with Prefect

:movie_camera: [Youtube](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27).

### Intro

> [00:00](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=0s) Intro

### Prepare environment and modify script

> [00:35](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=35s) Prepare environment and modify script

### Change output file destination to S3

> [09:21](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=561s) Change output file destination to S3

### Turn script into proper Prefect flow

> [12:47](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=767s) Turn script into proper Prefect flow

### Deploy Prefect flow

> [16:39](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=999s) Deploy Prefect flow

### Apply model to previous months: Backfilling

> [23:24](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=1404s) Apply model to previous months: Backfilling

### Recap

> [28:47](https://www.youtube.com/watch?v=ekT_JW213Tc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=27&t=1727s) Recap


**Note**: There are several changes to deployment in Prefect 2.3.1 since 2.0b8:

* `DeploymentSpec` in 2.0b8 now becomes `Deployment`.
* `work_queue_name` is used instead of `tags` to submit the deployment to the a specific work queue.
* You don't need to create a work queue before using the work queue. A work queue will be created if it doesn't exist.
* `flow_location` is replaced with `flow`.
* `flow_runner` and `flow_storage` are no longer supported.

```python
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from score import ride_duration_prediction

deployment = Deployment.build_from_flow(
    flow=ride_duration_prediction,
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "e1efc53e9bd149078b0c12aeaa6365df",
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
    work_queue_name="ml",
)

deployment.apply()
```

## 4.7 Choosing the right way of deployment

:movie_camera: [Youtube]().

COMING SOON

## 4.8 Homework

