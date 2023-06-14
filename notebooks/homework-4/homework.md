## Homework

In this homework, we'll deploy the ride duration model in batch mode. Like in homework 1, we'll use the Yellow Taxi Trip Records dataset. 

You'll find the starter code in the [homework](homework) directory.


## Q1. Notebook

We'll start with the same notebook we ended up with in homework 1.
We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](homework/starter.ipynb).

Run this notebook for the February 2022 data.

What's the standard deviation of the predicted duration for this dataset?

* 5.28
* 10.28
* 15.28
* 20.28

### Answer

```python
import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(y_pred)
# [18.52778307 23.06578208 33.68635854 ... 11.89045938 15.10268128
#  9.46059157]

print(y_pred.std())
# 5.28140357655334
```

## Q2. Preparing the output

Like in the course videos, we want to prepare the dataframe with the output. 

First, let's create an artificial `ride_id` column:

```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
```

Next, write the ride id and the predictions to a dataframe with results. 

Save it as parquet:

```python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

* 28M
* 38M
* 48M
* 58M

__Note:__ Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
dtypes of the columns and use pyarrow, not fastparquet. 

### Answer 

```python
year = 2022
month = 2
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = "output.parquet"
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

```python
import os

file_stats = os.stat(output_file)

print(f"File size is {file_stats.st_size} bytes, "
      f"{file_stats.st_size / 1024:.1f} kB, "
      f"{file_stats.st_size / (1024 * 1024):.3f} MB.")
# File size is 59994935 bytes, 58588.8 kB, 57.216 MB.
```

## Q3. Creating the scoring script

Now let's turn the notebook into a script. 

Which command you need to execute for that?


### Answer

```bash
jupyter nbconvert --to script starter.ipynb
```


## Q4. Virtual environment

Now let's put everything into a virtual environment. We'll use pipenv for that.

Install all the required libraries. Pay attention to the Scikit-Learn version:
check the starter notebook for details. 

After installing the libraries, pipenv creates two files: `Pipfile`
and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
dependencies we use for the virtual env.

What's the first hash for the Scikit-Learn dependency?

### Answer

```bash
pip install pipenv
pipenv --version
# pipenv, version 2023.6.12
pipenv --python=3.9 
pipenv install scikit-learn==1.2.2 pandas 
```

We should see inside `Pipfile.lock` this.

```json
"scikit-learn": {
    "hashes": [
        "sha256:065e9673e24e0dc5113e2dd2b4ca30c9d8aa2fa90f4c0597241c93b63130d233",
        "sha256:2dd3ffd3950e3d6c0c0ef9033a9b9b32d910c61bd06cb8206303fb4514b88a49",
        "sha256:2e2642baa0ad1e8f8188917423dd73994bf25429f8893ddbe115be3ca3183584",
        "sha256:44b47a305190c28dd8dd73fc9445f802b6ea716669cfc22ab1eb97b335d238b1",

```


## Q5. Parametrize the script

Let's now make the script configurable via CLI. We'll create two 
parameters: year and month.

Run the script for March 2022. 

What's the mean predicted duration? 

* 7.76
* 12.76
* 17.76
* 22.76

Hint: just add a print statement to your script.

### Answer 

Below is the `starter.py` script.

```python
#!/usr/bin/env python
# coding: utf-8
import sys
import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def ride_duration_prediction(year: int, month: int) -> float:
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(filename)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def run():
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # 3

    y_pred = ride_duration_prediction(
        year=year,
        month=month
    )

    # Mean predicted duration.
    print(f"Mean predicted duration = {y_pred.mean()}")

if __name__ == '__main__':
    run()
```

I ran the following commands in the terminal.

```bash
pipenv shell
pipenv install pyarrow fastparquet
python starter.py 2022 3
# Mean predicted duration = 12.758556818790902
```


## Q6. Docker container 

Finally, we'll package the script in the docker container. 
For that, you'll need to use a base image that we prepared. 

This is how it looks like:

```
FROM python:3.10.0-slim

WORKDIR /app
COPY [ "model2.bin", "model.bin" ]
```

We pushed it to [`svizor/zoomcamp-model:mlops-3.10.0-slim`](https://hub.docker.com/layers/svizor/zoomcamp-model/mlops-3.10.0-slim/images/sha256-595bf690875f5b9075550b61c609be10f05e6915609ef4ea4ce9797116c99eff?context=repo),
which you should use as your base image.

That is, this is how your Dockerfile should start:

```docker
FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# do stuff here
```

This image already has a pickle file with a dictionary vectorizer
and a model. You will need to use them.

Important: don't copy the model to the docker image. You will need
to use the pickle file already in the image. 

Now run the script with docker. What's the mean predicted duration
for April 2022? 


* 7.92
* 12.83
* 17.92
* 22.83

### Answer

The `Dockerfile` file.

```txt
FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# set a directory for the app
WORKDIR /app

# copy these files to the container
COPY [ "Pipfile", "Pipfile.lock", "starter.py", "./" ]

# install dependencies
RUN pip install -U pip
RUN pip install pipenv 
RUN pipenv install --system --deploy

# run the script
ENTRYPOINT ["python", "starter.py"]
``` 

I ran the following commands in the terminal.

```bash
docker build --platform linux/amd64 -t starter_py .
docker run --platform linux/amd64 -it starter_py 2022 4
```


## Bonus: upload the result to the cloud (Not graded)

Just printing the mean duration inside the docker image 
doesn't seem very practical. Typically, after creating the output 
file, we upload it to the cloud storage.

Modify your code to upload the parquet file to S3/GCS/etc.

## Answer

We should do something like this:

```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['mean_duration'] = y_pred.mean()

output_file = f's3://nyc-duration-prediction/year={year:04d}/month={month:02d}/{run_id}.parquet'
df_result.to_parquet(output_file, index=False)
```

## Publishing the image to dockerhub

This is how we published the image to Docker hub:

```bash
docker build -t mlops-zoomcamp-model:v1 .
docker tag mlops-zoomcamp-model:v1 svizor/zoomcamp-model:mlops-3.10.0-slim
docker push svizor/zoomcamp-model:mlops-3.10.0-slim
```


## Submit the results

* Submit your results here: https://forms.gle/4tnqB5yGeMrTtKKa6
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
* You can submit your answers multiple times. In this case, the last submission will be used for scoring.


## Deadline

The deadline for submitting is 19 June 2023 (Monday) 23:00 CEST. 
After that, the form will be closed.