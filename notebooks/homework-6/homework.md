# 6A.7 Homework

See [Homework](notebooks/homework-6/homework.md).


In this homework, we'll take the ride duration prediction model that we deployed in batch mode in homework 4 and improve the 
reliability of our code with unit and integration tests. 

You'll find the starter code in the [homework]([homework/](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/cohorts/2023/06-best-practices/homework)) directory.

## Q1. Refactoring

Before we can start converting our code with tests, we need to 
refactor it. We'll start by getting rid of all the global variables. 

* Let's create a function `main` with two parameters: `year` and `month`.
* Move all the code (except `read_data`) inside `main`
* Make `categorical` a parameter for `read_data` and pass it inside `main`

Now we need to create the "main" block from which we'll invoke
the main function. How does the `if` statement that we use for
this looks like? 

Hint: after refactoring, check that the code still works. Just run
it e.g. for Feb 2022 and see if it finishes successfully. 

To make it easier to run it, you can write results to your local
filesystem. E.g. here:

```python
output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
```

### Answer

Below, the refactored script.

```python
#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


def read_data(filename: str, categorical: list):
    """Read data"""
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main():
    """Main function"""
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == "__main__":
    main()
```

The result.

```txt
predicted mean duration: 12.513422116701408
```

Below are the bash instructions to run the script.

```bash
$ pip install pipenv
$ pipenv install 
$ pipenv shell
$ which python
/Users/boisalai/.local/share/virtualenvs/homework-6-g3QxDIhy/bin/python
$ mkdir output
$ python batch.py 2022 2
```

So, the answer is this.

```python
if __name__ == "__main__":
    main()
```

## Q2. Installing pytest

Now we need to install `pytest`:

```bash
pipenv install --dev pytest
```

Next, create a folder `tests` and then two files inside. 

The first one will be the file with tests. We can name it `test_batch.py`. 

The second file will be `__init__.py`. So, why do we need this second file?

- ✅ To define a package and specify its boundaries
- To manage the import of modules from the package 
- Both of the above options are correct
- To initialize a new object

### Answer

`__init__.py` is a special Python file that is used to indicate that the directory it is present in is a Python package. 
It can contain initialization code for the package, or it can be an empty file.

See [here](https://www.w3docs.com/snippets/python/what-is-init-py-for.html#:~:text=__init__.py%20is,using%20%22dotted%20module%20names%22.) for more.


## Q3. Writing first unit test

Now let's cover our code with unit tests.

We'll start with the pre-processing logic inside `read_data`.

It's difficult to test right now because first reads
the file and then performs some transformations. We need to split this 
code into two parts: reading (I/O) and transformation. 

So let's create a function `prepare_data` that takes in a dataframe 
(and some other parameters too) and applies some transformation to it.

(That's basically the entire `read_data` function after reading 
the parquet file)

Now create a test and use this as input:

```python
data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)
```

Where `dt` is a helper function:

```python
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)
```

Define the expected output and use the assert to make sure 
that the actual dataframe matches the expected one

Tip: When you compare two Pandas DataFrames, the result is also a DataFrame.
The same is true for Pandas Series. Also, a DataFrame could be turned into a
list of dictionaries.  

How many rows should be there in the expected dataframe?

- 1
- 2
- ✅ 3
- 4

### Answer

See the test script [tests/test_batch.py](tests/test_batch.py).

Below, the test script.

```python
#!/usr/bin/env python
# coding: utf-8

import sys
import os
from datetime import datetime
import logging

import pandas as pd


# Directory reach.
directory = os.path.dirname(os.path.realpath(__file__))

# Setting path.
parent = os.path.dirname(directory)
sys.path.append(parent)

# Importing.
from batch_q3 import prepare_data 


def dt(hour, minute, second=0) -> datetime:
    return datetime(2022, 1, 1, hour, minute, second)

def create_data() -> pd.DataFrame:
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    return df

def create_expected_output() -> pd.DataFrame:
    data = [
        ('-1', '-1', dt(1, 2), dt(1, 10), 8.0),
        ('1', '-1', dt(1, 2), dt(1, 10), 8.0),
        ('1', '2', dt(2, 2), dt(2, 3), 1.0)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    df = pd.DataFrame(data, columns=columns)

    return df

def test_data():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('Begin test_data()')

    actual_input = create_data()
    categorical = ['PULocationID', 'DOLocationID']
    actual_input = prepare_data(actual_input, categorical)

    logger.info("actual_input")
    logger.info(actual_input)
    logger.info(actual_input.info())

    expected_output = create_expected_output()

    logger.info("experted_output")
    logger.info(expected_output)
    logger.info(expected_output.info())

    assert actual_input.to_dict() == expected_output.to_dict()
```

The bash command to run the test and to display the log.

```bash
pytest -rP tests/
```

See [here](https://stackoverflow.com/a/59156707/13969) the correct way to display the log output of passing tests.

The result.

```txt
--------------------------------------------------- Captured log call ----------------------------------------------------
INFO     tests.test_batch:test_batch.py:57 Begin test_data()
INFO     tests.test_batch:test_batch.py:63 actual_input
INFO     tests.test_batch:test_batch.py:64   PULocationID DOLocationID  ... tpep_dropoff_datetime duration
0           -1           -1  ...   2022-01-01 01:10:00      8.0
1            1           -1  ...   2022-01-01 01:10:00      8.0
2            1            2  ...   2022-01-01 02:03:00      1.0

[3 rows x 5 columns]
INFO     tests.test_batch:test_batch.py:65 None
INFO     tests.test_batch:test_batch.py:69 experted_output
INFO     tests.test_batch:test_batch.py:70   PULocationID DOLocationID  ... tpep_dropoff_datetime duration
0           -1           -1  ...   2022-01-01 01:10:00      8.0
1            1           -1  ...   2022-01-01 01:10:00      8.0
2            1            2  ...   2022-01-01 02:03:00      1.0

[3 rows x 5 columns]
INFO     tests.test_batch:test_batch.py:71 None
=================================================== 1 passed in 0.26s ===================================================
```

## Q4. Mocking S3 with Localstack 

Now let's prepare for an integration test. In our script, we  write data to S3. So we'll use Localstack to mimic S3.

First, let's run Localstack with Docker compose. Let's create a  `docker-compose.yaml` file with just one service: localstack. Inside
localstack, we're only interested in running S3. 

Start the service and test it by creating a bucket where we'll keep the output. Let's call it "nyc-duration".

With AWS CLI, this is how we create a bucket:

```bash
aws s3 mb s3://nyc-duration
```

Then we need to check that the bucket was successfully created. With AWS, this is how we typically do it:

```bash
aws s3 ls
```

In both cases we should adjust commands for localstack. Which option do we need to use for such purposes?

- ✅ `--endpoint-url`
- `--profile`
- `--region`
- `--version`

### Answer 

Below, the `docker-compose.yml` file.

```yaml
version: "3.8"

services:
  localstack:
    image: localstack/localstack:latest
    environment:
      - AWS_REGION=ca-central-1
      - EDGE_PORT=4566
      - SERVICES=s3
    ports:
      - '4566-4583:4566-4583'
    volumes:
      - "${TEMPDIR:-/tmp/localstack}:/tmp/localstack"
      - "/var/run/docker.sock:/var/run/docker.sock"
```

Start the docker-compose file using `docker-compose up`.

Install the [AWS CLI](https://aws.amazon.com/cli/). Even though we aren't going to be working with "real" AWS, we'll use this to talk to our local docker containers.

To verify that the shell can find and run the `aws` command in your `$PATH`, use the following commands in a new terminal window.

```bash
$ which aws
/usr/local/bin/aws
$ aws --version
aws-cli/2.13.0 Python/3.11.4 Darwin/22.5.0 exe/x86_64 prompt/off
```

Once the AWS CLI is installed, run aws configure to create some credentials. Even though we're talking to our 
"fake" local service, we still need credentials. 
You can enter real credentials (as described [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)), 
or dummy ones. Localstack requires that these details are present, but doesn't actually validate them. 

```bash
$ aws configure
```

![s01](s01.png)

This command created the `~/.aws/credentials` file.

```bash
$ cat ~/.aws/credentials
[default]
aws_access_key_id = test
aws_secret_access_key = test
```

Create a bucket and list buckets.

```bash
$ aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration
make_bucket: nyc-duration
$ aws --endpoint-url=http://localhost:4566 s3 ls
2023-07-10 11:52:50 nyc-duration
```

### Make input and output paths configurable

Right now the input and output paths are hardcoded, but we want
to change it for the tests. 

One of the possible ways would be to specify `INPUT_FILE_PATTERN` and `OUTPUT_FILE_PATTERN` via the env 
variables. Let's do that:

```bash
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
```

And this is how we can read them:

```python
def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    # rest of the main function ... 
```

### Answer 

See the adjusted script [batch_q4.py](batch_q4.py).

### Reading from Localstack S3 with Pandas

So far we've been reading parquet files from S3 with using
pandas `read_parquet`. But this way we read it from the
actual S3 service. Now we need to replace it with our localstack
one.

For that, we need to specify the endpoint url:

```python
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

df = pd.read_parquet('s3://bucket/file.parquet', storage_options=options)
```

Let's modify our `read_data` function:

- check if `S3_ENDPOINT_URL` is set, and if it is, use it for reading
- otherwise use the usual way

### Answer

See the adjusted script [batch_q4.py](batch_q4.py).

Run the following commands.

```bash
$ aws --endpoint-url=http://localhost:4566 s3 cp ~/downloads/yellow_tripdata_2022-02.parquet s3://nyc-duration/in/2022-02.parquet
upload: ../../../../downloads/yellow_tripdata_2022-02.parquet to s3://nyc-duration/in/2022-02.parquet
$ python batch_q4.py 2022 2 
predicted mean duration: 12.513422116701408
$ aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration --recursive
2023-07-10 13:46:12   45616512 in/2022-02.parquet
2023-07-10 13:35:04   20084276 out/2022-02.parquet
```

See this [section](https://docs.aws.amazon.com/cli/latest/reference/s3/index.html) explains prominent concepts and notations in the set of high-level S3 commands provided.


## Q5. Creating test data

Now let's create `integration_test.py`

We'll use the dataframe we created in Q3 (the dataframe for the unit test)
and save it to S3. You don't need to do anything else: just create a dataframe 
and save it.

We will pretend that this is data for January 2022.

Run the `integration_test.py` script. After that, use AWS CLI to verify that the 
file was created. 

Use this snipped for saving the file:

```python
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)
```

What's the size of the file?

- 3667
- 23667
- 43667
- ✅ 63667

Note: it's important to use the code from the snippet for saving
the file. Otherwise the size may be different depending on the OS,
engine and compression. Even if you use this exact snippet, the size
of your dataframe may still be a bit off. Just select the closest option.

### Answer

See the script [tests/integration_test_q5.py](tests/integration_test_q5.py).

Run the following commands.

```bash
$ python tests/integration_test.py 2022 01
$ aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration --recursive
2023-07-10 14:02:51   66624771 in/2022-01.parquet
2023-07-10 13:46:12   45616512 in/2022-02.parquet
2023-07-10 13:47:46   20084276 out/2022-02.parquet
$ aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration --recursive --human-readable --summarize
2023-07-10 14:02:51   63.5 MiB in/2022-01.parquet
2023-07-10 13:46:12   43.5 MiB in/2022-02.parquet
2023-07-10 13:47:46   19.2 MiB out/2022-02.parquet

Total Objects: 3
   Total Size: 126.2 MiB
```

See [AWS CLI S3 ls reference](https://docs.aws.amazon.com/cli/latest/reference/s3/ls.html).


## Q6. Finish the integration test

We can read from our localstack s3, but we also need to write to it.

Create a function `save_data` which works similarly to `read_data`,
but we use it for saving a dataframe. 

Let's run the `batch.py` script for "January 2022" (the fake data
we created in Q5). 

We can do that from our integration test in Python: we can use
`os.system` for doing that (there are other options too). 

Now it saves the result to localstack.

The only thing we need to do now is to read this data and 
verify the result is correct. 

What's the sum of predicted durations for the test dataframe?

- 10.50
- 31.51
- 59.28
- 81.22

### Answer

See the adjusted script [batch_q6.py](batch_q6.py).

Run the following commands.

```bash
$ python batch_q6.py 2022 01
predicted mean duration: 12.671278846471523
```

### Running the test (ungraded)

The rest is ready, but we need to write a shell script for doing 
that. 

Let's do that!


## Submit the results

* Submit your results here: https://forms.gle/vi7k972SKLmpwohG8
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
* You can submit your answers multiple times. In this case, the last submission will be used for scoring.

## Deadline

The deadline for submitting is 16 July (Sunday) 23:00 CEST. After that, the form will be closed.