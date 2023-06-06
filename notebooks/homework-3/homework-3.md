## Homework

The goal of this homework is to familiarize users with workflow orchestration. 

Start with the orchestrate.py file in the 03-orchestration/3.4 folder
of the course repo: https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/03-orchestration/3.4/orchestrate.py


## Q1. Human-readable name

You’d like to give the first task, `read_data` a nicely formatted name.
How can you specify a task name?

> Hint: look in the docs at https://docs.prefect.io or 
> check out the doc string in a code editor.

- `@task(retries=3, retry_delay_seconds=2, name="Read taxi data")`
- `@task(retries=3, retry_delay_seconds=2, task_name="Read taxi data")`
- `@task(retries=3, retry_delay_seconds=2, task-name="Read taxi data")`
- `@task(retries=3, retry_delay_seconds=2, task_name_function=lambda x: f"Read taxi data")`


### Answer

See [Task arguments](https://docs.prefect.io/2.10.12/concepts/tasks/#task-arguments) for `@task` syntax.

## Q2. Cron

Cron is a common scheduling specification for workflows. 

Using the flow in `orchestrate.py`, create a deployment.
Schedule your deployment to run on the third day of every month at 9am UTC.
What’s the cron schedule for that?

- `0 9 3 * *`
- `0 0 9 3 *`
- `9 * 3 0 *`
- `* * 9 3 0`

### Answer 

The [crontab guru](https://crontab.guru/) tool helps to find the answer.

## Q3. RMSE 

Download the January 2023 Green Taxi data and use it for your training data.
Download the February 2023 Green Taxi data and use it for your validation data. 

Make sure you upload the data to GitHub so it is available for your deployment.

Create a custom flow run of your deployment from the UI. Choose Custom
Run for the flow and enter the file path as a string on the JSON tab under Parameters.

Make sure you have a worker running and polling the correct work pool.

View the results in the UI.

What’s the final RMSE to five decimal places?

- 6.67433
- 5.19931
- 8.89443
- 9.12250

### Answer 

First, create and activate a conda environment for this question.

```bash
mkdir mlops
cd mlops
conda create -n prefect-ops python=3.9.12
conda activate prefect-ops
```

I'm getting errors with the above instructions on my MacBook M1. So, I use the following instead.

```bash
# How to Manage Conda Environments on an Apple Silicon M1 Mac
# See https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12
create_x86_conda_environment prefect-ops python=3.9.12
python -V
```

You should see this.

```bash
Python 3.9.12
```

Fork the https://github.com/DataTalksClub/mlops-zoomcamp.git repository. 

Clone this forked repository and install packages.

```bash
git clone git@github.com:boisalai/mlops-zoomcamp.git
cd mlops-zoomcamp
conda activate prefect-ops
pip install -r 03-orchestration/requirements.txt
```

Modify the `03-orchestration/3.4/orchestrate.py` script to read the correct versions of the data files.

```python
def main_flow(
    train_path: str = "./data/green_tripdata_2023-01.parquet",
    val_path: str = "./data/green_tripdata_2023-02.parquet",
) -> None:
```

Download the two parquet files from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 
and save them in the directory `mlops-zoomcamp/data`.

```bash
cd mlops-zoomcamp
wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet
wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet
```

Remove `data/` and `*.parquet` from `.gitignore` file.

Upload to GitHub.

```bash
git add .
git commit -m "Parquet files added"
git push -u origin main
```

Start a local Prefect server by running the following.

```bash
prefect server start
```

You should see this.

```txt
 ___ ___ ___ ___ ___ ___ _____ 
| _ \ _ \ __| __| __/ __|_   _| 
|  _/   / _|| _|| _| (__  | |  
|_| |_|_\___|_| |___\___| |_|  

Configure Prefect to communicate with the server with:

    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

View the API reference documentation at http://127.0.0.1:4200/docs

Check out the dashboard at http://127.0.0.1:4200
```

Open another terminal window and run the following commands to set the Prefect API URL.

```bash
conda activate prefect-ops
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Run the following to initialize the project. 
Make sure some files are deleted before.

```bash
cd mlops-zoomcamp
rm deployment.yaml prefect.yaml .prefectignore 
rm -rf .prefect/
prefect project init
```

You should get this.

```txt
Created project in /Users/boisalai/GitHub/mlops-zoomcamp with the following new files:
.prefectignore
deployment.yaml
prefect.yaml
.prefect/
```

<!--
Upload again to GitHub.

```bash
git add .
git commit -m "Prefect deployment files added"
git push -u origin main
```
-->

Deploy a flow for this project by creating a deployment.

```bash
cd mlops-zoomcamp
prefect deploy 03-orchestration/3.4/orchestrate.py:main_flow -n homework -p zoompool
``` 

You should get something like this.

```txt
Deployment 'main-flow/homework' successfully created with id 'c2fa690d-af2a-4642-a8a5-dfe0a73b4822'.
View Deployment in UI: http://127.0.0.1:4200/deployments/deployment/c2fa690d-af2a-4642-a8a5-dfe0a73b4822

To execute flow runs from this deployment, start a worker that pulls work from the 'zoompool' work pool
````

Start a worker that pulls work from the `zoompool` work pool.

```bash
prefect worker start -p zoompool
```

You should get something like this.

```txt
Discovered worker type 'process' for work pool 'zoompool'.
Worker 'ProcessWorker ce7ff1bb-86bb-41a1-8771-880f37948250' started!
```

Open the Prefect UI at http://127.0.0.1:4200, select **Flows** tab, 
and **main-flow**. You should see **homework** deployment on `zoompool` work pool. Click on Quick run.

You should get something like this.

<table>
    <tr>
        <td>
            <img src="images\s57.png">
        </td>
        <td>
            <img src="images\s58.png">
        </td>
    </tr>
</table>


## Q4. RMSE (Markdown Artifact)

Download the February 2023 Green Taxi data and use it for your training data.
Download the March 2023 Green Taxi data and use it for your validation data. 

Create a Prefect Markdown artifact that displays the RMSE for the validation data.
Create a deployment and run it.

What’s the RMSE in the artifact to two decimal places ?

- 9.71
- 12.02
- 15.33
- 5.37


## Q5. Emails


It’s often helpful to be notified when something with your dataflow doesn’t work
as planned. Create an email notification for to use with your own Prefect server instance.
In your virtual environment, install the prefect-email integration with 

```bash
pip install prefect-email
```

Make sure you are connected to a running Prefect server instance through your
Prefect profile.
See the docs if needed: https://docs.prefect.io/latest/concepts/settings/#configuration-profiles

Register the new block with your server with 

```bash
prefect block register -m prefect_email
```

Remember that a block is a Prefect class with a nice UI form interface.
Block objects live on the server and can be created and accessed in your Python code. 

See the docs for how to authenticate by saving your email credentials to
a block and note that you will need an App Password to send emails with
Gmail and other services. Follow the instructions in the docs.

Create and save an `EmailServerCredentials` notification block.
Use the credentials block to send an email.

Test the notification functionality by running a deployment.

What is the name of the pre-built prefect-email task function?

- `send_email_message`
- `email_send_message`
- `send_email`
- `send_message`



## Q6. Prefect Cloud

The hosted Prefect Cloud lets you avoid running your own Prefect server and
has automations that allow you to get notifications when certain events occur
or don’t occur. 

Create a free forever Prefect Cloud account at app.prefect.cloud and connect
your workspace to it following the steps in the UI when you sign up. 

Set up an Automation from the UI that will send yourself an email when
a flow run completes. Run one of your existing deployments and check
your email to see the notification.

Make sure your active profile is pointing toward Prefect Cloud and
make sure you have a worker active.

What is the name of the second step in the Automation creation process?

- Details
- Trigger
- Actions
- The end


## Submit the results

* Submit your results here: https://forms.gle/nVSYH5fGGamdY1LaA
* You can submit your solution multiple times. In this case, only the last submission will be used
* If your answer doesn't match options exactly, select the closest one


## Deadline

The deadline for submitting is 12 June (Monday), 23:00 CEST (Berlin time). 

After that, the form will be closed.



8888888888888888




