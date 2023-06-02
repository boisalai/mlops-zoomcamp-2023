# Module 3: Orchestration and ML Pipelines

See this [GitHub page](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/03-orchestration) 
on [DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

In module 3, we will learn to use Prefect to orchestrate and observe our ML workflows.

## 3.1 Introdution to Workflow Orchestration

:movie_camera: [Youtube](https://www.youtube.com/watch?v=Cqb7wyaNF08&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=16).

Key Takeaways:

* The video discusses orchestration and machine learning (ML) pipelines with Prefect
* It introduces the concept of workflow orchestration and explains its importance for ML projects
* The challenges of working with complex systems are discussed, and how orchestration can help manage these challenges
* The benefits of using an orchestration tool like Prefect are also highlighted, including increased efficiency and reduced errors
* The video then dives into the Prefect UI tools and shows how they can help manage workflows through task visualization, error handling, and monitoring
* The video is a useful resource for anyone looking to learn more about workflow orchestration and how it can improve efficiency in ML pipelines.

### Intro to orchestration & ML pipelines with Prefect

> [00:00](https://www.youtube.com/watch?v=Cqb7wyaNF08&t=0s) - Intro to orchestration & ML pipelines with Prefect.

You might have an MLOps workflow that looks like this.

![MLOps](images/s55.png)

You could have:

* a database that retrieves data
* this could be picked up by a python script using pandas code to clean it up
* data is saved in a parquet file as a checkpoint, maybe readed back later
* scikit-learn used for engineering some features or running some models
* XGBoost to run a model
* MLflow can track anything saved by a database and information is written back and forth
* finally that your model is served via a [Flask](https://flask.palletsprojects.com/) or something like [FastAPI](https://fastapi.tiangolo.com/lo/)

We can have points of failure at many different parts of the workflow.

### Overview of challenges and benefits of workflow orchestration

> [02:49](https://www.youtube.com/watch?v=Cqb7wyaNF08&t=169s) - Overview of challenges and benefits of workflow orchestration.

If you give an MLOps engineer a job...

* Could you just set up this pipeline to train this model?
* Could you set up logging?
* Could you do it every day?
* Could you make it retry if it fails?
* Could you send me a message when it succeeds?
* Could you visualize the dependencies?
* Could you add caching?
* Could you add collaborators to run ad hoc - who don't code?

What is Prefect?<br>
It’s a flexible framework to build, reliably execute and observe your dataflow while supporting a wide variety of execution and data access patterns.<br>
See [Why Prefect](https://www.prefect.io/guide/blog/why-prefect/) from Anna Geller.

### Review: Prefect UI tools for complex systems and ML workflows

> [05:15](https://www.youtube.com/watch?v=Cqb7wyaNF08&t=315s) - Review: Prefect UI tools for complex systems and ML workflows.

In the Prefect UI you can quickly set up notifications, visualize run history, and schedule your dataflows.

![MLOps](images/s56.png)

Source: https://docs.prefect.io/2.10.12/ 

Prefect provides tools for working with comple systems so you can stop wondering about your workflows.

## 3.2 Introduction to Prefect

:movie_camera: [Youtube](https://www.youtube.com/watch?v=rTUBTvXvXvM&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=17).

Key Takeaways:

* The video is about Prefect and its various components.
* The video will provide an overview of Prefect terminology and show how to configure a local database.
* The process of setting up the environment and running scripts on the Prefect server will be demonstrated.
* The video will also show how to use retry logic and the workflow UI.
* Flow runs and logs in Prefect will be reviewed towards the end of the video.
* The video is suitable for those interested in learning more about Prefect and its capabilities.

### Introduction to Prefect and its components

> [00:00](https://www.youtube.com/watch?v=rTUBTvXvXvM&t=0s) - Introduction to Prefect and its components.

In this section we will see how regular python code can be converted into a perfect script and
we will run our own prefect server locally and run scripts on the server.

Goals:

* Clone GitHub repository
* Setup a conda environment
* Start a Prefect server
* Run a Prefect flow
* Checkout Prefect UI

Why use Prefect? Flexible, open-source Python framwork to turn standard pipelines into fault-tolerant dataflows.

Installation? See https://docs.prefect.io/latest/getting-started/installation/

Prefect is published as a Python package. To install the latest Prefect release, run the following in a shell or terminal session:

```bash
pip install -U prefect
```

### Overview of Prefect terminology and local database configuration

> [02:27](https://www.youtube.com/watch?v=rTUBTvXvXvM&t=147s) - Overview of Prefect terminology and local database configuration.

Self Hosting a Prefect Server

* **Orchestration API** - Used by server to work with workflow metadata
* **Database** - Stored workflow metadata
* **UI** - Visualizes workflows
* **Hosting a Prefect server** - See https://docs.prefect.io/latest/host/
* **Task** - A discrete unit of work in a Prefect workflow. See https://docs.prefect.io/latest/concepts/tasks/
* **Flow** - Container for workflow logic. See https://docs.prefect.io/latest/concepts/flows/
* **Subflow** - Flow called by another flow. See https://docs.prefect.io/latest/concepts/flows/#composing-flows

Below an example.

```python
from prefect import flow, task

@task(name="Print Hello")
def print_hello(name):
    msg = f"Hello {name}!"
    print(msg)
    return msg

@flow(name="Subflow")
def my_subflow(msg):
    print(f"Subflow says: {msg}")

@flow(name="Hello Flow")
def hello_world(name="world"):
    message = print_hello(name)
    my_subflow(message)

hello_world("Marvin")
```

### Setting up environment and running scripts on Prefect server

> [05:30](https://www.youtube.com/watch?v=rTUBTvXvXvM&t=330s) - Setting up environment and running scripts on Prefect server.

Create a conda environment.

```bash
mkdir mlops
cd mlops
# conda create -n prefect-ops python=3.9.12
# conda activate prefect-ops
# How to Manage Conda Environments on an Apple Silicon M1 Mac
# See https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12
create_x86_conda_environment prefect-ops python=3.9.12
python -V
```

You should see this.

```txt
Python 3.9.12
```

Install GitHub repo and packages.

```bash
git clone https://github.com/discdiver/prefect-mlops-zoomcamp.git
cd prefect-mlops-zoomcamp
pip install -r requirements.txt
prefect version
```

You should see this.

```txt
Version:             2.10.8
API version:         0.8.4
Python version:      3.9.12
Git commit:          79093235
Built:               Mon, May 8, 2023 12:23 PM
OS/Arch:             darwin/x86_64
Profile:             default
Server type:         server
```

Start Prefect server.

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
cd mlops/prefect-mlops-zoomcamp
conda activate prefect-ops
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

You should see this.

```txt
Set 'PREFECT_API_URL' to 'http://127.0.0.1:4200/api'.
Updated profile 'default'.
```


### Demo of retry logic & workflow UI in action

> [9:50](https://www.youtube.com/watch?v=rTUBTvXvXvM&t=590s) - Demo of retry logic & workflow UI in action.

Go to the `mlops/prefect-mlops-zoomcamp/3.2` folder.

```bash
cd mlops/prefect-mlops-zoomcamp/3.2
ls
```

There are two Python scripts `cat_facts.py` and `cat_dog_facts.py` in this folder.
Below, the first one (`cat_facts.py`).

```python
import httpx
from prefect import flow, task


@task(retries=4, retry_delay_seconds=0.1, log_prints=True)
def fetch_cat_fact():
    cat_fact = httpx.get("https://f3-vyx5c2hfpq-ue.a.run.app/")
    # An endpoint that is designed to fail sporadically
    if cat_fact.status_code >= 400:
        raise Exception()
    print(cat_fact.text)


@flow
def fetch():
    fetch_cat_fact()


if __name__ == "__main__":
    fetch()
```

This script calls an API to retreive cats. The function calling the API has been decorated with a task decorator
which has been configured with some arguments.

* Prefect will retry the task upo to 4 times if the task were to fail for some reason.
* Between each retry, Prefect will wait for a short period of time before trying to run the task again.
* Lastly, any print statements that are made within this task will be shared within the logs whenever this script is run.

Run this script with the following commands.

```bash
cd mlops/prefect-mlops-zoomcamp/3.2
conda activate prefect-ops
python cat_facts.py
```

You should see this in the terminal and in the Prefect dashboard at http://127.0.0.1:4200.

On the left, we see that the flow encountered an exception during the execution.
On the right, we see a timeline of the flow run and the logs that were produced down at the bottom.

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

Let's try running the other script in the folder.

Below, the `cat_dog_facts.py` code.

```python
import httpx
from prefect import flow

@flow
def fetch_cat_fact():
    '''A flow that gets a cat fact'''
    return httpx.get("https://catfact.ninja/fact?max_length=140").json()["fact"]

@flow
def fetch_dog_fact():
    '''A flow that gets a dog fact'''
    return httpx.get(
        "https://dogapi.dog/api/v2/facts",
        headers={"accept": "application/json"},
    ).json()["data"][0]["attributes"]["body"]

@flow(log_prints=True)
def animal_facts():
    cat_fact = fetch_cat_fact()
    dog_fact = fetch_dog_fact()
    print(f"🐱: {cat_fact} \n🐶: {dog_fact}")

if __name__ == "__main__":
    animal_facts()
```

We have a parent flow at the bottom which calls the `dog_fact` flow and `cat_fact` flow.
Run this script with the following commands and see what happens.

```bash
cd mlops/prefect-mlops-zoomcamp/3.2
conda activate prefect-ops
python cat_dog_facts.py
```

You should see three new flow run records (`animal-facts`, `fetch-cat-fact`, `fetch-dog-fact`) that's because we had three flows in the script.
Let's take a look at the record for the parent flow called `animal-facts`.

<table>
    <tr>
        <td>
            <img src="images\s59.png">
        </td>
        <td>
            <img src="images\s60.png">
        </td>
    </tr>
</table>

## 3.3 Prefect Workflow

## 3.4 Deploying Your Workflow

## 3.5 Working with Deployments

## 3.6 Prefect Cloud (optional)

## 3.7 Homework

Coming soon!

## Quick setup

### Install packages

In a conda environment with Python 3.10.12 or similar, install all package dependencies with

```bash
git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
cd mlops-zoomcamp
cd 03-orchestration 
conda create -n prefect-env python=3.10
conda activate prefect-env
pip install -r requirements.txt
```

### Start the Prefect server locally

Create another window and activate your conda environment. Start the Prefect API server locally with

```bash
prefect server start
``` 

### Alternative to self-hosted server use Prefect Cloud for added capabilties

Signup and use for free at https://app.prefect.cloud

Authenticate through the terminal with

```bash
prefect cloud login
```

Use your [Prefect profile](https://docs.prefect.io/latest/concepts/settings/) to switch between a self-hosted server and Cloud.

