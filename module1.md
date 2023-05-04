# Module 1: Introduction

See this [GitHub page](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro) 
on [DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

## 1.1 Introduction

[Youtube](https://www.youtube.com/watch?v=s0uaFZSzwfI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK).

## 1.2 Environment preparation (Linux/macOS)

[Youtube](https://www.youtube.com/watch?v=IXSiYkP23zo&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=3).

### Create AWS EC2 instance (00:45)

* Create a new account on AWS Console
* Go to EC2 et create a new instance `mlops-zoomcamp` with Ubuntu (x86), t2.xlarge.
* Create key pair (RSA type and `.pem` file format), and download this key to our local `~/.ssh/` folder as `~/.ssh/razer.pem`.
* Select 30 GiB 
* Launch instance

See also:

* [Setup Jupyter Notebook on Azure via SSH using Linux](https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/README.md)

### Connect local machine to the EC2 instance (03:27)

Connect to this instance with this command. 

```bash
ssh -i ~/.ssh/razer.pem ubuntu@34.248.245.41
```

You don't need to run this command every time. Just create a config file `~/.ssh/config` like this.

```bash
Host mlops-zoomcamp
    HostName 34.248.245.41
    User ubuntu
    IdentityFile c:/Users/alexe/.ssh/razer.pem
    StrictHostKeyChecking no
```

Every time we restart the instance, we will have to change the public IP address.

Now, we can connect to our instance with this command.

```bash
ssh mlops-zoomcamp
```

See also:

* [Reserve External Static Address](https://github.com/syahrulhamdani/dtc-mlops/blob/main/week-1-introduction/README.md#reserve-external-static-address) from [Syahrul Bahar Hamdani](https://github.com/syahrulhamdani).

### Install Anaconda (05:25)

Go to https://www.anaconda.com/download/ to get the latest version of Anaconda.
Copy the link for **Linux 64-Bit (x86) Installer**.
Run this command on the remote instance.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```

Update existing packages.
It's a good idea to run this command often, once per day or every few days, to keep your VM up to date.

```bash
sudo apt update && sudo apt -y upgrade
```

See also:

* [Conda Cheat sheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d#conda) from [Alvaro Navas](https://gist.github.com/ziritrion).
* [Conda Cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)


### Install Docker on Linux (06:40)

Install Docker on Ubuntu.

```bash
sudo apt install docker.io
```

Install Docker Compose in a separate directory on our instance.
Go to https://github.com/docker/compose/releases and copy the link of `docker-compose-linux-x86_64`.

```bash
mkdir soft
cd soft
wget https://github.com/docker/compose/releases/download/v2.17.3/docker-compose-linux-x86_64 -O docker-compose
chmod +x docker-compose
```

### Modify Path Variable (08:13)

Add the soft directory to PATH. Open the `.bashrc` file with `nano ~/.bashrc`
and add the following line:

```bash
export PATH="${HOME}/soft:${PATH}"
```

Save it (`Ctrl+O` to save and `Ctrl+X` to exit) and run the following command to make sure the changes are applied.

```bash
source .bashrc
which docker-compose
# /home/ubuntu/soft/docker-compose
```

### Add user to the Docker group (09:00)

Run Docker.

```bash
sudo docker run hello-world

# To run docker without sudo:
sudo groupadd docker
sudo usermod -aG docker $USER

# Log out and log back in so that your group membership is re-evaluated.
logout
ssh mlops-zoomcamp
docker run hello-world
```

### Init Anaconda (10:03)

Once Anaconda has finished its installation, we must initialize it.

We need also log out and log back before
check where Python is installed and see what Anaconda has changed.

```bash
logout
ssh mlops-zoomcamp
which python
# /home/ubuntu/anaconda3/bin/python
python
# Python 3.9.7
less .bashrc
```

See also:

* [Docker](https://www.docker.com/)
* [Docker Compose](https://docs.docker.com/compose/).
* [Docker & Docker Compose](https://gist.github.com/ziritrion/3214aa570e15ae09bf72c4587cb9d686#docker) from [Alvaro Navas](https://gist.github.com/ziritrion)
* [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)
* [Overview of nano's shortcuts](https://www.nano-editor.org/dist/latest/cheatsheet.html)

### Clone mlops-zoomcamp GitHub repo (10:50)

```bash
git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
cd mlops-zoomcamp
```

### Connect VS code with remote machine (11:48)

Open VS Code from your local machine. 
In VS Code, find and install the **Remote - SSH** extension. 
Then go to the **Command Palette** (`Shift+Cmd+P`) and select **Remote-SSH: Connect to Host…**​ and `mlops-zoomcamp`.

### Use Jupyter Notebook from remote machine (13:10)

On the remote instance, run the following commands.

```bash
cd ~
mkdir notebooks
cd notebooks/
jupyter notebook
```

In VS Code, open terminal, select **PORTS**, open the port `8888`

### Additional materials

#### Install pyenv instead of Anaconda

We may prefer [pyenv](https://github.com/pyenv/pyenv) and pipenv for managing python 
version and virtual environment. 
Anaconda could be just to heavy and not reliable for us.

The instructions below for [pyenv](https://github.com/pyenv/pyenv)
are from [Using pyenv instead of Anaconda](https://github.com/syahrulhamdani/dtc-mlops/blob/main/week-1-introduction/README.md#using-pyenv-instead-of-anaconda) from 
[Syahrul Bahar Hamdani](https://github.com/syahrulhamdani).

If you want it, you can follow these steps:

* if you love being hardcore, follow the installation steps with git 
in the official [docs](https://github.com/pyenv/pyenv#basic-github-checkout)
* if you want fast installation, use [pyenv-installer](https://github.com/pyenv/pyenv-installer)
* after that, continue to follow the instructions in pyenv official installation 
guide for [shell environment](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv)
* don't forget to `source` your `.profile` file (or `.bash_profile`, `.zshrc`, etc.)

Before installing specific python version, first you need to install some python dependencies.
Just follow this [wiki](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) 
and look for Ubuntu suggested build.

For this bootcamp, I use **Python 3.9.12** (no specific reason). The command used 
to install that version with pyenv:

```bash
pyenv install 3.9.12
```

If you want other versions, you can look at the available list by using command:

```bash
pyenv install --list
```

You can even install more than one python version using pyenv.

Last command to use the python version is to make it as global python version. Use:

```bash
pyenv global 3.9.12
```

For sanity check, type `python` and you should be entering Python 3.9.12 interface. 
Or, you can use command `pyenv which python`.
To check the list of python versions you already installed, just run `pyenv versions`.

To create a virtual environment using pyenv, just type:

```bash
pyenv virtualenv [version] [env_name]
```

For example, if I want to create a virtual environment using Python 3.9.7 and 
name it as `mlops-zoomcamp`, I just type:

```bash
pyenv virtualenv 3.9.7 mlops-zoomcamp
```

To activate it, I suggest you don't. You can use it directly in you desired 
workspace by attach it as local environment.
For example, if you want to use the environment in mlops-zoomcamp directory, 
just `cd` inside it, and run command below:

```bash
pyenv local mlops-zoomcamp
```

Every time you enter the directory, the virtual environment will be activated automatically.

See also:

* [Pipenv Cheat sheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d#pipenv) from [Alvaro Navas](https://gist.github.com/ziritrion).

#### Install Docker on macOS

Install Docker with Docker Compose already included.
Note that `--cask` flag installs modules into `/usr/local` directory.

```bash
brew install docker --cask
```

Run Docker.

```bash
docker run hello-world
```

#### Add an Anaconda env as Jupyter kernel

```bash
conda activate base
conda install -c anaconda ipykernel
python -m ipykernel install --user --name base-anaconda --display-name "(base) Anacoda Py 3.9.12"
jupyter kernelspec list
jupyter notebook
```

#### Upload/download files to/from your instance

1. Download a file.

    ```sh
    # From your local machine
    scp <instance_name>:path/to/remote/file path/to/local/file
    ```

1. Upload a file.

    ```sh
    # From your local machine
    scp path/to/local/file <instance_name>:path/to/remote/file
    ```

1. You can also drag & drop stuff in VSCode with the [remote SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

1. If you use a client like [https://cyberduck.io/](Cyberduck), you can connect with SFTP to your instance using the `instance.zone.project` name as server, and adding the generated private ssh key.

See also :

* [How to Use SCP Command to Securely Transfer Files](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/)

## 1.3 Reading Parquet files instead of CSV

[Youtube](https://www.youtube.com/watch?v=r94QjpX9vSE&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=4).

We will use [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

We need to install missing dependency.

```python
!pip install pyarrow
```

Below, the code to read a parquet file.

```python
import pandas as pd

!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2021-01.parquet

df = pd.read_parquet("/content/yellow_tripdata_2021-01.parquet")
df.dtypes
```

## 1.3 (Optional) Training a ride duration prediction model

[Youtube](https://www.youtube.com/watch?v=iRunifGSHFc&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=5).

See [duration-prediction.ipynb](https://github.com/syahrulhamdani/dtc-mlops/blob/main/week-1-introduction/duration-prediction.ipynb)

## 1.4 Course overview

[Youtube](https://www.youtube.com/watch?v=teP9KWkP6SM&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=6).

**Experiment Tracking** is the process of managing all the different experiments and their components, 
such as parameters, metrics, models and other artifacts and it enables us to:

* **Organize** all the necessary components of a specific experiment. It's important to have everything in one place and know where it is so you can use them later.
* **Reproduce** past results (easily) using saved experiments.
* **Log** iterative improvements across time, data, ideas, teams, etc.

A **Model Registry**  is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an Model.

One definition of an **ML pipeline** is a means of automating the machine learning workflow by enabling data to be transformed and
correlated into a model that can then be analyzed to achieve outputs. This type of ML pipeline makes the process of inputting data
into the ML model fully automated.

See also:

* [Experiment Tracking](https://madewithml.com/courses/mlops/experiment-tracking/)
* [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
* [ML Model Registry: What It Is, Why It Matters, How to Implement It](https://neptune.ai/blog/ml-model-registry)
* [ML Pipelines](https://www.databricks.com/glossary/what-are-ml-pipelines) from Databricks

## 1.5 MLOps maturity model

[Youtube](https://www.youtube.com/watch?v=XwTH8BDGzYk&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=7).

See [MLOps Maturity model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

## 1.6 Homework

See https://github.com/syahrulhamdani/dtc-mlops/blob/main/week-1-introduction/homework.ipynb