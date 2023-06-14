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
