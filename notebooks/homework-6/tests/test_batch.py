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