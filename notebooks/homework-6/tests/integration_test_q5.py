#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd

options = {
    'client_kwargs': {
        'endpoint_url': "http://localhost:4566"
    }
}

def read_data(filename: str) -> pd.DataFrame:
    """Read data"""
    df = pd.read_parquet(filename)
    return df

def get_input_path(year, month) -> str:
    """Get input path"""
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def main():
    """Main function"""
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_input = read_data(input_file)
    
    s3_input_file = get_input_path(year, month)
    df_input.to_parquet(
        s3_input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


if __name__ == "__main__":
    os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    main()