"""
Finding and Downloading OCMR Data Files
"""
import pandas
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

# %% Load the ocmr_data_attributes file into a pandas DataFrame
ocmr_data_attributes_location = './ocmr_data_attributes.csv'

df = pandas.read_csv('ocmr_data_attributes.csv')
# Cleanup empty rows and columns
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)

# Show the first 10 items in the list
df.head(10)

# %% Filter the files based on their attributes
# This is a sample query that filters on file names that contain "fs_", scn equals "15avan", and viw equals "lax"
# (i.e. fully sampled, LAX view, collected on 1.5T Avanto)
selected_df = df.query('`file name`.str.contains("fs_") and scn=="15avan" and viw=="lax"', engine='python')

# %% Download each file from S3
# The local path where the files will be downloaded to
download_path = './ocmr_data'

# Replace this with the name of the OCMR S3 bucket
bucket_name = 'ocmr'

if not os.path.exists(download_path):
    os.makedirs(download_path)

count = 1
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Iterate through each row in the filtered DataFrame and download the file from S3.
# Note: Test after finalizing data in S3 bucket
for index, row in selected_df.iterrows():
    print('Downloading {} to {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
    s3_client.download_file(bucket_name, 'data/{}'.format(row['file name']),
                            '{}/{}'.format(download_path, row['file name']))
    count += 1


