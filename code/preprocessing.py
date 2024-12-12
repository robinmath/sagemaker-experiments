# preprocessing.py


from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    VectorIndexer,
)
from pyspark.sql.functions import *
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)


def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()


    import boto3
    import json
    from urllib.parse import quote_plus
    from pyspark.sql.functions import col

    # Fetch the secret
    secretsmanager_client = boto3.client('secretsmanager', region_name='us-east-1')  # Specify your region
    secret_response = secretsmanager_client.get_secret_value(SecretId="redshift!default-namespace-admin")
    
    # Parse the SecretString
    credentials = json.loads(secret_response['SecretString'])
    
    # Extract username and password
    username = credentials['username']
    password = credentials['password']
    
    encoded_password = quote_plus(password)
    
    # Configuration options
    jdbc_url = f"jdbc:redshift://default-workgroup.442426877041.us-east-1.redshift-serverless.amazonaws.com:5439/re-pricing-db?user={username}&password={encoded_password}"
    aws_iam_role_arn = "arn:aws:iam::442426877041:role/service-role/AmazonRedshift-CommandsAccessRole-20241009T164003"
    s3_temp_dir = "s3://apartment-pricing/temp/"
    output_s3_path = "s3://apartment-pricing/preprocessed/data/"

    # SQL query to execute
    query = """SELECT * FROM "train-db".rental_pricing_table"""
    
    # Read from Redshift
    df = spark.read \
        .format("io.github.spark_redshift_community.spark.redshift") \
        .option("url", jdbc_url) \
        .option("query", query) \
        .option("tempdir", s3_temp_dir) \
        .option("aws_iam_role", aws_iam_role_arn) \
        .load()
    
    
    df = df.withColumn("walkscore", col("walkscore") / 10)
    
    # Save preprocessed data to S3
    df.write.parquet(output_s3_path, mode="overwrite")
    print(f"Preprocessed data written to {output_s3_path}")


if __name__ == "__main__":
    main()
