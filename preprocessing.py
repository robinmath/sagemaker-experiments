# preprocessing.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType

def main():
    parser = argparse.ArgumentParser(description="Specify input and output formats.")
    parser.add_argument("--input-format", type=str, required=True, choices=["parquet", "csv"],
                        help="Specify the input file format: 'parquet' or 'csv'.")
    parser.add_argument("--output-format", type=str, required=True, choices=["parquet", "csv"],
                        help="Specify the output file format: 'parquet' or 'csv'.")
    args = parser.parse_args()
    input_format = args.input_format
    output_format = args.output_format

    # Initialize Spark Session
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    # Paths
    input_parquet = "s3://apartment-pricing/preprocessed/data/parquet/"
    input_csv = "s3://apartment-pricing/preprocessed/data/csv/"
    output_parquet = "s3://apartment-pricing/preprocessed/data/parquet/"
    output_csv = "s3://apartment-pricing/preprocessed/data/csv/"

    # Select input path based on the input format
    if input_format == "parquet":
        input_path = input_parquet
        df = spark.read.parquet(input_path)
    else:
        input_path = input_csv
        df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"Loaded data from {input_path}")

    # Transformations
    df = df.withColumn("schoolrating", col("schoolrating") * 10)

    # Select output path based on the output format
    if output_format == "parquet":
        output_path = output_parquet
        df.write.parquet(output_path, mode="overwrite")
    else:
        output_path = output_csv
        # Reorder columns: Ensure the target column ('label') is first
        target_column = "rent"
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Convert all columns to numeric, replacing errors with 0
        for column in [target_column] + feature_columns:  # Include the target column in the conversion
            df = df.withColumn(column, when(col(column).cast(DoubleType()).isNotNull(), col(column).cast(DoubleType())).otherwise(0))
        
        # Select the target column first and then features
        df = df.select(target_column, *feature_columns)
        
        # Write to CSV
        df.write.csv(output_path, mode="overwrite", header=False)


    print(f"Preprocessed data written to {output_path}")


if __name__ == "__main__":
    main()
