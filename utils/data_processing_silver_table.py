import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_features(snapshot_date_str, bronze_feature_directory, silver_feature_directory, spark):

    snapshot_date_clean = snapshot_date_str.replace("-", "_")

    bronze_files = {
        "clickstream": f"{bronze_feature_directory}bronze_clickstream_{snapshot_date_clean}.csv",
        "attributes": f"{bronze_feature_directory}bronze_attributes_{snapshot_date_clean}.csv",
        "financials": f"{bronze_feature_directory}bronze_financials_{snapshot_date_clean}.csv"
    }

    for feature_name, file_path in bronze_files.items():
        print(f"Processing silver feature table: {feature_name} - {snapshot_date_str}")

        df = (
            spark.read
            .csv(file_path, header=True, inferSchema=True)
            .dropDuplicates()
        )
        
        # Clean up columns with "_" in Financials
        if feature_name == "financials":
        
            double_columns = [
                "Annual_Income",
                "Monthly_Inhand_Salary",
                "Changed_Credit_Limit",
                "Num_Credit_Inquiries",
                "Outstanding_Debt",
                "Credit_Utilization_Ratio",
                "Total_EMI_per_month",
                "Amount_invested_monthly",
                "Monthly_Balance"
            ]
        
            integer_columns = [
                "Num_Bank_Accounts",
                "Num_Credit_Card",
                "Interest_Rate",
                "Num_of_Loan",
                "Delay_from_due_date",
                "Num_of_Delayed_Payment"
            ]
        
            string_columns = [
                "Customer_ID",
                "Type_of_Loan",
                "Credit_Mix",
                "Credit_History_Age",
                "Payment_of_Min_Amount",
                "Payment_Behaviour"
            ]
        
            for column_name in double_columns:
                if column_name in df.columns:
                    df = df.withColumn(
                        column_name,
                        F.regexp_replace(col(column_name).cast("string"), "_", "").cast(FloatType())
                    )
        
            for column_name in integer_columns:
                if column_name in df.columns:
                    df = df.withColumn(
                        column_name,
                        F.regexp_replace(col(column_name).cast("string"), "_", "").cast(IntegerType())
                    )
        
            for column_name in string_columns:
                if column_name in df.columns:
                    df = df.withColumn(column_name, col(column_name).cast(StringType()))

    if "snapshot_date" in df.columns:
        df = df.withColumn("snapshot_date", F.to_date(col("snapshot_date")))

        if "snapshot_date" in df.columns:
            df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))

        output_path = f"{silver_feature_directory}silver_{feature_name}_{snapshot_date_clean}.parquet"

        (
            df.write
            .mode("overwrite")
            .parquet(output_path)
        )

        print(f"saved to: {output_path}")
        