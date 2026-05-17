import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
       
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_gold_feature_store(snapshot_date_str, silver_feature_directory, gold_feature_store_directory, spark):

    snapshot_date_clean = snapshot_date_str.replace("-", "_")
    
    # Load silver feature tables
    clickstream_path = f"{silver_feature_directory}silver_clickstream_{snapshot_date_clean}.parquet"
    attributes_path = f"{silver_feature_directory}silver_attributes_{snapshot_date_clean}.parquet"
    financials_path = f"{silver_feature_directory}silver_financials_{snapshot_date_clean}.parquet"

    clickstream_df = spark.read.parquet(clickstream_path)
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)

    print(f"Loaded silver clickstream from: {clickstream_path}, row count: {clickstream_df.count()}")
    print(f"Loaded silver attributes from: {attributes_path}, row count: {attributes_df.count()}")
    print(f"Loaded silver financials from: {financials_path}, row count: {financials_df.count()}")

    for name, df in {
        "attributes": attributes_df,
        "financials": financials_df,
        "clickstream": clickstream_df
    }.items():
        print(f"Checking duplicates for {name}")
        df.groupBy("Customer_ID", "snapshot_date").count().filter(col("count") > 1).show()
    
    # Join feature tables
    gold_df = (
        attributes_df
        .join(financials_df, on=["Customer_ID", "snapshot_date"], how="left")
        .join(clickstream_df, on=["Customer_ID", "snapshot_date"], how="left")
    )

    # Feature creation
    gold_df = gold_df.withColumn("debt_to_income_ratio",F.when(col("Annual_Income") > 0,col("Outstanding_Debt") / col("Annual_Income")).otherwise(None))
    
    gold_df = gold_df.withColumn("emi_to_salary_ratio",F.when(col("Monthly_Inhand_Salary") > 0,col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).otherwise(None))
    
    gold_df = gold_df.withColumn("high_credit_utilization_flag",F.when(col("Credit_Utilization_Ratio") > 50,1).otherwise(0))

    
    print(f"Gold feature store row count for {snapshot_date_str}: {gold_df.count()}")

    # Save gold feature store
    partition_name = f"gold_feature_store_{snapshot_date_clean}.parquet"
    filepath = gold_feature_store_directory + partition_name

    gold_df.write.mode("overwrite").parquet(filepath)

    print("saved to:", filepath)

    return gold_df