import os
from review_data_utils import *
import hashlib
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, LongType

#USER_REVIEW_DATA = "yelp_academic_dataset_review.json"
USER_REVIEW_DATA = "yelp_academic_dataset_review_small.json"
def load_and_parse_review_data(base_dir,spark_session,file_name = USER_REVIEW_DATA):

    json_full_path = os.path.join(base_dir, file_name)

    df = spark_session.read.json(path=json_full_path)

    df_rdd = df.rdd

    def convert_row(row):
        user_id = int(hashlib.sha1(row.user_id.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        business_id = int(hashlib.sha1(row.business_id.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        stars = float(row.stars)
        return (user_id, business_id, stars)

    converted_df_rdd = df_rdd.map(convert_row)
    print(converted_df_rdd.take(3))
    return converted_df_rdd

if __name__ == '__main__':

    appName = "yelp file load tester"

    conf = SparkConf().setAppName(appName).setMaster("local")
    spark_context = SparkContext(conf=conf)
    sql_context = SQLContext(spark_context)

    spark_session = SparkSession.builder \
            .master("local") \
            .appName(appName) \
            .getOrCreate()

    base_dir = "./data/"

    USER_REVIEW_SMALL_DATA = "yelp_academic_dataset_review_small.json"

    data = load_and_parse_review_data(base_dir,spark_session,file_name = USER_REVIEW_SMALL_DATA)
