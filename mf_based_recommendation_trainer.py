import os
import math
from review_data_utils import *
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS


class MFBasedRecommendationTrainer:

    def __init__(self, base_dir, spark_context, sql_context, spark_session):
        self.base_dir = base_dir
        self.spark_context = spark_context
        self.sql_context = sql_context

    def __split_data(self, df_rdd):
        training_RDD, validation_RDD, test_RDD = df_rdd.randomSplit([6, 2, 2], seed=0)
        validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
        test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
        print (validation_RDD.take(10))
        return (training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD)

    def get_best_model(self, df_rdd):
        seed = 5
        iterations = 10
        regularization_parameter = 0.1
        ranks = [4, 8, 12]
        errors = [0, 0, 0]
        err = 0
        tolerance = 0.02

        min_error = float('inf')
        best_rank = -1
        best_iteration = -1

        sp = self.__split_data(df_rdd)
        training_RDD = sp[0]
        validation_RDD = sp[1]
        test_RDD = sp[2]
        validation_for_predict_RDD = sp[3]
        test_for_predict_RDD = sp[4]

        for rank in ranks:
            model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                              lambda_=regularization_parameter)
            predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
            rates_and_preds = validation_RDD.map(lambda r: ((float(r[0]), float(r[1])), float(r[2]))).join(predictions)
            error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            errors[err] = error
            err += 1
            print ('For rank %s the RMSE is %s' % (rank, error))
            if error < min_error:
                min_error = error
                best_rank = rank

        print ('The best model was trained with rank %s' % best_rank)
        print (predictions.take(3))
        print (rates_and_preds.take(3))

        best_model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                              lambda_=regularization_parameter)
        predictions = best_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = test_RDD.map(lambda r: ((float(r[0]), float(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        print ('For testing data the RMSE is %s' % (error))
        return best_model

    def export_model(self, model):
        model_file_name = "business_recomm_model"
        model_full_path = os.path.join(self.base_dir, "mf_based_models", model_file_name)
        model.save(self.spark_context, model_full_path)
        print ("{} saved!".format(model_file_name))


def build_model():
    appName = "mf based trainer app"

    conf = SparkConf().setAppName(appName).setMaster("local")
    spark_context = SparkContext(conf=conf)
    # sc.setLogLevel("WARN") # config log level to make console less verbose
    sql_context = SQLContext(spark_context)

    spark_session = SparkSession.builder \
            .master("local") \
            .appName(appName) \
            .getOrCreate()

    base_dir = "./data/"
    
    trainer = MFBasedRecommendationTrainer(base_dir, spark_context, sql_context, spark_session)
    df_rdd = load_and_parse_review_data(base_dir, spark_session)
    model = trainer.get_best_model(df_rdd)
    trainer.export_model(model)

if __name__ == '__main__':
    build_model()

    

