import pathlib
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from preprocess import Preprocessor
from db import Database


class KMeansClustering:

    def __init__(self, db):
        self.db = db

    def clustering(self, scaled_data):
        evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol='scaled_features',
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )

        for k in range(2, 10):
            kmeans = KMeans(featuresCol='scaled_features', k=k)
            model = kmeans.fit(scaled_data)
            predictions = model.transform(scaled_data)
            self.db.insert_df(predictions.select("prediction"), "Predictions")
            score = evaluator.evaluate(predictions)
            print(f'k = {k}, silhouette score = {score}')


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    spark = SparkSession.builder \
    .appName(config['spark']['app_name']) \
    .master(config['spark']['deploy_mode']) \
    .config("spark.driver.cores", config['spark']['driver_cores']) \
    .config("spark.executor.cores", config['spark']['executor_cores']) \
    .config("spark.driver.memory", config['spark']['driver_memory']) \
    .config("spark.executor.memory", config['spark']['executor_memory']) \
    .config("spark.jars", config['spark']['mysql_connector_jar']) \
    .config("spark.driver.extraClassPath", config['spark']['mysql_connector_jar']) \
    .getOrCreate()

    db = Database(spark)
    # path_to_data = os.path.join(cur_dir.parent.parent, config['data']['small_openfoodfacts'])
    preprocessor = Preprocessor()

    assembled_data = preprocessor.load_dataset(db)
    scaled_data = preprocessor.scale_assembled_dataset(assembled_data)
    kmeans = KMeansClustering(db)
    kmeans.clustering(scaled_data)

    spark.stop()


if __name__ == '__main__':
    main()