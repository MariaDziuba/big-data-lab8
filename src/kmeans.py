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
from datamart import DataMart


class KMeansClustering:

    def __init__(self, datamart):
        self.datamart = datamart

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
            self.datamart.write_predictions(predictions.select("prediction"))
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
    .config("spark.jars", f"{config['spark']['mysql_connector_jar']},jars/datamart.jar") \
    .config("spark.driver.extraClassPath", config['spark']['mysql_connector_jar']) \
    .getOrCreate()

    datamart = DataMart(spark=spark)

    assembled_data = datamart.read_dataset()
    kmeans = KMeansClustering(datamart)
    kmeans.clustering(assembled_data)

    spark.stop()


if __name__ == '__main__':
    main()