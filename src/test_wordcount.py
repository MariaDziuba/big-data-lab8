from pyspark.sql import SparkSession
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")


spark = SparkSession.builder \
    .master("local") \
    .appName("WordCount") \
    .getOrCreate()

sc = spark.sparkContext


def word_count(path_to_file: str):
    text_file = sc.textFile(path_to_file)

    return text_file.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .collect()


def main():

    config = configparser.ConfigParser()
    config.read('config.ini')

    test_input = os.path.join(cur_dir.parent.parent, config['data']['test_input'])

    word_counts = word_count(test_input)
    for word, count in word_counts:
        print(f'{word}: {count}')

    sc.stop()
    spark.stop()


if __name__ == '__main__':
    main()