import pandas as pd

class Database:
    def __init__(self, spark, host="127.0.0.1", port=3306, database='lab6'):
        self.jdbcUrl = f"jdbc:mysql://{host}:{port}/{database}"
        self.username = "root"
        self.password = "password"
        self.spark = spark
    
    def read_table(self, tablename: str):
        return self.spark.read \
            .format("jdbc") \
            .option("url", self.jdbcUrl) \
            .option("user", self.username) \
            .option("password", self.password) \
            .option("dbtable", tablename) \
            .option("inferSchema", "true") \
            .load()
    
    def insert_df(self, df, tablename):
        df.write \
            .format("jdbc") \
            .option("url", self.jdbcUrl) \
            .option("user", self.username) \
            .option("password", self.password) \
            .option("dbtable", tablename) \
            .mode("append") \
            .save()
            