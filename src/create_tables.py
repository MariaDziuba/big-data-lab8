import mysql.connector as c
import pandas as pd
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")

db_connection = c.connect(
    user="root",
    password="password",
    database='lab6',
    host='127.0.0.1',
    port=3306
)

db_cursor = db_connection.cursor()

db_cursor.execute(
    "CREATE TABLE IF NOT EXISTS OpenFoodFacts(completeness FLOAT, energy_kcal_100g FLOAT, energy_100g FLOAT, fat_100g FLOAT, saturated_fat_100g FLOAT, carbohydrates_100g FLOAT, sugars_100g FLOAT, proteins_100g FLOAT, salt_100g FLOAT, sodium_100g FLOAT);")

config = configparser.ConfigParser()
config.read('config.ini')
path_to_data = os.path.join(cur_dir.parent.parent, config['data']['small_openfoodfacts'])
df = pd.read_csv(path_to_data, sep='\t')
cols = [
        'completeness',
        'energy-kcal_100g',
        'energy_100g',
        'fat_100g',
        'saturated-fat_100g',
        'carbohydrates_100g',
        'sugars_100g',
        'proteins_100g',
        'salt_100g',
        'sodium_100g'
]

df = df[cols]
df = df.dropna()
df_tuples = list(df.itertuples(index=False, name=None))
df_tuples_string = ",".join(["(" + ",".join([str(w) for w in wt]) + ")" for wt in df_tuples])
db_cursor.execute("INSERT INTO OpenFoodFacts(completeness, energy_kcal_100g, energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, proteins_100g, salt_100g, sodium_100g) VALUES" + df_tuples_string + ';')
db_cursor.execute("FLUSH TABLES;")

