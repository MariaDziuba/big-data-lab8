from pyspark.ml.feature import VectorAssembler, StandardScaler
from db import Database


class Preprocessor:

    def load_dataset(self, db: Database):
        dataset = db.read_table("OpenFoodFacts")
        # dataset = spark.read.csv(
            # path_to_data,
            # header=True,
            # inferSchema=True,
            # sep='\t',
        # )
        dataset.fillna(value=0)

        output_col = 'features'
        vector_assembler = VectorAssembler(
            inputCols=[
                # 'last_updated_t',
                'completeness',
                # 'last_image_t',
                'energy_kcal_100g',
                'energy_100g',
                'fat_100g',
                'saturated_fat_100g',
                'carbohydrates_100g',
                'sugars_100g',
                'proteins_100g',
                'salt_100g',
                'sodium_100g'
            ],
            outputCol=output_col,
            handleInvalid='skip',
        )

        assembled_data = vector_assembler.transform(dataset)

        return assembled_data


    def scale_assembled_dataset(self, assembled_data):
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features'
        )
        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)

        return scaled_data