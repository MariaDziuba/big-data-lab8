package preprocess

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame


object Preprocessor {

  def fillNa(df: DataFrame): DataFrame = df.na.fill(0.0)

  def assembleVector(df: DataFrame): DataFrame = {
    val outputCol = "features"
    val inputCols = "completenes" :: "energy_kcal_100g" :: "energy_100g" :: "fat_100g" ::
      "saturated_fat_100g" :: "carbohydrates_100g" :: "sugars_100g" :: "proteins_100g" ::
      "salt_100g" :: "sodium_100g" :: Nil

    val vector_assembler = new VectorAssembler()
      .setInputCols(df.columns)
      .setOutputCol(outputCol)
      .setHandleInvalid("skip")
    vector_assembler.transform(df)
  }

  def scaleAssembledDataset(df: DataFrame): DataFrame = {
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
    val scalerModel = scaler.fit(df)
    scalerModel.transform(df)
  }
}