package ai.h2o.automl


import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, LogManager, Logger}

import org.apache.spark.sql._
import org.apache.spark.h2o._
import _root_.hex.tree.gbm.GBMModel
import water.support.{H2OFrameSupport, ModelSerializationSupport}
import water.support.H2OFrameSupport._
import water.support.ModelSerializationSupport._
import ai.h2o.sparkling.ml.models._


object my_h2o_pred extends H2OFrameSupport with ModelSerializationSupport  {
  def main(args: Array[String]): Unit = {
    val logger: Logger =LogManager.getLogger(this.getClass)
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark: SparkSession =SparkSession.builder().appName("PredictData").getOrCreate()
    import spark.implicits._


    val h2oContext: H2OContext =H2OContext.getOrCreate(spark)
    val sqlContext =spark.sqlContext

    val df=spark.read.parquet("file:///home/bdp/h2o/sichuan/submitData").drop("label")

    df.count()

    val gbmModel=loadH2OModel[GBMModel]("/home/bdp/h2o/sichuan/bestModel")

    H2OMOJOModel.createFromMojo(gbmModel.asBytes(),"av")
    val settings = H2OMOJOSettings(convertUnknownCategoricalLevelsToNa = true, convertInvalidNumbersToNa = true)
    val gbmMojomodel=H2OMOJOModel.createFromMojo("/home/bdp/h2o/sichuan/bestModel",settings)

    val gbmresult=gbmMojomodel.transform(df)
    val p=gbmresult.select(col("user") +: ((0 until 5).map(i=>$"prediction_output.probabilities"(i).as(i.toString))):_*)
    p.coalesce(1).write.mode(SaveMode.Overwrite).csv("file:///home/bdp/h2o/sichuan/h2o_probabilities")


    val h2oModelData=h2oContext.asH2OFrame(df)
//    val data=withLockAndUpdate(h2oframe){allStringVecToCategorical}
//    val h2oModelData=withLockAndUpdate(data){columnsToCategorical(_,Array("membership_level","gender","star_level"))}

    val result=gbmModel.score(h2oModelData)
    h2oModelData.add(result)
    h2oModelData.update()
    val scoreDF=h2oContext.asDataFrame(h2oModelData)
    scoreDF.groupBy("predict").count().show()


    val r=scoreDF.select("user","predict")
  r.coalesce(1).write.mode(SaveMode.Overwrite).csv("/home/bdp/h2o/sichuan/h2o_csv")
  }

}
