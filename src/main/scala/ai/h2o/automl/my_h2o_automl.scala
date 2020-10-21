package ai.h2o.automl

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.types._
import _root_.hex.Model
import org.apache.spark.h2o._
import ai.h2o.automl.Algo._
import water.fvec.{Frame, Vec}
import water.util.FrameUtils
import water._
import water.support.{H2OFrameSupport, ModelSerializationSupport}
import water.util.TwoDimTable
import scala.collection.mutable.ArrayBuffer
import java.text.SimpleDateFormat

import _root_.hex.tree.gbm.GBMModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.Bucketizer


object my_h2o_automl {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark: SparkSession =SparkSession.builder()
        .appName("H2OModelBuilding")
        .getOrCreate()

    val conf=new H2OConf(spark)
    val h2oContext: H2OContext =H2OContext.getOrCreate(spark,conf)
    val sqlContext: SQLContext =spark.sqlContext
    val h2oModel=new my_h2o_automl(spark,h2oContext)

    val modelData=h2oModel.getModelData
//    val (trainFrame,testFrame)=h2oModel.splitFrame(modelData)
    val trainFrame=h2oModel.splitFrame(modelData)

    val modelConf=h2oModel.modelConf(trainFrame)
    val myAutoml=h2oModel.train(modelConf)

    val bestModel=myAutoml.leader()
    h2oModel.logger.info("bestModel info==========================================================")
    bestModel.exportBinaryModel("/home/bdp/h2o/sichuan/bestModel",true)
//    h2oModel.score(testFrame,bestModel)
    h2oModel.logger.info(bestModel.toString)


    val bestModelvarimp=h2oModel.getVarImp(bestModel)
    if(bestModelvarimp.nonEmpty)h2oModel.logger.info(bestModelvarimp)
    else h2oModel.logger.info(s"${bestModel._key.toString}未返回字段重要性排序情况！")
    h2oModel.logger.info("===========================================================================")


    myAutoml.leaderboard().getModels().foreach(
      m=>{
        h2oModel.logger.info(s"---------------------${m._key.toString}---------------------------")
        h2oModel.logger.info(m.toString)
        val varimp=h2oModel.getVarImp(m)
        if(varimp.nonEmpty)h2oModel.logger.info(varimp)
        else h2oModel.logger.info(s"${m._key.toString}未返回字段重要性排序情况！")
//        h2oModel.score(testFrame,m)
        m.exportBinaryModel(s"/home/bdp/h2o/sichuan/allModel/${m._key.toString}",true)
      }
    )
    h2oModel.logger.info("建模完毕！")

    h2oContext.stop()
    spark.stop()
  }
}


class my_h2o_automl(@transient  val spark:SparkSession,
                 @transient val h2oContext:H2OContext)
    extends H2OFrameSupport with ModelSerializationSupport with java.io.Serializable {
  @transient val logger: Logger =LogManager.getLogger(this.getClass)
  import spark.implicits._
  import h2oContext.implicits._


  def getModelData: DataFrame ={
    val allData=spark.read.parquet("file:///home/bdp/h2o/sichuan/allData")
    val featureData=allData.select("user",modelCols :+ "label":_*)

    val submitData=featureData.filter("label is null")
    logger.info(s"提交行数为：${submitData.count()}")
    submitData.write.mode(SaveMode.Overwrite).parquet("file:///home/bdp/h2o/sichuan/submitData")

    val modelData=featureData.filter("label is not null")
    modelData.printSchema()
    logger.info(s"样本总数为：${modelData.count()}")

    modelData
  }


  def modelConf(modelData:Frame): AutoMLBuildSpec ={
    val autoMLBuildSpec=new AutoMLBuildSpec()
    autoMLBuildSpec.input_spec.training_frame=modelData._key
    autoMLBuildSpec.input_spec.response_column="label"
    autoMLBuildSpec.input_spec.ignored_columns=Array("user")

    autoMLBuildSpec.build_control.project_name = "ModelBuilding.proj"
    autoMLBuildSpec.build_control.stopping_criteria.set_max_models(12)
    autoMLBuildSpec.build_control.stopping_criteria.set_max_runtime_secs(14400)
    autoMLBuildSpec.build_control.keep_cross_validation_models=true
    autoMLBuildSpec.build_control.keep_cross_validation_predictions=true
    autoMLBuildSpec.build_control.nfolds=5
//    autoMLBuildSpec.build_control.stopping_criteria.set_stopping_metric()

    autoMLBuildSpec.build_control.balance_classes = true
    autoMLBuildSpec.build_control.class_sampling_factors = new Array[Float](5)
    autoMLBuildSpec.build_control.class_sampling_factors(0) =1.0f
    autoMLBuildSpec.build_control.class_sampling_factors(1) =1.0f
    autoMLBuildSpec.build_control.class_sampling_factors(2) =0.2f
    autoMLBuildSpec.build_control.class_sampling_factors(3) =1.0f
    autoMLBuildSpec.build_control.class_sampling_factors(4) =1.0f
    autoMLBuildSpec.build_control.max_after_balance_size=2.0f
    autoMLBuildSpec.build_models.exclude_algos=Array(XGBoost)

    autoMLBuildSpec
  }


  def splitFrame(df:DataFrame) ={
    val h2oframe=h2oContext.asH2OFrame(df)
    val data=withLockAndUpdate(h2oframe){allStringVecToCategorical}
    val h2oModelData=withLockAndUpdate(data){
      columnsToCategorical(_,
        Array("gender","star_level","last_year_capture_user_flag",
          "market_price_level","cust_point_level","credit_level","dt_m_1012_type",
      "dt_m_1027_type","dt_m_1032_type","dt_m_1034_type","dt_m_1075_type","dt_m_1086_type","dt_m_1087_type",
      "dt_m_1096_type","dt_m_1102_type","dt_m_1108_type","dt_m_1594_type","dt_m_1617_type","dt_m_1620_type",
      "dt_m_1630_type","dt_m_1633_type","app1_visits_type","app2_visits_type","app3_visits_type","dt_m_1035_pref",
      "app4_visits_type","app5_visits_type","app6_visits_type","app7_visits_type","app8_visits_type","dt_m_1087_pref"))}

    logger.info("========================================================================================================")
    parseResults(h2oModelData)
    logger.info("========================================================================================================")

//    val frs = splitFrame(h2oModelData, Array("train.hex", "test.hex"), Array(0.8, 0.2))
//    (frs(0), frs(1))
    h2oModelData
  }

  val modelCols=Array("credit_level","gender","inet_pd_inst_cnt","star_level",
    "dt_m_1000","dt_m_1003","dt_m_1004","dt_m_1005","dt_m_1006","dt_m_1009","dt_m_1011","dt_m_1012","dt_m_1015","dt_m_1017",
    "dt_m_1027","dt_m_1028","dt_m_1041","dt_m_1044","dt_m_1051","dt_m_1053","dt_m_1067",
    "dt_m_1069","dt_m_1073","dt_m_1074","dt_m_1075","dt_m_1085","dt_m_1086","dt_m_1087","dt_m_1096","dt_m_1099","dt_m_1102",
    "dt_m_1105","dt_m_1108","dt_m_1111","dt_m_1594","dt_m_1601","dt_m_1617","dt_m_1618","dt_m_1620","dt_m_1630","dt_m_1633",
    "last_year_capture_user_flag","app1_visits","app2_visits","app3_visits","app4_visits","app5_visits","app6_visits",
    "app7_visits","app8_visits","access_net_dur","tmlRegister_net_dur","prdOpen_dur","tmlRegister_dur","prdOpen_net_dur",
    "brand","product","cust_point_level","age","market_price_level","app_cnt","dt_m_1012_type",
    "dt_m_1027_type","dt_m_1032_type","dt_m_1034_type","dt_m_1075_type","dt_m_1086_type","dt_m_1087_type",
    "dt_m_1096_type","dt_m_1102_type","dt_m_1108_type","dt_m_1594_type","dt_m_1617_type","dt_m_1620_type",
    "dt_m_1630_type","dt_m_1633_type","app1_visits_type","app2_visits_type","app3_visits_type",
    "app4_visits_type","app5_visits_type","app6_visits_type","app7_visits_type","app8_visits_type",
    "dt_m_1087_pref","in_10s_per","in_10_30s_per","in_30_60s_per","out_60s_per","dt_m_1035_pref")

  private def train(conf:AutoMLBuildSpec)={
    val aml =AutoML.startAutoML(conf)

    aml.get()

    logger.info("训练结束!=============================================")
    val ld=aml.leaderboard().toTwoDimTable()
    val leaderBoard =leaderboardAsSparkFrame(ld)
    leaderBoard match {
      case Some(df)=>{
        logger.info("排行榜信息！========================================================================================================")
        df.show()
//        df.coalesce(1).write.option("header",true).mode(SaveMode.Overwrite).csv("file:///home/bdp/h2o/sichuan/leaderBoard")
        logger.info("========================================================================================================")
      }
      case None =>{
        logger.info("========================================================================================================")
        logger.warn("没有返回任何信息！")
        logger.info("========================================================================================================")
      }
    }

    aml
  }

  def score(h2ofr:Frame,model:Model[_,_,_]): Unit ={
    val keyid=model._key.toString
    val fr=h2oContext.asH2OFrame(h2ofr.deepCopy(s"${keyid}_frame.hex"))
    val testResult=model.score(fr)
    fr.add(testResult)
    fr.update()

    val scoreDF=h2oContext.asDataFrame(fr)

    logger.info("混淆矩阵========================================================================================================")
    scoreDF.groupBy("label").pivot("predict").count().show()

  }


  private def leaderboardAsSparkFrame(dt: TwoDimTable): Option[DataFrame] = {
    // Get LeaderBoard
    val colNames = dt.getColHeaders
    val data =dt.getCellValues.map(_.map(_.toString))
    val rows = data.map {
      Row.fromSeq(_)
    }
    val schema = StructType(colNames.map { name => StructField(name, StringType) })
    val rdd = h2oContext.sparkContext.parallelize(rows)
    Some(h2oContext.sparkSession.createDataFrame(rdd, schema))
  }


  private def parseResults(fr: Frame): Unit = {
    val numRows = fr.anyVec.length
    logger.info("Parse result for " + fr._key + " (" + java.lang.Long.toString(numRows) + " rows, " + Integer.toString(fr.numCols) + " columns):")
    // get all rollups started in parallell, otherwise this takes ages!
    val fs = new Futures
    val vecArr = fr.vecs
    for (v <- vecArr) {v.startRollupStats(fs)}
    fs.blockForPending()

    var namelen = 0
    for (s <- fr.names) {namelen = Math.max(namelen, s.length)}

    val format = " %" + namelen + "s %7s %12.12s %12.12s %12.12s %12.12s %11s %8s %6s"
    logger.info(String.format(format, "ColV2", "type", "min", "max", "mean", "sigma", "NAs", "constant", "cardinality"))
    val sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

    println(vecArr.toString)

    val parseBuffer=new ArrayBuffer[parseInfo]()
    for ( i <- 0 until vecArr.length) {
      val v = vecArr(i)
      val isCategorical = v.isCategorical
      val isConstant = v.isConst
      val CStr = String.format("%" + namelen + "s:", fr.names()(i))
      var typeStr = ""
      var minStr = ""
      var maxStr = ""
      var meanStr = ""
      var sigmaStr = ""
      v.get_type match {
        case Vec.T_BAD => typeStr = "all_NA"
        case Vec.T_UUID => typeStr = "UUID"
        case Vec.T_STR => typeStr = "string"
        case Vec.T_NUM =>
          typeStr = "numeric"
          minStr = s"${new java.lang.Double(v.min)}"
          maxStr = s"${new java.lang.Double(v.max)}"
          meanStr = s"${new java.lang.Double(v.mean)}"
          sigmaStr = s"${new java.lang.Double(v.sigma)}"
        case Vec.T_CAT =>
          typeStr = "factor"
          minStr = v.factor(0)
          maxStr = v.factor(v.cardinality - 1)
        case Vec.T_TIME =>
          typeStr = "time"
          minStr = sdf.format(v.min)
          maxStr = sdf.format(v.max)
        case _ => throw H2O.unimpl
      }
      val numNAs = v.naCnt
      val naStr = if (numNAs > 0) s"${new java.lang.Long(numNAs)}" else ""
      val isConstantStr = if (isConstant) "constant" else ""
      val numLevelsStr = if (isCategorical) s"${new java.lang.Integer(v.domain.length)}" else ""

      val s = String.format(format, CStr, typeStr, minStr, maxStr, meanStr, sigmaStr, naStr, isConstantStr, numLevelsStr)
      logger.info(s)
      val infoData=parseInfo(CStr, typeStr, minStr, maxStr, meanStr, sigmaStr, naStr, isConstantStr, numLevelsStr)
      parseBuffer += infoData
    }

    val infoDF=spark.sparkContext.parallelize(parseBuffer).toDS()
//    infoDF.coalesce(1).write.option("header",true).mode(SaveMode.Overwrite).csv("file:///home/bdp/h2o/sichuan/parse_dt_info")

    logger.info(FrameUtils.chunkSummary(fr).toString)
  }

  def getVarImp(model:Model[_,_,_]): String = {
    val sb=new StringBuilder
    val output=model._output
    for (f <- Weaver.getWovenFields(output.getClass))
    {
      val c = f.getType
      if (c.isAssignableFrom(classOf[TwoDimTable]))
        try {
          val t = f.get(output).asInstanceOf[TwoDimTable]
          f.setAccessible(true)
          if (t != null)
          {sb.append(t.toString(1, true /*don't print the full table if too long*/))
          }
        } catch {
          case e: IllegalAccessException =>
            e.printStackTrace()
        }
    }
    sb.toString()
  }
}

