package ai.h2o.automl

import _root_.hex.Model
import _root_.hex.ensemble.{Metalearner, StackedEnsemble, StackedEnsembleModel}
import _root_.hex.ensemble.StackedEnsembleModel.StackedEnsembleParameters
import _root_.hex.naivebayes.NaiveBayesModel.NaiveBayesParameters
import _root_.hex.tree.drf.DRF
import _root_.hex.tree.drf.DRFModel.DRFParameters
import org.apache.spark.h2o.H2OFrame
import _root_.hex.tree.gbm.{GBM, GBMModel}
import _root_.hex.tree.gbm.GBMModel.GBMParameters
import _root_.hex.tree.xgboost.{XGBoost, XGBoostModel}
import _root_.hex.tree.xgboost.XGBoostModel.XGBoostParameters
import _root_.hex.tree.xgboost.XGBoostModel.XGBoostParameters._
import _root_.hex.naivebayes.NaiveBayes
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.h2o.{H2OConf, H2OContext}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import water.support.{H2OFrameSupport, ModelSerializationSupport}
import water.{Key, Weaver}
import water.fvec.Frame
import water.util.TwoDimTable


object my_h2o_single {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark: SparkSession =SparkSession.builder()
        .appName("H2OModelBuilding")
        .getOrCreate()
    val conf=new H2OConf(spark)
    val h2oContext: H2OContext =H2OContext.getOrCreate(spark,conf)
    import h2oContext.implicits._
    val h2oModel=new my_h2o_single(spark,h2oContext)
    val (modelData,submitData)=h2oModel.getModelData

    val (trainFrame,testFrame)=h2oModel.splitDataframe(modelData)

    val GBM_01=h2oModel.GBMModel_01(trainFrame,testFrame,"GBM_01")
    val XGB_01=h2oModel.xgboostModel_01(trainFrame,testFrame,"XGB_01")
    val GBM_02=h2oModel.GBMModel_02(trainFrame,testFrame,"GBM_02")
    val BYS=h2oModel.bysModel(trainFrame,testFrame,"BYS")
    val rf=h2oModel.rfModel(trainFrame,testFrame,"rf")
    val XGB_02=h2oModel.xgboostModel_02(trainFrame,testFrame,"XGB_02")
    val stactModel=h2oModel.stackEnsembleModel(trainFrame,testFrame,Array(GBM_01,XGB_01,GBM_02,BYS,rf,XGB_02))

    val submitFr=h2oContext.asH2OFrame(submitData)
    val resultFr=stactModel.score(submitFr)
    submitFr.add(resultFr)
    submitFr.update()
    val result=h2oContext.asDataFrame(submitFr)
    result.groupBy("predict").count().show()

    h2oContext.stop()
    spark.stop()
  }

}

class my_h2o_single(@transient  val spark:SparkSession,
                    @transient val h2oContext:H2OContext)
    extends H2OFrameSupport with ModelSerializationSupport with java.io.Serializable {
  @transient val logger: Logger =LogManager.getLogger(this.getClass)
  import spark.implicits._
  import h2oContext.implicits._

  val modelSavaPath="file:///home/bdp/h2o/sichuan/stactModel/"

  def getModelData: (Dataset[Row], Dataset[Row]) ={
    val allData=spark.read.parquet("file:///home/bdp/h2o/sichuan/allData")
//    val allData=spark.read.option("header","true").option("inferSchema","true").csv("E:\\zte\\data\\data.csv")
    val featureData=allData.select("user",modelCols :+ "label":_*)

    val submitData=featureData.filter("label is null")
    logger.info(s"提交行数为：${submitData.count()}")
    submitData.write.mode(SaveMode.Overwrite).parquet("file:///home/bdp/h2o/sichuan/submitData")

    val modelData=featureData.filter("label is not null")
//    modelData.printSchema()
    logger.info(s"样本总数为：${modelData.count()}")

    (modelData,submitData)
  }

  def GBMModel_01(train: H2OFrame, test: H2OFrame,modelName:String): GBMModel = {
    logger.info("GBMModel_01开始构建====================================")
    val gbmParams = new GBMParameters()
    gbmParams._train = train._key
    gbmParams._valid = test._key
    gbmParams._response_column = "label"
    gbmParams._ignored_columns=Array("user")
    gbmParams._ntrees = 120
    gbmParams._max_depth = 8
    gbmParams._col_sample_rate_per_tree=0.2
    gbmParams._sample_rate=0.7
    gbmParams._learn_rate_annealing=1.0
    gbmParams._keep_cross_validation_models=true
    gbmParams._keep_cross_validation_predictions=true
    gbmParams._nfolds=5
    gbmParams._seed=67829L
    val gbm = new GBM(gbmParams)
    val model = gbm.trainModel().get()
    logger.info("GBMModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("GBMModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }

  def GBMModel_02(train: H2OFrame, test: H2OFrame,modelName:String): GBMModel = {
    logger.info("GBMModel_02开始构建====================================")
    val gbmParams = new GBMParameters()
    gbmParams._train = train._key
    gbmParams._valid = test._key
    gbmParams._response_column = "label"
    gbmParams._ignored_columns=Array("user")
    gbmParams._ntrees = 120
    gbmParams._max_depth = 7
    gbmParams._col_sample_rate_per_tree=0.2
    gbmParams._sample_rate=0.7
    gbmParams._learn_rate_annealing=1.0
    gbmParams._balance_classes=true
    gbmParams._class_sampling_factors=Array(1.0f,1.0f,0.2f,1.0f,1.0f)
    gbmParams._keep_cross_validation_models=true
    gbmParams._keep_cross_validation_predictions=true
    gbmParams._nfolds=5
    gbmParams._seed=67829L
    val gbm = new GBM(gbmParams)
    val model = gbm.trainModel().get()
    logger.info("GBMModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("GBMModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }

  def xgboostModel_01(train: H2OFrame, test: H2OFrame,modelName:String):XGBoostModel={
    logger.info("xgboostModel_01开始构建====================================")
    val xgbParams=new XGBoostParameters()
    xgbParams._booster=Booster.gbtree
    xgbParams._n_estimators=120
    xgbParams._col_sample_rate_per_tree=0.3
    xgbParams._max_depth=10
    xgbParams._sample_rate=0.7
    xgbParams._learn_rate=0.1
    xgbParams._nthread=10
    xgbParams._train=train._key
    xgbParams._valid = test._key
    xgbParams._response_column = "label"
    xgbParams._ignored_columns=Array("user")
    xgbParams._keep_cross_validation_models=true
    xgbParams._keep_cross_validation_predictions=true
    xgbParams._nfolds=5
    xgbParams._seed=67829L
    xgbParams._backend= Backend.cpu
    val xgb=new XGBoost(xgbParams)
    val model = xgb.trainModel().get()
    logger.info("XGBModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("XGBModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }

  def xgboostModel_02(train: H2OFrame, test: H2OFrame,modelName:String):XGBoostModel={
    logger.info("xgboostModel_02开始构建====================================")
    val xgbParams=new XGBoostParameters()
    xgbParams._booster=Booster.gbtree
    xgbParams._grow_policy= GrowPolicy.lossguide
    xgbParams._n_estimators=150
    xgbParams._col_sample_rate_per_tree=0.3
    xgbParams._max_depth=7
    xgbParams._sample_rate=0.7
    xgbParams._learn_rate=0.1
    xgbParams._nthread=10
    xgbParams._train=train._key
    xgbParams._valid = test._key
    xgbParams._response_column = "label"
    xgbParams._ignored_columns=Array("user")
    xgbParams._balance_classes=true
    xgbParams._class_sampling_factors=Array(1.0f,1.0f,0.2f,1.0f,1.0f)
    xgbParams._keep_cross_validation_models=true
    xgbParams._keep_cross_validation_predictions=true
    xgbParams._nfolds=5
    xgbParams._seed=67829L
    xgbParams._backend= Backend.cpu
    val xgb=new XGBoost(xgbParams)
    val model = xgb.trainModel().get()
    logger.info("XGBModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("XGBModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }


  def rfModel(train: H2OFrame, test: H2OFrame,modelName:String)={
    logger.info("rfModel开始构建====================================")
    val rfParams=new DRFParameters()
    rfParams._train=train._key
    rfParams._valid = test._key
    rfParams._response_column = "label"
    rfParams._ignored_columns=Array("user")
    rfParams._ntrees=200
    rfParams._max_depth=15
    rfParams._col_sample_rate_per_tree=0.3
    rfParams._balance_classes=true
    rfParams._class_sampling_factors=Array(1.0f,1.0f,0.2f,1.0f,1.0f)
    rfParams._keep_cross_validation_models=true
    rfParams._keep_cross_validation_predictions=true
    rfParams._nfolds=5
    rfParams._seed=67829L
    val rf=new DRF(rfParams)
    val model = rf.trainModel().get()
    logger.info("DRFModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("DRFModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }


  def bysModel(train: H2OFrame, test: H2OFrame,modelName:String)={
    logger.info("bysModel开始构建====================================")
    val bysParams=new NaiveBayesParameters()
    bysParams._train=train._key
    bysParams._valid=test._key
    bysParams._laplace=0.1

    bysParams._ignored_columns=Array("user")
    bysParams._response_column="label"
    bysParams._keep_cross_validation_models=true
    bysParams._keep_cross_validation_predictions=true
    bysParams._nfolds=5
    bysParams._seed=67829L
    val bys=new NaiveBayes(bysParams)
    val model=bys.trainModel().get()
    logger.info("BYSModel Summary:=====================================================")
    logger.info(model.toString)
    logger.info("BYSModel VarImp:======================================================")
    val varimp=getVarImp(model)
    if(varimp.nonEmpty)logger.info(varimp)
    else {logger.info(s"${model._key.toString}未返回字段重要性排序情况！")}
    model.exportBinaryModel(modelSavaPath + modelName,true)
    model
  }


  def stackEnsembleModel(train: H2OFrame, test: H2OFrame,models:Array[Model[_,_,_]]) ={
    logger.info("stackEnsembleModel开始构建====================================")
    val metaParams=new GBMParameters()
    metaParams._ntrees=50
    metaParams._max_depth=4
    metaParams._learn_rate_annealing=0.99

    val stackParams=new StackedEnsembleParameters()
    stackParams._base_models= models.map(m=> m._key.asInstanceOf[T_MODEL_KEY])
    stackParams._metalearner_algorithm= Metalearner.Algorithm.gbm
    stackParams._metalearner_parameters=metaParams
    stackParams._train=train._key
    stackParams._valid=test._key
    stackParams._response_column="label"
    stackParams._ignored_columns=Array("user")

    val stackedEnsembleJob=new StackedEnsemble(stackParams)
    val model=stackedEnsembleJob.trainModel().get()
    logger.info("BYSModel Summary:=====================================================")
    logger.info(model.toString)
    model.exportBinaryModel(modelSavaPath + "stactModel.bin",true)
//    model.exportMojo(modelSavaPath+"stactModel.mojo",true)
    model
  }

  def splitDataframe(df:DataFrame): (Frame, Frame) ={
    val h2oframe=h2oContext.asH2OFrame(df)
    val data=withLockAndUpdate(h2oframe){StringToC}
    val h2oModelData=withLockAndUpdate(data){
      colsToC(_,
        Array("gender","star_level","last_year_capture_user_flag",
          "market_price_level","cust_point_level","credit_level","dt_m_1012_type",
          "dt_m_1027_type","dt_m_1032_type","dt_m_1034_type","dt_m_1075_type","dt_m_1086_type","dt_m_1087_type",
          "dt_m_1096_type","dt_m_1102_type","dt_m_1108_type","dt_m_1594_type","dt_m_1617_type","dt_m_1620_type",
          "dt_m_1630_type","dt_m_1633_type","app1_visits_type","app2_visits_type","app3_visits_type","dt_m_1035_pref",
          "app4_visits_type","app5_visits_type","app6_visits_type","app7_visits_type","app8_visits_type","dt_m_1087_pref"))}

//    logger.info("========================================================================================================")
//    parseResults(h2oModelData)
//    logger.info("========================================================================================================")
        val frs = splitFrame(h2oModelData, Array("train.hex", "test.hex"), Array(0.8, 0.2))
//    val (train,test)=(frs(0), frs(1))
        (frs(0), frs(1))
  }


  type T_MODEL_KEY = Key[Model[o,p,q] forSome {
    type o <: hex.Model[o, p, q]
    type p <: hex.Model.Parameters
    type q <: hex.Model.Output}]

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


  def StringToC[T <: Frame](fr: T): T = {
    fr.vecs().indices
        .filter(idx => fr.vec(idx).isString)
        .foreach(idx => fr.replace(idx, fr.vec(idx).toCategoricalVec).remove()
        )
    fr
  }

  def colsToC[T <: Frame](fr: T, colNames: Array[String]): T  = {
    colNames.map(fr.names().indexOf(_))
        .foreach(idx => fr.replace(idx, fr.vec(idx).toCategoricalVec).remove())
    fr
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
