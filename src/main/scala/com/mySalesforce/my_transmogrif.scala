package com.mySalesforce

import com.salesforce.op.features.FeatureBuilder
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{MultiClassificationModelSelector, OpLogisticRegression, OpNaiveBayes, OpRandomForestClassifier, OpXGBoostClassifier}

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.tuning.ParamGridBuilder


object my_transmogrif {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    implicit val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._

    val source=spark.read.parquet("/home/bdp/h2o/sichuan/wideTable")
//    val source=spark.read.parquet(args(0))
    val dataProcesser=new my_transmogrif()

    val data=dataProcesser.dataGet(source)

    val label = FeatureBuilder.Text[dataSchema].extract(_.label.toText).asResponse
    val features=dataProcesser.fearureEngineer()


    val indexed_label=label.indexed()
    val checkedFeatures = indexed_label.sanityCheck(features, removeBadFeatures = true)

    val models=dataProcesser.models()
    val cutter = DataCutter(reserveTestFraction = 0.2, seed = 42L)

    val prediction = MultiClassificationModelSelector
        .withCrossValidation(splitter = Option(cutter), seed = 42L, modelsAndParameters = models)
        .setInput(indexed_label, checkedFeatures)
        .getOutput()

    val evaluator = Evaluators.MultiClassification.f1().setLabelCol(indexed_label).setPredictionCol(prediction)
    val workflow = new OpWorkflow().setResultFeatures(prediction, indexed_label).setInputDataset(data)
    val model = workflow.train()
    println(s"Model summary:\n${model.summaryPretty()}")

    println("Scoring the model")
    val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)
    println("Metrics:\n" + metrics)
    scores.show(false)

    spark.stop()

  }

}

class my_transmogrif(@transient implicit val spark:SparkSession) extends java.io.Serializable {
  import spark.implicits._

  def dataGet(data:DataFrame)={
    val cust_point_level=for(i <- 1.0 to(20001.0,500.0)) yield i
    val cust_point_buck=0.0 +: cust_point_level :+ Double.PositiveInfinity
    val cust_point_bucket=new Bucketizer().setSplits(cust_point_buck.toArray).setInputCol("cust_point").setOutputCol("cust_point_level").setHandleInvalid("keep")
    val apps=Array("app1_visits","app2_visits","app3_visits","app4_visits","app5_visits","app6_visits",
      "app7_visits","app8_visits")
    val df=cust_point_bucket.transform(data)
        .drop("age","market_price","cust_point")
        .withColumn("app_cnt",apps.map(c=>when(col(c)>0,1).otherwise(0)).reduce(_ + _))
        .withColumn("dt_m_1069",$"dt_m_1068"-$"dt_m_1067").drop("dt_m_1068")
        .withColumn("dt_m_1044",$"dt_m_1043"-$"dt_m_1041").drop("dt_m_1043")
        .withColumn("dt_m_1053",$"dt_m_1052"-$"dt_m_1051").drop("dt_m_1052")
        .as[dataSchema]

    val submitData=df.filter("label is null")
    submitData.write.mode(SaveMode.Overwrite).parquet("file:///home/bdp/h2o/sichuan/submitData")

    val modelData=df.filter("label is not null")
    modelData.printSchema()

    modelData
  }


  val pickList=Array("credit_level","gender","age_level","market_price_level", "cust_point_level")
  val realCols=Array("dt_m_1004","dt_m_1032","dt_m_1034","dt_m_1035","dt_m_1085","dt_m_1086","dt_m_1087",
    "dt_m_1618","dt_m_1620","access_net_dur","tmlRegister_dur","prdOpen_dur","tmlRegister_net_dur","prdOpen_net_dur")
  val intCols_01=Array("dt_m_1000","dt_m_1003","dt_m_1005","dt_m_1006",
    "dt_m_1009","dt_m_1011","dt_m_1012","dt_m_1015","dt_m_1017","dt_m_1027","dt_m_1028","dt_m_1041",
    "dt_m_1051","dt_m_1067","dt_m_1073")
  val intCols_02=Array("dt_m_1074","dt_m_1075","dt_m_1096","dt_m_1099","dt_m_1102",
    "dt_m_1105","dt_m_1108","dt_m_1111","dt_m_1594","dt_m_1601","dt_m_1617","dt_m_1630","dt_m_1633",
    "dt_m_1069","dt_m_1044","dt_m_1053")
  val intCols_03=Array("inet_pd_inst_cnt","star_level","app1_visits","app2_visits","app3_visits","app4_visits","app5_visits","app6_visits","app7_visits",
    "app8_visits","last_year_capture_user_flag","app_cnt")


  def models()={
    val lr = new OpLogisticRegression()
    val rf = new OpRandomForestClassifier()
    val nb = new OpNaiveBayes()
    val xgb = new OpXGBoostClassifier()

    val models = Seq(
      lr -> new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.05, 0.1))
          .addGrid(lr.elasticNetParam, Array(0.01))
          .addGrid(lr.maxIter, Array(50,100,80))
          .build(),
      rf -> new ParamGridBuilder()
          .addGrid(rf.maxDepth, Array(7,10,8))
          .addGrid(rf.minInstancesPerNode, Array(30, 50, 40))
          .addGrid(rf.seed, Array(42L))
          .addGrid(rf.numTrees,Array(100,300,200))
          .build(),
      nb -> new ParamGridBuilder()
          .addGrid(nb.smoothing, Array(1.0))
          .build(),
      xgb -> new ParamGridBuilder()
          .addGrid(xgb.numRound, Array(100))
          .addGrid(xgb.eta, Array(0.1 , 0.3))
          .addGrid(xgb.maxDepth, Array(7,10,8))
          .addGrid(xgb.minChildWeight, Array(1.0, 5.0, 10.0))
          .build()
    )

    models
  }


  def fearureEngineer()={
    val brand = FeatureBuilder.PickList[dataSchema].extract(_.brand.toPickList).asPredictor
    val product = FeatureBuilder.PickList[dataSchema].extract(_.product.toPickList).asPredictor
    val Array(credit_level,gender,age_level,market_price_level,cust_point_level) =
      pickList.map(c=>FeatureBuilder.PickList[dataSchema].extract(getValue(_,c).map(_.asInstanceOf[Int].toString).toPickList).asPredictor)
    val Array(dt_m_1004,dt_m_1032,dt_m_1034,dt_m_1035,dt_m_1085,dt_m_1086,dt_m_1087,dt_m_1618,dt_m_1620,
    access_net_dur,tmlRegister_dur,prdOpen_dur,tmlRegister_net_dur,prdOpen_net_dur)=
      realCols.map(c=>FeatureBuilder.Real[dataSchema].extract(getValue(_,c).map(_.asInstanceOf[Double]).toReal).asPredictor)

    val Array(dt_m_1000,dt_m_1003,dt_m_1005,dt_m_1006,
    dt_m_1009,dt_m_1011,dt_m_1012,dt_m_1015,dt_m_1017,dt_m_1027,dt_m_1028,dt_m_1041,
    dt_m_1051,dt_m_1067,dt_m_1073)=
      intCols_01.map(c=> FeatureBuilder.Integral[dataSchema].extract(getValue(_,c).map(_.asInstanceOf[Int]).toIntegral).asPredictor)

    val Array(dt_m_1074,dt_m_1075,dt_m_1096,dt_m_1099,dt_m_1102,
    dt_m_1105,dt_m_1108,dt_m_1111,dt_m_1594,dt_m_1601,dt_m_1617,dt_m_1630,dt_m_1633,
    dt_m_1069,dt_m_1044,dt_m_1053)=
      intCols_02.map(c=> FeatureBuilder.Integral[dataSchema].extract(getValue(_,c).map(_.asInstanceOf[Int]).toIntegral).asPredictor)

    val Array(inet_pd_inst_cnt,star_level,app1_visits,app2_visits,app3_visits,app4_visits,app5_visits,app6_visits,app7_visits,
    app8_visits,last_year_capture_user_flag,app_cnt)=
      intCols_03.map(c=> FeatureBuilder.Integral[dataSchema].extract(getValue(_,c).map(_.asInstanceOf[Int]).toIntegral).asPredictor)


    val pivotGender=gender.pivot()
    val dt_m_1085_v=dt_m_1085 * dt_m_1006
    val dt_m_1086_v=dt_m_1086 * dt_m_1006
    val dt_m_1087_v=dt_m_1087 * dt_m_1006


    val features=Seq(inet_pd_inst_cnt,star_level,dt_m_1000,dt_m_1003,dt_m_1005,dt_m_1006,dt_m_1009,dt_m_1011,
      dt_m_1012,dt_m_1015,dt_m_1017,dt_m_1027,dt_m_1028,dt_m_1041,dt_m_1051,dt_m_1067,dt_m_1073,dt_m_1074,
      dt_m_1075,dt_m_1096,dt_m_1099,dt_m_1102,dt_m_1105,dt_m_1108,dt_m_1111,dt_m_1594,dt_m_1601,dt_m_1617,
      dt_m_1630,dt_m_1633,app1_visits,app2_visits,app3_visits,app4_visits,app5_visits,app6_visits,app7_visits,
      app8_visits,last_year_capture_user_flag,app_cnt,dt_m_1069,dt_m_1044,dt_m_1053,dt_m_1004,dt_m_1032,dt_m_1034,
      dt_m_1035,dt_m_1085_v,dt_m_1086_v,dt_m_1087_v,dt_m_1618,dt_m_1620, access_net_dur,tmlRegister_dur,
      prdOpen_dur,tmlRegister_net_dur,prdOpen_net_dur,credit_level,pivotGender,age_level,market_price_level,
      cust_point_level,brand,product).transmogrify()
    features
  }


  def getValue(data:dataSchema,field:String)={
    field match {
      case "credit_level"=>data.credit_level
      case "gender"=>data.gender
      case "age_level"=>data.age_level
      case "market_price_level"=>data.market_price_level
      case "cust_point_level"=>data.cust_point_level
      case "brand"=>data.brand
      case "product"=>data.product
      case "inet_pd_inst_cnt"=>data.inet_pd_inst_cnt
      case "star_level"=>data.star_level
      case "dt_m_1000"=>data.dt_m_1000
      case "dt_m_1003"=>data.dt_m_1003
      case "dt_m_1005"=>data.dt_m_1005
      case "dt_m_1006"=>data.dt_m_1006
      case "dt_m_1009"=>data.dt_m_1009
      case "dt_m_1011"=>data.dt_m_1011
      case "dt_m_1012"=>data.dt_m_1012
      case "dt_m_1015"=>data.dt_m_1015
      case "dt_m_1017"=>data.dt_m_1017
      case "dt_m_1027"=>data.dt_m_1027
      case "dt_m_1028"=>data.dt_m_1028
      case "dt_m_1041"=>data.dt_m_1041
      case "dt_m_1051"=>data.dt_m_1051
      case "dt_m_1067"=>data.dt_m_1067
      case "dt_m_1073"=>data.dt_m_1073
      case "dt_m_1074"=>data.dt_m_1074
      case "dt_m_1075"=>data.dt_m_1075
      case "dt_m_1096"=>data.dt_m_1096
      case "dt_m_1099"=>data.dt_m_1099
      case "dt_m_1102"=>data.dt_m_1102
      case "dt_m_1105"=>data.dt_m_1105
      case "dt_m_1108"=>data.dt_m_1108
      case "dt_m_1111"=>data.dt_m_1111
      case "dt_m_1594"=>data.dt_m_1594
      case "dt_m_1601"=>data.dt_m_1601
      case "dt_m_1617"=>data.dt_m_1617
      case "dt_m_1630"=>data.dt_m_1630
      case "dt_m_1633"=>data.dt_m_1633
      case "app1_visits"=>data.app1_visits
      case "app2_visits"=>data.app2_visits
      case "app3_visits"=>data.app3_visits
      case "app4_visits"=>data.app4_visits
      case "app5_visits"=>data.app5_visits
      case "app6_visits"=>data.app6_visits
      case "app7_visits"=>data.app7_visits
      case "app8_visits"=>data.app8_visits
      case "last_year_capture_user_flag"=>data.last_year_capture_user_flag
      case "app_cnt"=>data.app_cnt
      case "dt_m_1069"=>data.dt_m_1069
      case "dt_m_1044"=>data.dt_m_1044
      case "dt_m_1053"=>data.dt_m_1053
      case "dt_m_1004"=>data.dt_m_1004
      case "dt_m_1032"=>data.dt_m_1032
      case "dt_m_1034"=>data.dt_m_1034
      case "dt_m_1035"=>data.dt_m_1035
      case "dt_m_1085"=>data.dt_m_1085
      case "dt_m_1086"=>data.dt_m_1086
      case "dt_m_1087"=>data.dt_m_1087
      case "dt_m_1618"=>data.dt_m_1618
      case "dt_m_1620"=>data.dt_m_1620
      case "access_net_dur"=>data.access_net_dur
      case "tmlRegister_dur"=>data.tmlRegister_dur
      case "prdOpen_dur"=>data.prdOpen_dur
      case "tmlRegister_net_dur"=>data.tmlRegister_net_dur
      case "prdOpen_net_dur"=>data.prdOpen_net_dur
    }}
}



case class dataSchema(user:String,credit_level:Option[Int],membership_level:Option[Int],gender:Option[Int],
                      inet_pd_inst_cnt:Option[Int],star_level:Option[Int],dt_m_1000:Option[Int],
                      dt_m_1003:Option[Int],dt_m_1004:Option[Double],dt_m_1005:Option[Int],
                      dt_m_1006:Option[Int],dt_m_1009:Option[Int],dt_m_1011:Option[Int],
                      dt_m_1012:Option[Int],dt_m_1015:Option[Int],dt_m_1017:Option[Int],
                      dt_m_1027:Option[Int],dt_m_1028:Option[Int],dt_m_1032:Option[Double],
                      dt_m_1034:Option[Double],dt_m_1035:Option[Double],dt_m_1041:Option[Int],
                      dt_m_1051:Option[Int],dt_m_1067:Option[Int],dt_m_1073:Option[Int],
                      dt_m_1074:Option[Int],dt_m_1075:Option[Int],dt_m_1085:Option[Double],
                      dt_m_1086:Option[Double],dt_m_1087:Option[Double],dt_m_1096:Option[Int],
                      dt_m_1099:Option[Int],dt_m_1102:Option[Int],dt_m_1105:Option[Int],
                      dt_m_1108:Option[Int],dt_m_1111:Option[Int],dt_m_1594:Option[Int],
                      dt_m_1601:Option[Int],dt_m_1617:Option[Int],dt_m_1618:Option[Long],
                      dt_m_1620:Option[Double],dt_m_1630:Option[Int],dt_m_1633:Option[Int],
                      app1_visits:Option[Int],app2_visits:Option[Int],app3_visits:Option[Int],
                      app4_visits:Option[Int],app5_visits:Option[Int],app6_visits:Option[Int],
                      app7_visits:Option[Int],app8_visits:Option[Int],
                      last_year_capture_user_flag:Option[Int],label:String,access_net_dur:Option[Double],
                      tmlRegister_dur:Option[Double],prdOpen_dur:Option[Double],
                      tmlRegister_net_dur:Option[Double],prdOpen_net_dur:Option[Double],
                      brand:Option[String],product:Option[String],age_level:Option[Double],
                      market_price_level:Option[Double],cust_point_level:Option[Double],
                      app_cnt:Option[Int],dt_m_1069:Option[Int],dt_m_1044:Option[Int],dt_m_1053:Option[Int]
                     )

