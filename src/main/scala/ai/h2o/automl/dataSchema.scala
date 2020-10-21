package ai.h2o.automl


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

case class parseInfo(CStr:String,
                     typeStr:String,
                     minStr:String,
                     maxStr:String,
                     meanStr:String,
                     sigmaStr:String,
                     naStr:String,
                     isConstantStr:String,
                     numLevelsStr:String)
