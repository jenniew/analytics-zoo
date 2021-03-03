package com.intel.analytics.zoo.examples.dien

import scala.util.Random
import scala.util.control._
import com.intel.analytics.zoo.common.NNContext.initNNContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import collection.mutable.WrappedArray
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.ArrayType

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object Preprocess {

  case class GroupedIndex(asin_index: Int, cat_index: Int, unixReviewTime: Int, asin_history: Array[Int], cat_history: Array[Int])

  def read_json(spark: SparkSession, path: String, cols: Array[String]=null): DataFrame = {

    var df = spark.read.json(path)
    if (cols != null)
      df = df.select(cols.map(name => col(name)):_*)
    df

  }

//
  def addNeg(df: DataFrame, itemSize: Int, userID: String="uid", itemID:String="item_id",
             label:String="label", negNum:Int=1): DataFrame = {
    val sqlContext = df.sqlContext
    import sqlContext.implicits._
    val combinedDF = df.rdd.flatMap(row => {
      val result = new Array[Tuple3[Int, Int, Int]](negNum + 1)
      val r = new Random()
      for (i <- 0 until negNum) {
        var neg = 0
        do {
          neg = r.nextInt(itemSize)
        } while (neg == row.getAs[Int](itemID))

        result(i) = (row.getAs[Int](userID), neg, 0)
      }
      result(negNum) = (row.getAs[Int](userID), row.getAs(itemID), 1)
      result

    }).toDF(userID, itemID, label)

    combinedDF
  }

  def addNegExcludeClickedItemList(df: DataFrame, itemSize: Int, userItemListDF: DataFrame,
                                   userID: String="uid", itemID:String="item_id", label:String="label",
                                   negNum:Int=1): DataFrame = {
    val sqlContext = df.sqlContext
    import sqlContext.implicits._
    val combinedDF = df.join(userItemListDF, "uid").rdd.flatMap(row => {
      val itemList = row.getAs[Seq[Int]]("itemList")
      val result = new Array[Tuple3[Int, Int, Int]](negNum + 1)
      val r = new Random()
      for (i <- 0 until negNum) {
        var neg = 0
        do {
          neg = r.nextInt(itemSize)
        } while (itemList.contains((neg)))

        result(i) = (row.getAs[Int](userID), neg, 0)
      }
      result(negNum) = (row.getAs[Int](userID), row.getAs(itemID), 1)
      result

    }).toDF(userID, itemID, label)

    combinedDF
  }

  def addNegSamplingForSequence(df: DataFrame, itemSize: Int, userID: String="uid", itemID:String="item_id",
             label:String="label", negNum:Int=1): DataFrame = {
    val sqlContext = df.sqlContext
    val colNames = df.columns
    val restCols = colNames.filter(!_.contains(itemID))
    import sqlContext.implicits._
    val combinedRDD = df.rdd.flatMap(row => {
      val restValues = row.getValuesMap[Any](restCols).values
      val result = new Array[Row](negNum + 1)
      val r = new Random()
      for (i <- 0 until negNum) {
        var neg = 0
        do {
          neg = r.nextInt(itemSize)
        } while (neg == row.getAs[Int](itemID))

        result(i) = Row.fromSeq(restValues.toSeq ++ Array[Any](neg, 0))
      }
      result(negNum) = Row.fromSeq(restValues.toSeq ++ Array[Any](row.getAs(itemID), 1))
      result

    })
    val newSchema = StructType(df.schema.fields.filter(_.name != itemID) ++ Array(
      StructField(itemID, IntegerType, false), StructField(label, IntegerType, false)))

    val combinedDF = sqlContext.createDataFrame(combinedRDD, newSchema)
    combinedDF
  }

  def getItemListForEachUser(df: DataFrame, userID: String="uid", itemID:String="item_id"): DataFrame = {
    val newDF = df.withColumn("item_time", array(col(itemID), col("time"))).drop("time")

    //    spark.udf.register("getItemList", new GetItemList)

    // Create an instance of UDAF GeometricMean.
    val gm = new GetItemList()

    // Show the geometric mean of values of column "id".
    val grouped = newDF.groupBy(userID)
    val itemListDF = grouped.agg(gm(col("item_time")).as("itemList"))
    itemListDF
  }

  class GetItemList extends UserDefinedAggregateFunction {
    // This is the input fields for your aggregate function.
    override def inputSchema: org.apache.spark.sql.types.StructType =
      StructType(StructField("item_time", ArrayType(IntegerType)) :: Nil)

    // This is the internal fields you keep for computing your aggregate.
    override def bufferSchema: StructType = StructType(
      StructField("item_time_list", MapType(IntegerType, IntegerType)) :: Nil
    )

    // This is the output type of your aggregatation function.
    override def dataType: DataType = ArrayType(IntegerType)

    override def deterministic: Boolean = true

    // This is the initial value for your buffer schema.
    override def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer.update(0, Map[Int, Int]())
     }

    // This is how to update your buffer schema given an input.
    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      if (input.isNullAt(0)) return

      val map = buffer.getAs[Map[Int, Int]](0)
      val item_time = input.getSeq[Int](0)

      buffer.update(0, map ++ Map(item_time(1) -> item_time(0)))

    }

    // This is how to merge two objects with the bufferSchema type.
    override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      val map1 = buffer1.getAs[Map[Int, Int]](0)
      val map2 = buffer2.getAs[Map[Int, Int]](0)
      buffer1.update(0, map1 ++ map2)
    }

    // This is where you output the final value, given the final value of your bufferSchema.
    override def evaluate(buffer: Row): Any = {
      buffer.getAs[Map[Int, Int]](0).toSeq.sortBy(_._1).map(_._2)
    }
  }


  def createHistorySeq(df: DataFrame, maxLen: Int=100): DataFrame = {

    //    def itemSeqUdf = {
    //      val func = (itemCollect: Seq[Row]) =>
    //        itemCollect.sortBy(x => x.getAs[Long](1)).map(x => x.getAs[Long](0))
    //      udf(func)
    //    }
    //
    //    def catSeqUdf = {
    //      val func = (catCollect: Seq[Row]) =>
    //        catCollect.sortBy(x => x.getAs[Long](1)).map(x => x.getAs[WrappedArray[Long]](0)(0))
    //      udf(func)
    //    }

    val asinUdf = udf((asin_collect: Seq[Row]) => {
      val full_rows = asin_collect.sortBy(x => x.getAs[Int](2)).toArray

      (0 to full_rows.size - 1).map(x =>
        GroupedIndex(asin_index = full_rows(x).getAs[Int](0),
          cat_index = full_rows(x).getAs[Int](1),
          unixReviewTime = full_rows(x).getAs[Int](2),
          asin_history = if (x <= maxLen) full_rows.slice(0, x).map(row => row.getAs[Int](0))
          else full_rows.slice(x-maxLen, x).map(row => row.getAs[Int](0)),
          cat_history = if (x <= maxLen) full_rows.slice(0, x).map(row => row.getAs[Int](1))
          else full_rows.slice(x-maxLen, x).map(row => row.getAs[Int](1)))).filter(_.asin_history.length > 0)
    })

    df.groupBy("uid")
      .agg(collect_list(struct(col("item_id"), col("cat_id"), col("time"))).as("asin_collect"))
      .withColumn("item_history", asinUdf(col("asin_collect")))
      .withColumn("item_history", explode(col("item_history")))
      .drop("asin_collect")
      .select(col("uid"),
        col("item_history.asin_index").as("item_id"),
        col("item_history.cat_index").as("cat_id"),
        col("item_history.asin_history").as("item_history"),
        col("item_history.cat_history").as("cat_history"))

  }

  def addNegHistorySequence(df: DataFrame, itemSize: Int, itemCategoryMap: Map[Int, Int], negNum:Int=1): DataFrame = {
    val sqlContext = df.sqlContext
    import sqlContext.implicits._
    val combinedRDD = df.rdd.map(row => {
      val item_history = row.getAs[WrappedArray[Int]]("item_history")
      val r = new Random()
      val negItemSeq = Array.ofDim[Int](item_history.length, negNum)
      val negCatSeq = Array.ofDim[Int](item_history.length, negNum)
      for (i <- 0 until item_history.length) {
        for (j <- 0 until negNum) {
          var negItem = 0
          do {
            negItem = r.nextInt(itemSize)
          } while (negItem == item_history(i))
          negItemSeq(i)(j) = negItem
          negCatSeq(i)(j) = itemCategoryMap(negItem)
        }
      }

      val result = Row.fromSeq(row.toSeq ++ Array[Any](negItemSeq, negCatSeq))
      result

    })
    val newSchema = StructType(df.schema.fields ++ Array(
      StructField("noclk_item_list", ArrayType(ArrayType(IntegerType))),
      StructField("noclk_cat_list", ArrayType(ArrayType(IntegerType)))))

    val combinedDF = sqlContext.createDataFrame(combinedRDD, newSchema)
    combinedDF
  }

  def main(args: Array[String]): Unit = {
    val sc = initNNContext("test")
    val spark = SparkSession.builder().getOrCreate()

    // read meta data
    val meta_path = "/home/jwang/git/recommendation_jennie/public_dien/meta_books_5000.json"
    var meta_df = read_json(spark, meta_path, Array("asin", "categories")).na.drop("any", Seq("asin", "categories"))
    val getCategory = udf((categories: WrappedArray[WrappedArray[String]]) => {
      categories(0).last
    })

    meta_df = meta_df.withColumn("category", getCategory(col("categories"))).drop("categories")
    meta_df.show()

    // generate string index of item
    val itemIndexer = new StringIndexer()
      .setInputCol("asin")
      .setOutputCol("item_id")
      .setHandleInvalid("keep")

    val itemModel = itemIndexer.fit(meta_df)

    meta_df = itemModel.transform(meta_df)
    meta_df = meta_df.withColumn("item_id",col("item_id").cast(IntegerType)).drop("asin")

    // generate string index of category
    val categoryIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("cat_id")
    meta_df = categoryIndexer.fit(meta_df).transform(meta_df).drop("category")
      .withColumn("cat_id",col("cat_id").cast(IntegerType))

    val itemCatMap = meta_df.select("item_id", "cat_id").distinct().rdd
      .collect().map(row => (row(0), row(1))).toMap.asInstanceOf[Map[Int, Int]]

    val invalidCatID = meta_df.select("cat_id").distinct().count().toInt

    val itemSize = meta_df.select("item_id").distinct().count().toInt
    println(s"item size is: ${itemSize}")

    // read review data
    val review_path = "/home/jwang/git/recommendation_jennie/public_dien/reviews_10000_lines.json"
    var review_df = read_json(spark, review_path, Array[String]("reviewerID", "asin", "unixReviewTime"))
      .na.drop("any", Seq("reviewerID", "asin", "unixReviewTime"))

    review_df = review_df.withColumn("time",col("unixReviewTime").cast(IntegerType)).drop("unixReviewTime")

    // generate string indexer of user
    val userIndexer = new StringIndexer()
      .setInputCol("reviewerID")
      .setOutputCol("uid")

    review_df = userIndexer.fit(review_df).transform(review_df).drop("reviewerID")
    review_df = review_df.withColumn("uid",col("uid").cast(IntegerType))

    val userSize = review_df.select("uid").distinct().count()
    println(s"user size is: ${userSize}")

    // change item in review df to item id
    review_df = itemModel.transform(review_df).withColumn("item_id",col("item_id").cast(IntegerType)).drop("asin")

    // generate itemlist for each user
    val userItemListDF = getItemListForEachUser(review_df)
    userItemListDF.show()

    // add negative sampling which is different than current item in each record
    val review_df1 = addNeg(review_df, itemSize)
    review_df1.show()

    // add negative sampling excluding itemlist for each user
    val review_df2 = addNegExcludeClickedItemList(review_df, itemSize, userItemListDF, negNum = 5)
    review_df2.show()

    // join review df and meta data
    var review_df3 = review_df.join(meta_df, Seq("item_id"), "inner")
    review_df3 = review_df3.na.fill(invalidCatID, Seq("cat_id"))
    review_df3.show()

    // generate history sequence
    review_df3 = createHistorySeq(review_df3)
    review_df3.show()

    // add negtive sample
    review_df3 = addNegSamplingForSequence(review_df3, itemSize, negNum = 1)
    review_df3.show()

    // add negtive history
    review_df3 = addNegHistorySequence(review_df3, itemSize, itemCatMap, 5)
    review_df3.show()
  }

}
