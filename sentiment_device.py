import pyspark
import pyspark.sql.functions as psf
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from textblob import TextBlob

output_path = 'agg/'

spark = SparkSession \
    .builder \
    .appName("Spark Tweet") \
    .getOrCreate()

@psf.udf(StringType())
def udf_get_sentiment(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    if sentiment.polarity > 0: 
        return 'positive'
    elif sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

# The input csv file is one with cleaned tweets. 
# Saves aggregated data on sentiments
# Saves aggregated data on device types
def source_sentiment(csv_file):
    clean_df = spark.read.csv(csv_file, header=True, sep=',')
    clean_df2 = clean_df.withColumn("Sentiment", udf_get_sentiment(clean_df.Tweet))
    sentiment = clean_df2.groupBy('Sentiment').count()
    sentiment.toPandas().to_csv(output_path+'sentiment_spark.csv', index=False)
    source = clean_df2.groupBy('Source').count()
    source.toPandas().to_csv(output_path+'device_spark.csv', index=False)