import pyspark
import pyspark.sql.functions as psf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType
from textblob import TextBlob

output_path = 'agg/'

spark = SparkSession \
    .builder \
    .appName("Spark Tweet") \
    .getOrCreate()

@psf.udf(FloatType())
def udf_get_polarity(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    return sentiment.polarity
    
@psf.udf(StringType())
def udf_polarity_label(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    if sentiment.polarity > 0: 
        return 'positive'
    elif sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'
    
@psf.udf(FloatType())
def udf_get_subj(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    return sentiment.subjectivity
    
@psf.udf(StringType())
def udf_subj_label(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    if sentiment.polarity > 0.5: 
        return 'subjective'
    elif sentiment.polarity == 0.5: 
        return 'neutral'
    else: 
        return 'objective'

# The input csv file is one with cleaned tweets. 
# Saves aggregated data on sentiments
# Saves aggregated data on device types

def source_sentiment(csv_file):
    clean_df = spark.read.csv(csv_file, header=True, sep=',')
    
    polarity_df = clean_df.withColumn("polarity", udf_get_polarity(clean_df.Tweet))
    polarity_df.toPandas().to_csv(output_path+'polarity_spark.csv', index=False)

    source = clean_df.groupBy('Source').count()
    source = source.withColumn('percent', psf.col('count')*100/psf.sum('count').over(Window.partitionBy()))
    source.toPandas().to_csv(output_path+'device_spark.csv', index=False)

    pol_df = clean_df.withColumn("polarity_label", udf_polarity_label(clean_df.Tweet))
    pol_label = pol_df.groupBy('polarity_label').count()
    pol_label = pol_label.withColumn('percent', psf.col('count')*100/psf.sum('count').over(Window.partitionBy()))
    pol_label.toPandas().to_csv(output_path+'polarity_label.csv', index=False)

    subj_df = clean_df.withColumn("subj_label", udf_subj_label(clean_df.Tweet))
    subj_label = subj_df.groupBy('subj_label').count()
    subj_label = subj_label.withColumn('percent', psf.col('count')*100/psf.sum('count').over(Window.partitionBy()))
    subj_label.toPandas().to_csv(output_path+'subj_label.csv', index=False)

    subj = clean_df.withColumn("subjectivity", udf_get_subj(clean_df.Tweet))
    subj.toPandas().to_csv(output_path+'subjectivity.csv', index=False)
