#!/home/ulrich/anaconda3/envs/bigdata/bin/python
import credentials # Import api/access_token keys from credentials.py
import settings # Import database and search related setting constants from settings.py 
import re
import tweepy
import mysql.connector
import psycopg2
import pandas as pd
from textblob import TextBlob
import ssl
import time
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError
# Streaming With Tweepy 
# http://docs.tweepy.org/en/v3.4.0/streaming_how_to.html#streaming-with-tweepy


# This is Main function.
# Extracting streaming data from Twitter, pre-processing, and loading into MySQL
# Override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    '''
    Tweets are known as “status updates”. So the Status class in tweepy has properties describing the tweet.
    https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
    '''
    
    def on_status(self, status):
        '''
        Extract info from tweets
        '''
        
        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        #verified = 1 if status.verified else 0
        id_str = status.id_str
        created_at = status.created_at
        text = deEmojify(status.text)    # Pre-processing the text  
        sentiment = TextBlob(text).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        verified = status.user.verified
        user_created_at = status.user.created_at
        user_location = deEmojify(status.user.location)
        user_description = deEmojify(status.user.description)
        user_followers_count =status.user.followers_count
        retweet_count = status.retweet_count
        favorite_count = status.favorite_count
        if 'android' in status.source.lower():
            source = 'Android'
        elif 'iphone' in status.source.lower():
            source = 'iPhone'
        elif 'web' in status.source.lower():
            source = 'Web App'
        else:
            source = 'Bot'

        longitude = None
        latitude = None
        if status.coordinates:
            longitude = status.coordinates['coordinates'][0]
            latitude = status.coordinates['coordinates'][1]
        
        print(status.text)
        #print("Long: {}, Lati: {}".format(longitude, latitude))
        
        # Store all data in PostgreSQL
        if connection:
            cursor = connection.cursor()
            query = """
            INSERT INTO {} (id_str, 
                            created_at, 
                            text, 
                            source,
                            polarity, 
                            subjectivity, 
                            user_created_at, 
                            user_location, 
                            user_description, 
                            user_followers_count, 
                            longitude, 
                            latitude, 
                            retweet_count, 
                            favorite_count,
                            verified_user) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)
            """.format(settings.TABLE_NAME)
            values = (id_str, created_at, text,source, polarity, subjectivity, user_created_at, user_location, \
                      user_description, user_followers_count, longitude, latitude, retweet_count, favorite_count,verified)
            cursor.execute(query, values)
            connection.commit()
            cursor.close()
    
    
    def on_error(self, status_code):
        '''
        Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.
        '''
        if status_code == 420:
            # return False to disconnect the stream
            return False
def clean_tweet(self, tweet): 
    ''' 
    Use sumple regex statemnents to clean tweet text by removing links and special characters
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                |(\w+:\/\/\S+)", " ", tweet).split()) 
def deEmojify(text):
    '''
    Strip all non-ASCII characters to remove emoji characters
    '''
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None

if __name__ == '__main__':
    #DATABASE_URL = os.environ['DATABASE_URL']
    #connection = psycopg2.connect(DATABASE_URL, sslmode='require')
    connection = psycopg2.connect(user='xxxx',
                            password='xxxx',
                            host='127.0.0.1', 
                            port='5432', 
                            database='twitter')
    cur = connection.cursor()

    #Check if the table of interest exists or not
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (settings.TABLE_NAME,))

    #Create table if not existing and close the cursor.
    if not cur.fetchone()[0]: 
        try:
            cur.execute("CREATE TABLE {} ({});".format(settings.TABLE_NAME, settings.TABLE_ATTRIBUTES))
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            print('Error while connecting to PostgreSQL', error)
        finally:
            cur.close()
    else:
    #Simple close the cursor() if the table already exists.
        print("Table {} exitsts".format(settings.TABLE_NAME))
        cur.close()

    auth  = tweepy.OAuthHandler(credentials.API_KEY, credentials.API_SECRET_KEY)
    auth.set_access_token(credentials.ACCESS_TOEKN, credentials.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    myStreamListener = MyStreamListener()

    # Initializing the twitter streap Stream
    try:
        myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
        #myStream.filter(languages=["en","fr"], track = settings.TRACK_WORDS)
        myStream.filter(track = settings.TRACK_WORDS)

    # Stop temporarily when hitting Twitter rate Limit
    except tweepy.RateLimitError:
        print("RateLimitError...waiting ~15 minutes to continue")
        time.sleep(1001)
        myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
        myStream.filter(track = settings.TRACK_WORDS)

    # Stop temporarily when getting a timeout or connection error
    except (Timeout, ssl.SSLError, ReadTimeoutError,
            ConnectionError) as exc:
        print("Timeout/connection error...waiting ~15 minutes to continue")
        time.sleep(1001)
        myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
        myStream.filter(track = settings.TRACK_WORDS)

    # Stop temporarily when getting other errors
    except tweepy.TweepError as e:
        if 'Failed to send request:' in e.reason:
            print("Time out error caught.")
            time.sleep(1001)
            myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
            myStream.filter(track = settings.TRACK_WORDS)
        else:
            print("Other error with this user...passing")
            pass
    finally:
        # Close the PostgreSQL connection as it finished
        # However, this won't be reached as the stream listener won't stop automatically
        # Press STOP button to finish the process.
        connection.close() 
