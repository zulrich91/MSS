TRACK_WORDS = ['chloroquine']
TABLE_NAME = "tweet"
TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at timestamp DEFAULT NULL, text VARCHAR(255), source VARCHAR(255), \
            polarity INT, subjectivity INT, user_created_at VARCHAR(255), user_location VARCHAR(255), \
            user_description VARCHAR(255), user_followers_count INT, longitude DOUBLE PRECISION, latitude DOUBLE PRECISION, \
            retweet_count INT, favorite_count INT, verified_user BOOLEAN DEFAULT FALSE"
