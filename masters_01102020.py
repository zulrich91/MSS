import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import flask
#import dash_dangerously_set_inner_html
#import dash_auth
import json
import plotly.graph_objs as go
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
import base64
import time
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import re, pickle, os
import datetime
import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.corpora import MmCorpus
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
from gensim.test.utils import datapath
from pathlib import Path
import io
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as psf
from pyspark.sql.functions import udf, col, sum,mean
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.types import StringType, StructType
from textblob import TextBlob

spark = SparkSession \
    .builder \
    .appName("Masters Thesis") \
    .getOrCreate()



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
HERE = Path('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/')
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
wordcloud = '/home/ulrich/Documents/dash-plotly/Masters2/dhifli/wordcloud.png' # replace with your own image
locations = pd.read_csv('locations.csv')
sentiment_df = pd.read_csv('sentiment.csv')
verif_df = pd.read_csv('verified_user.csv')
device_df = pd.read_csv('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/device_spark.csv')
polarity_label_df = pd.read_csv("/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/polarity_label.csv")
sub_label_df = pd.read_csv("/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/subj_label.csv")
encoded_image = base64.b64encode(open(wordcloud, 'rb').read())

lda_out_file = 'lda.html'
lda_html_file = 'lda.html'
DATA_PATH = r'./data/'
MODEL_PATH = r'./models/'
#/home/ulrich/Documents/master2/thesis_code/summary_vis/no_dup_sample.csv
ORIG_TWEET_FILE = DATA_PATH + 'no_dup_sample.csv'
#ORIG_TWEET_FILE = DATA_PATH + '2020-03-05.csv'
CLEANED_TWEET_FILE = DATA_PATH+'cleaned_tweets_df.csv'
CLEANED_BIGRAM_TWEET_FILE = DATA_PATH+'cleaned_bigram_tweets_df.csv'
CORPUS_FILE = MODEL_PATH + 'clean_tweets_corpus.mm'
CORPUS_TFIDF_FILE = MODEL_PATH + 'clean_tweets_corpus_tfidf.mm'
DICT_FILE = MODEL_PATH + 'clean_tweets.dict'
LDA_MODEL_FILE = MODEL_PATH + 'tweets_lda.model'
FR_OUTPUT_FILE = DATA_PATH + "FR_DF.csv"
US_OUTPUT_FILE = DATA_PATH + "US_DF.csv"
DAILY_TWEETS_COUNT_FILE = DATA_PATH +"daily_tweets_count.csv"
#Load previously saved itfidf_corpus
corpus = MmCorpus(CORPUS_TFIDF_FILE)
#Load previously saved Dictionary of tokens
loaded_dict = corpora.Dictionary.load(DICT_FILE)

#Load previously saved LDA Model
lda_model = models.ldamodel.LdaModel.load(LDA_MODEL_FILE)

# ngrams or multi-word expressions
NUM_GRAMS = 2
# ----------------------
# LDA model parameters
# ----------------------
# Number of topics
NUM_TOPICS = 10
# Number of training passes
NUM_PASSES = 50
# Document-Topic Density. The lower alpha is, the more likely that
# a document may contain mixture of just a few of the topics.
# Default is 1.0/NUM_TOPICS
ALPHA = 0.001
# Word-Topic Density. The lower eta is, the more likely that
# a topic may contain a mixture of just a few of the words
# Default is 1.0/NUM_TOPICS
ETA = 'auto'
bubble = False
lda_sim_model = None
lda_gen_model = None
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not',
                    'would', 'say', 'could', '_', 'be', 'know', 'good',
                    'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice',
                    'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot',
                    'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right',
                    'line', 'even', 'also', 'may', 'take', 'come'])
additional_stop_words=['hrtechconf','peopleanalytics','hrtech','hr','hrconfes',
                       'hrtechnology','voiceofhr','hrtechadvisor','gen','wait',
                       'next','see','hcm','booth','tech','la','vega','last',
                       'look','technology','work', 'announce','product','new',
                       'team','use','happen','time','take','make','everyone',
                       'anyone','week','day','year','let','go','come','word',
                       'employee','get','people','today','session','need',
                       'meet','help','talk','join','start','awesome','great',
                       'achieve','job','tonight','everyday','room','ready',
                       'one','company','say','well','data','share','love',
                       'want','like','good','business','sure','miss','demo',
                       'live','min','play','always','would','way','almost',
                       'thank','still','many','much','info','wow','play','full',
                       'org','create','leave','back','front','first','may',
                       'tomorrow','yesterday','find','stay','add','conference',
                       'top','stop','expo','hall','detail','row','award','hey',
                       'continue','put','part','whole','some','any','everywhere',
                       'convention','center','forget','congratulation','every',
                       'agenda','gift','card','available','behind','meeting',
                       'best','happen','unlockpotentialpic','half','none',
                       'human', 'resources','truly','win','possible','thanks',
                       'know','check','visit','fun','give','think','forward',
                       'twitter','com','pic','rt','via', 'chloroquine', 'hydrochloroquine', 'https', 'http']

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

colors_list = list(mcolors._colors_full_map.values())

initial_topic_coordinates = pd.DataFrame()
initial_token_table = pd.DataFrame()
all_tweets_df = pd.DataFrame()
cleaned_tweets_df = pd.DataFrame()
polarity_topics_df = pd.DataFrame()
topics_polarity_df = pd.DataFrame()
topics_subj_df = pd.DataFrame()
renamed_topics_polarity_df = pd.DataFrame()
renamed_topics_subj_df = pd.DataFrame()
topics_names_exist = False
FR_DF = pd.DataFrame()
US_DF = pd.DataFrame()

topic_new_name_dict = dict()

fig = dcc.Graph()

tabs_styles = {
    'height': '44px',
    #'width':'800px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'paddingLeft': '70px',
    'fontWeight': 'bold',
    'margin':'10px'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

bar_style = {
    'width':'90%',
    'border':'1px solid',
    'border-radius': 10,
    'backgroundColor':'#FFFFFF',
    'margin':10
}
search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

def topic_rename_f(topic):
    global topic_new_name_dict
    if topic in list(topic_new_name_dict.keys()):
        topic_name = topic_new_name_dict[topic]
    else:
        topic_name = 'Unknown Topic'
    return topic_name

@psf.udf(StringType())
def udf_get_sentiment(tweet):
    tweet_blob = TextBlob(str(tweet))
    sentiment = tweet_blob.sentiment
    if sentiment.polarity > 0:
        pol_label =  'positive'
    elif sentiment.polarity == 0:
        pol_label = 'neutral'
    else:
        pol_label = 'negative'

    if sentiment.subjectivity > 0.5:
        subj_label = 'subjective'
    elif sentiment.subjectivity == 0.5:
        subj_label = 'neutral'
    else:
        subj_label = 'objective'
    return [pol_label,sentiment.polarity, subj_label, sentiment.subjectivity]

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

def get_all_tweets():
	tweets_df = pd.read_csv(ORIG_TWEET_FILE)
	return tweets_df

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_cleanup(text):
	'''
	Text pre-processing
		return tokenized list of cleaned words
	'''
	# Convert to lowercase
	text_clean = text.lower()
	# Remove non-alphabet
	text_clean = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ', text_clean).split()
	# Remove short words (length < 3)
	text_clean = [w for w in text_clean if len(w)>2]
	# Lemmatize text with the appropriate POS tag
	lemmatizer = WordNetLemmatizer()
	text_clean = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_clean]
	# Filter out stop words in English
	stops = set(stopwords.words('english')).union(additional_stop_words)
	text_clean = [w for w in text_clean if w not in stops]

	return text_clean

def preprocess_tweets(all_tweets_df):
    print('Start preprocessing tweets ...\n')
    # dataframe to add parsed tweets
    cleaned_tweets_df = all_tweets_df.copy(deep=True)
    # parsing tweets
    cleaned_tweets_df['token'] = [text_cleanup(x) for x in all_tweets_df['text']]
    # Save cleaned tweets to file
    cleaned_tweets_df.to_csv(CLEANED_TWEET_FILE, index=False)
    print ('Cleaned tweets saved\n')
    return cleaned_tweets_df

def word_grams(words, min=1, max=2):
    '''
    Build ngrams word list
    '''
    word_list = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            word_list.append(' '.join(str(i) for i in ngram))
    return word_list

# Parse data uploaded
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

app.layout = html.Div([
    # dbc.Nav(
    #     [
    #         dbc.NavItem(dbc.NavLink("Disclaimer", active=True, href="#")),
    #         dbc.NavItem(dbc.NavLink("Data Access", href="#")),
    #         dbc.NavItem(dbc.NavLink("Contact", href="#")),
    #         # dbc.NavItem(dbc.NavLink("Disabled", disabled=True, href="#")),
    #         # dbc.DropdownMenu(
    #         #     [dbc.DropdownMenuItem("Item 1"),
    #         #      dbc.DropdownMenuItem("Item 2")
    #         #     ],label="Dropdown", nav=True,
    #         # ),
    #     ], style={'float': 'right', 'marging':'100px'}),
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Data Summary',
                value='tab-1',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([dcc.Upload(id='upload_your_data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select dataset file', style=dict(color='blue'))]),
                                style={'width': '470px',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        "marginLeft":"20px",
                                        "marginTop":"20px"},
                                # Allow multiple files to be uploaded
                                multiple=False),
                        dbc.Tooltip(
                                    [
                                    "Want to try the model on your own textual data? Upload your data here.",
                                    html.P([html.B("NOTE: The text field should be named *text*")])
                                    ], target='upload_your_data'),

                        # dbc.Tooltip("Ignore this Button if your dataset is already clean", target='preprocess_data'),
                        html.Div(id='preprocess-status',
                                style=dict(#float='left',
                                            width = "470px",
                                            border='1px solid black',
                                            height="40px",
                                            marginLeft=20,
                                            marginTop=5,
                                            padding=5)),

                        html.Div(id='summary',
                                style=dict(marginLeft=20,
                                            width = "1500px",
                                            marginTop=5,
                                            padding=5),
                                children=[
                                        dbc.Row([
                                            dbc.Col(id='summary_daily_tweets', style=dict(width='500px')),
                                            dbc.Col(style=dict(marginTop=150, marginLeft=60),
                                                    children=[
                                                            html.Div(id='summary_total_tweets',),
                                                            html.Div(id='summary_size_dataframe',),
                                                            html.Div(id='summary_number_tweets_per_country')
                                                    ]),
                                            dbc.Col(id='summary_source_tweets'),
                                ])]),
                        # html.Div(id='total_tweets',
                        #         style=dict(marginLeft=80,
                        #                     marginTop=5,
                        #                     padding=5),
                        #         children=[
                        #                 dbc.Row([
                        #                     dbc.Col(id='summary_size_dataframe',),
                        #                     dbc.Col(id='summary_number_tweets_per_country')
                        #                 ])
                        #         ]),
                        # html.Div(id='summary_size_dataframe',
                        #         style=dict(marginLeft=80,
                        #                     marginTop=5,
                        #                     padding=5)),
                        # html.Div(id='summary_number_tweets_per_country',
                        #         style=dict(marginLeft=80,
                        #                     marginTop=5,
                        #                     padding=5)),
                        # dcc.Graph(
                        #     id='subjectivity_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x':sub_label_df['subj_label'],
                        #             'y': sub_label_df['percent'],
                        #             'type': 'bar',
                        #
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Percentage of tweets'},
                        #                 xaxis = {'title': 'Subjectivity'},
                        #         )}),
                        # dcc.Graph(
                        #     id='polarity_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x':polarity_label_df['polarity_label'],
                        #             'y': polarity_label_df['percent'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Percentage of tweets'},
                        #                 xaxis = {'title': 'Polarity'},
                        #         )}),
                        # dcc.Graph(
                        #     id='devices_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x': device_df['Source'],
                        #             'y': device_df['percent'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Percentage of tweets'},
                        #                 xaxis = {'title': 'Device'},
                        #         )}),
                        # dcc.Graph(
                        #     id='tweet_location_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x': locations['loc'],
                        #             'y': locations['Count'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Number of tweets'},
                        #                 xaxis = {'title': 'State'},
                        #         )}),
                        # dcc.Graph(
                        #     id='polarity_location_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x': sentiment_df['loc'],
                        #             'y': sentiment_df['polarity'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Overall polarity'},
                        #                 xaxis = {'title': 'State'},
                        #         )}),
                        # dcc.Graph(
                        #     id='sub_location_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x': sentiment_df['loc'],
                        #             'y': sentiment_df['subjectivity'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Overall subjectivity'},
                        #                 xaxis = {'title': 'State'},
                        #         )}),
                        # dcc.Graph(
                        #     id='ver_location_bar',
                        #     #style=dict(width='60'),
                        #     #config=dict(staticPlot=True),
                        #     figure={
                        #         'data': [{
                        #             'x': verif_df['loc'],
                        #             'y': verif_df['sum'],
                        #             'type': 'bar'
                        #         }],
                        #         'layout' : go.Layout(
                        #                 yaxis = {'title': 'Number of verified accounts'},
                        #                 xaxis = {'title': 'State'},
                        #         )})
                        ], style=dict( width="1500px", height="790px", marginTop='20px')
                        )]),
        dcc.Tab(label='Topics Modelling',
                value='tab-2',
            #    style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        # dcc.Dropdown(
                        #     id='tp_wd',
                        #     options=[{'label': i, 'value': i} for i in ['Word Cloud', 'Topics Analysis']],
                        #     value='Topics Analysis',
                        #     style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}
                        # ),
                        # html.Div([
                        # dcc.Upload(id='upload_your_data',
                        #             children=html.Div([
                        #                 'Drag and Drop or ',
                        #                 html.A('Select dataset file', style=dict(color='blue'))]),
                        #             style={'width': '470px',
                        #                     'height': '60px',
                        #                     'lineHeight': '60px',
                        #                     'borderWidth': '1px',
                        #                     'borderStyle': 'dashed',
                        #                     'borderRadius': '5px',
                        #                     'textAlign': 'center',
                        #                     'margin': '10px',
                        #                     "marginLeft":"20px",
                        #                     "marginTop":"20px"},
                        #             # Allow multiple files to be uploaded
                        #             multiple=False),
                        # dbc.Tooltip(
                        #             [
                        #             "Want to try the model on your own textual data? Upload your data here.",
                        #             html.P([html.B("NOTE: The text field should be named *text*")])
                        #             ], target='upload_your_data'),
                        #
                        # dbc.Tooltip("Ignore this Button if your dataset is already clean", target='preprocess_data'),
                        # html.Div(id='preprocess-status',
                        #         style=dict(#float='left',
                        #                     width = "470px",
                        #                     border='1px solid black',
                        #                     height="40px",
                        #                     marginLeft=20,
                        #                     marginTop=5,
                        #                     padding=5)),
                        html.B(html.P('How many topics to model?', style=dict(marginLeft=20, marginTop=20))),
                        dcc.Slider(
                            id='number_topics',
                            min=5,
                            max=30,
                            value=10,
                            step=1,
                            marks={i: str(i) for i in range(5, 31)}),
                        html.B(html.P('How many top terms per topic?', id="term_des",style=dict(marginLeft=20))),
                        dcc.Slider(
                            id='number_words',
                            min=5,
                            max=30,
                            value=10,
                            step=1,
                            marks={i: str(i) for i in range(5, 31)}),
                        dbc.Tooltip("You can vary this to show more or less terms in the wordcloud", target="term_des"),
                        html.B(html.P('Slide to select ALPHA', id='alpha_des', style=dict(marginLeft=20, marginTop=5, marginBottom=5))),
                        html.P(id='alpha_output', style=dict(marginLeft=20, marginTop=5,marginBottom=5)),

                        dcc.Slider(
                            id='alpha_value',
                            min=0.001,
                            max=1.0,
                            value=0.001,
                            step=0.001,
                            marks={i: str(i) for i in [0.001,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}),
                            # )], style={'marginLeft':'50px', "marginBottom":"5px",'width':'100%'}),
                        dbc.Tooltip("Document-Topic Density. The greater, a document will be assigned to more topics, vice versa",
                                    target='alpha_des'),

                        html.B(html.P('Slide to select ETA', id='eta_des',style=dict(marginLeft=20, marginTop=5, marginBottom=5))),

                        html.P(id='eta_output', style=dict(marginLeft=20, marginTop=5, marginBottom=5)),
                        dcc.Slider(
                            id='eta_value',
                            min=0.001,
                            max=1.0,
                            value=0.001,
                            step=0.001,
                            marks={i: str(i) for i in [0.001,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}),
                            # )], style={'marginLeft':'50px', "marginBottom":"5px",'width':'100%'}),
                        dbc.Tooltip("Topic-Word density. \
                                    The greater, each topic will contain more words, and vice versa",
                                    target="eta_des"),


                        html.B(children = [html.P('Slide to select minimum Probability of topic?', style=dict(marginLeft=20, marginTop=5, marginBottom=5))]),
                        html.P(id='min_prob_output', style=dict(marginLeft=20, marginTop=5, marginBottom=5)),
                        dcc.Slider(
                            id='min_prob_value',
                            min=0.01,
                            max=1.0,
                            value=0.01,
                            step=0.01,
                            marks={i: str(i) for i in [0.01,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}),
                            # )], style={'marginLeft':'50px', "marginBottom":"5px",'width':'100%'}),

                        html.Button(children='Run Model(Basic)',
                                    n_clicks = 0,
                                    id='submit-model',
                                    style={"marginTop":"25px",
                                            'marginLeft':'20px',
                                            "marginBottom":"5px",
                                            "marginRight":"10px"}),
                        dbc.Tooltip("The resulting visualization shall be available \
                                    for interaction in the first box on the right", target='submit-model'),

                        html.Button(children='Run Model(Advance)',
                                    n_clicks = 0,
                                    id='pylda-model',
                                    style={'float':'right',
                                            'marginLeft':'20px',
                                            "marginBottom":"5px",
                                            "marginRight":"25px",
                                            "marginTop":"25px"}),
                        dbc.Tooltip("The resulting visualization shall open for interaction \
                                    in a new browser tab", target='pylda-model'),
                        html.Div(id='pylda-status',
                                style=dict(#float='left',
                                            width = "455px",
                                            border='1px solid black',
                                            height="70px",
                                            marginLeft=20,
                                            marginTop=10,
                                            padding=10,
                                            display='none')),

                        #html.A(html.Button("Visualize Topics"), href="/lda_vis", target="_blank"),
                    ], style=dict( width="500px", float='left')),

                    html.Div(id='topics_vis_div',
                            style=dict(float='left',
                                        width = "500px",
                                        border='1px solid black',
                                        height="500px",
                                        marginLeft=20,
                                        marginTop=20)
                            ),
                    # html.Div(id='topics_terms_vis',
                    #         style=dict(float='left',
                    #                     width = "600px",
                    #                     border='1px solid black',
                    #                     height="600px",
                    #                     marginLeft=10,
                    #                     marginTop=20)
                    # ),
                    # html.Div(id='hover-data',
                    #         style=dict(float='left',
                    #                     width = "500px",
                    #                     border='1px solid black',
                    #                     height="100px",
                    #                     marginLeft=10,
                    #                     marginTop=20)),
                    html.Img(id='image_wc',
                            style=dict(float='left',
                                        width = "500px",
                                        border='1px solid black',
                                        height="500px",
                                        marginLeft=10,
                                        marginTop=20)),
                    dbc.Tooltip("Vary the number of terms per topic to visualize less \
                                or more bigrams on this word cloud",
                                target='image_wc'),
                    html.Div(id='rename_topics_div',
                            children=[],
                            style=dict(float='left',
                                        width = "1020px",
                                        border='1px solid black',
                                        height="300px",
                                        marginLeft=15,
                                        marginTop=20)),
                    html.Div(id='rename_topics_verif',
                            children=[],
                            style=dict(float='right',
                                        width = "350px",
                                        border='1px solid black',
                                        height="100px",
                                        marginLeft=5,
                                        marginRight=15,
                                        marginTop=20)),
                    html.Button(children='Rename more topics',
                                n_clicks = 0,
                                id='rename_topics_btn',
                                style={'marginLeft':'20px',
                                        "marginBottom":"5px",
                                        "marginRight":"10px",
                                        "float":'left',
                                        "marginTop":'20px'}),
                    dbc.Tooltip("Can only add as much fields as \
                                the number of topics modeled",
                                target='rename_topics_btn'),
                    html.Button(children='Update your data with new topic names',
                                n_clicks = 0,
                                id='save_topics_names_btn',
                                style={'marginLeft':'20px',
                                        "marginBottom":"5px",
                                        "marginRight":"10px",
                                        "float":'left',
                                        "marginTop":'100px'}),
                   dbc.Tooltip("Creates a new column with the topics names in your dataset",
                                target='save_topics_names_btn'),
                   html.Div(id='results_save_new_topics',
                            style={'display':'none'}),
                   dbc.Modal(id="simple_model_modal",
                            centered=True,
                            children=[
                                dbc.ModalHeader(id="simple_model_modal_heading"),
                                dbc.ModalBody(id='simple_model_modal_body'),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="simple_model_modal_close", className="ml-auto")),
                            ]),
                    dbc.Modal(id="advance_model_modal",
                             centered=True,
                             children=[
                                 dbc.ModalHeader(id="advance_model_modal_heading"),
                                 dbc.ModalBody(id='advance_model_modal_body'),
                                 dbc.ModalFooter(
                                     dbc.Button("Close", id="advance_model_modal_close", className="ml-auto")),
                             ]),
                     dbc.Modal(id="update_topics_modal",
                              centered=True,
                              children=[
                                  dbc.ModalHeader(id="update_topics_modal_heading"),
                                  dbc.ModalBody(id='update_topics_modal_body'),
                                  dbc.ModalFooter(
                                      dbc.Button("Close", id="update_topics_modal_close", className="ml-auto")),
                              ]),

                    # html.Div(id='preprocess-status',
                    #         style=dict(float='left',
                    #                     width = "250px",
                    #                     border='1px solid black',
                    #                     height="30px",
                    #                     marginLeft=15,
                    #                     marginTop=20))
                    # html.Div(id='topics_words_vis',
                    #         style=dict(float='left',
                    #                     width = "600px",
                    #                     border='1px solid black',
                    #                     height="200px",
                    #                     marginLeft=10,
                    #                     marginTop=20)),


                ]),

        dcc.Tab(label='Sentiments',
                value='tab-3',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([dcc.Dropdown(
                            id='sentiment_dropdown',
                            options=[{'label': i, 'value': i} for i in ["World", 'USA', 'France']],
                            value='World',
                            style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}
                            ),
                            html.Div(id='sentiment_output')], style=dict( width="1500px", height="790px"))]),
        dcc.Tab(label='User Mapping',
                value='tab-4',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        dcc.Dropdown(
                            id='user_mapping_dropdown',
                            options=[{'label': i, 'value': i} for i in ['USA', 'France']],
                            value='USA',
                            style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}),
                        html.Div(id='user_mapping_output')], style=dict( width="1500px", height="790px"))]),
        dcc.Tab(label='Personal Data',
                value='tab-6',
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        dcc.Upload(id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files', style=dict(color='blue'))]),
                                    style={'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'},
                                    # Allow multiple files to be uploaded
                                    multiple=False),
                        #dcc.Graph(id='vis_data_upload'),
                        html.Div(id='output-data-upload')
                    ])
                ])
        # dcc.Tab(label='Words Cloud',
        #         value='tab-5',
        #         #style=tab_style,
        #         selected_style=tab_selected_style,
        #         children=[
        #             html.Div([
        #                 html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
        #                         width=700,
        #                         height=700,
        #                         style=dict(float='left', marginLeft=500)
        #                         )
        #             ], style=dict( width="1500px", height="790px", marginTop='20px'))])
    ],  #vertical=True,
        #parent_style={'float': 'left', 'marginLeft':'5px', 'marginTop':90, "position":'absolute'}
    ),

])

# @app.server.route("/lda_vis")
# def get_report():
#     #lda_out_file = 'lda.html'
#     return flask.send_from_directory(Path('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/'), 'lda.html')

# Upload and preprocess the data
@app.callback(
    [Output('preprocess-status', 'children'),
    Output('summary_daily_tweets', 'children'),
    Output('summary_source_tweets', 'children'),
    Output('summary_total_tweets', 'children'),
    Output('summary_size_dataframe', 'children'),
    Output('summary_number_tweets_per_country', 'children'),],
    [Input('upload_your_data', 'contents'),Input('upload_your_data', 'filename')])
def preprocess_data(contents,filename):
    results = html.P("Using Default data for the moment", style=dict(color='red'))
    if contents is not None:
        try:
            #print(filename)
            df = parse_data(contents, filename)
            #print(df.columns)
            #print(df.shape)
            filen=filename
        except Exception as e:
            print('Error:--{}-- in callback preprocess_data[1st exception]'.format(e))
            results = html.P('Error {} '.format(e), style=dict(color='red'))
    else:
        print('Invalid content: {}....'.format(contents))
        print('Reverting to the default dataset....')
        results = html.P("Error. The text field should be named *text* \
                          Using the default datas", style=dict(color='red'))
        df = pd.read_csv(ORIG_TWEET_FILE)
        filen = "default file"
    try:
        global all_tweets_df
        df['text'] = df['text'].astype(str)
        df['loc']=df['loc'].astype(str)
        all_tweets_df = df.copy(deep=True)
        # Preprocess tweets
        cleaned_tweets_df = preprocess_tweets(all_tweets_df)
        # Generate ngram tokens
        cleaned_tweets_df['ngram_token'] = [word_grams(x, NUM_GRAMS, NUM_GRAMS+1) for
                                             x in cleaned_tweets_df['token']]
        # Build dictionary
        tweets_dict = corpora.Dictionary(cleaned_tweets_df['ngram_token'])
        # Remove words that occur in less than 10 documents,
        # or in more than 50% of the documents
        tweets_dict.filter_extremes(no_below=10, no_above=0.5)
        # Transform doc to a vectorized form by computing frequency of each word
        bow_corpus = [tweets_dict.doc2bow(doc) for doc in cleaned_tweets_df['ngram_token']]

        # Create tf-idf model and then apply transformation to the entire corpus
        tfidf = models.TfidfModel(bow_corpus)
        global corpus
        corpus = tfidf[bow_corpus]
        global loaded_dict
        loaded_dict = tweets_dict

        # Save freq and tfidf corpus, dictionary and cleaned dataset to files
        MmCorpus.serialize(CORPUS_FILE, bow_corpus)
        MmCorpus.serialize(CORPUS_TFIDF_FILE, corpus)
        tweets_dict.save(DICT_FILE)
        cleaned_tweets_df.to_csv(CLEANED_BIGRAM_TWEET_FILE, index=False)
        global FR_DF
        global US_DF
        FR_DF = cleaned_tweets_df[cleaned_tweets_df['loc'].str.lower().str.contains('france')]
        US_DF = cleaned_tweets_df[cleaned_tweets_df['loc'].str.contains('USA')]
        FR_DF.to_csv(FR_OUTPUT_FILE, index=False)
        US_DF.to_csv(US_OUTPUT_FILE, index=False)

        daily_tweets_df = pd.DataFrame(columns=['created_at', 'count'])
        df_days = cleaned_tweets_df[['created_at', 'text']]
        for idx, data in df_days.groupby('created_at'):
            date = idx
            count = data['created_at'].count()
            data_append = {daily_tweets_df.columns[0]:date,
                           daily_tweets_df.columns[1]:count,
                          }
            append_df = pd.DataFrame(data=[data_append])
            daily_tweets_df = pd.concat([daily_tweets_df, append_df], sort=False)
        daily_tweets_df.to_csv(DAILY_TWEETS_COUNT_FILE, index=False)

        daily_bar = dcc.Graph(
            id='daily_tweets_bar',
            #style=dict(width='60'),
            #config=dict(staticPlot=True),
            figure={
                'data': [{
                    'x': daily_tweets_df['created_at'],
                    'y': daily_tweets_df['count'],
                    'type': 'line'
                }],
                'layout' : go.Layout(
                        yaxis = {'title': 'Number of tweets'},
                        xaxis = {'title': 'Day'},
                )})
        total_tweets= html.P("Total Number of Tweets: {}".format(cleaned_tweets_df.shape[0]))
        size_df = round(cleaned_tweets_df.memory_usage(index=True, deep=True).sum()/10**6, 2)
        dataframe_size = html.P("Size of dataframe: {}MB".format(size_df))
        tweets_per_country = html.Div([
                                html.P('Tweets from France: {}'.format(FR_DF.shape[0])),
                                html.P('Tweets from USA: {}'.format(US_DF.shape[0]))
                            ])
        devices = cleaned_tweets_df.groupby(['source'])['source'].count().reset_index(name='count')
        source_bar = dcc.Graph(
            id='daily_tweets_bar',
            #style=dict(width='60'),
            #config=dict(staticPlot=True),
            figure={
                'data': [{
                    'x': devices['source'],
                    'y': devices['count'],
                    'type': 'bar'
                }],
                'layout' : go.Layout(
                        yaxis = {'title': 'Number of tweets'},
                        xaxis = {'title': 'Device'},
                )})


        if filen == "default file":
            results = html.P("Preprocessing completed and now using the default dataset", style=dict(color='blue'))
        else:
            results = html.P('Preprocessing {} completed and now using it'.format(filen), style=dict(color='royalblue'))
    except Exception as e:
        print('Error:-{}- in callback preprocess_data[final exception] '.format(e))
        results = html.P('Error {} '.format(e), style=dict(color='red'))
    return results, daily_bar,source_bar,total_tweets, dataframe_size, tweets_per_country

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents'),Input('upload-data', 'filename')])
def update_output(contents, filename):
    table = html.Div()
    print('Reading uploaded data....')
    if contents is not None:
        #print(contents)
        #contents = contents[0]
        #filename = filename[0]
        #print(filename)
        df = parse_data(contents, filename)
        #print(df.columns)
        #print(df.shape)

        table = html.Div([
                    html.P(["Name of uploaded file: ", html.B(filename)]),
                    html.Div('Raw Content'),
                    html.Pre(contents[0:200] + '...',
                            style={'whiteSpace': 'pre-wrap','wordBreak': 'break-all'}),
                    html.Hr(),
                    dash_table.DataTable(
                        data=df.to_dict('rows'),
                        columns=[{'name': i, 'id': i} for i in df.columns],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'},
                        style_data_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'}],
                        style_cell={'textAlign': 'center', 'padding': '5px', "marginLeft":'5px'},
                        )
                ])
    return table
# @app.callback(
#     Output(component_id="rename_topics_div", component_property="children"),
#     [Input(component_id='number_topics', component_property='value')])
# def set_alpha(number_topics):
#     # new_input = dcc.Input(
#     # id={'type': 'filter-dropdown',
#     #     'index': number_topics
#     # })
#     # children.append(new_input)
#     # return children
#     #return html.P("{} topics selected".format(number_topics))
#
#     new_dropdown = dcc.Dropdown(
#     id={
#         'type': 'filter-dropdown',
#         'index': value
#     },
#     options=[{'label': i, 'value': i} for i in ['NYC', 'MTL', 'LA', 'TOKYO']]
#     )
#     children.append(new_dropdown)
#     return children

#https://dash.plotly.com/pattern-matching-callbacks
@app.callback(
    Output(component_id="rename_topics_div", component_property="children"),
    [Input(component_id='rename_topics_btn', component_property='n_clicks')],
    [State(component_id="rename_topics_div", component_property="children"),
    State(component_id="number_topics", component_property="value")])
def new_input_field(n_clicks, children,number_topics):
    if n_clicks <= number_topics-1:
        new_input = dcc.Input(id={'type': 'add-input-field',
                                'index': n_clicks},
                                placeholder='Rename Topic {}'.format(n_clicks)
                                    )
        children.append(new_input)
        #print(new_dropdown.id)
        #print(children)
        return children
    elif n_clicks > number_topics-1:
        #print("You can't rename more than {} topics".format(num_topics))
        return html.P("You can't rename more than {} topics".format(num_topics))

@app.callback(
    Output('rename_topics_verif', 'children'),
    [Input({'type': 'add-input-field', 'index': ALL}, 'value')]
)
def display_output(values):
    #print("Trying to print something")
    return html.Div([
                    html.Div('Topic {} = {}'.format(i, value))
                    for (i, value) in enumerate(values)
                    ])
@app.callback(
    [Output(component_id='results_save_new_topics', component_property='children'),
    Output(component_id="update_topics_modal", component_property="is_open"),
    Output(component_id="update_topics_modal_heading", component_property="children"),
    Output(component_id='update_topics_modal_body', component_property="children")],
    [Input(component_id='save_topics_names_btn',component_property='n_clicks'),
    Input(component_id="update_topics_modal_close", component_property='n_clicks')],
    [State({'type': 'add-input-field', 'index': ALL}, 'value'),
    State({'type': 'add-input-field', 'index': ALL}, 'id'),
    State(component_id="update_topics_modal", component_property="is_open")]
)
def save_new_topics_name(n1, n2, values, id, is_open):
    if n1 or n2:
        try:
            update_df = pd.read_csv('topic_coordinates.csv')

            global topic_new_name_dict

            for i in range(len(values)):
                topic_new_name_dict[i]=values[i]
            print(topic_new_name_dict)

            global polarity_topics_df
            print(polarity_topics_df.columns)
            polarity_topics_df['topic_name'] = polarity_topics_df['predicted_topic'].apply(topic_rename_f)
            polarity_topics_df.to_csv('polarity_topic_sentiment.csv', index=False)

            #####################################################################################
            ####### Grouping polarity by topics
            pol_top_sel = polarity_topics_df[['polarity_label', 'topic_name']]
            pol_result_df = pd.DataFrame(columns=['predicted_topic','polarity_label', 'count'])
            for idx, data in pol_top_sel.groupby(['topic_name', 'polarity_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {pol_result_df.columns[0]:topic,
                               pol_result_df.columns[1]:label,
                               pol_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                pol_result_df = pd.concat([pol_result_df, append_df], sort=False)
            global renamed_topics_polarity_df
            renamed_topics_polarity_df = pol_result_df

            ###### Grouping subjectivity by topics
            subj_top_sel = polarity_topics_df[['sub_label', 'topic_name']]
            subj_result_df = pd.DataFrame(columns=['predicted_topic','subj_label', 'count'])
            for idx, data in subj_top_sel.groupby(['topic_name', 'sub_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {subj_result_df.columns[0]:topic,
                               subj_result_df.columns[1]:label,
                               subj_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                subj_result_df = pd.concat([subj_result_df, append_df], sort=False)
            global renamed_topics_subj_df
            renamed_topics_subj_df = subj_result_df

            #print(renamed_topics_polarity_df.head())
            #print(renamed_topics_subj_df.head())
            topics_names_exist = True
            #################################################################################

            update_df['topic_name'] = values
            update_df.to_csv('topic_coordinates_update.csv',index=False)
            results = html.P('Topics successfully renamed')
            heading = html.P('Success')
            body = html.P('Topics successfully renamed')
            return results, not is_open, heading, body
        except Exception as e:
            print('Error:-{}- in callback save_new_topics_name'.format(e))
            results = html.P('Error updating topics')
            heading = html.P('Failure')
            body = html.P('Error: {}'.format(e))
            return results, not is_open, heading, body

    #return html.P('Topics successfully renamed'), is_open, html.P('Success'), html.P('Topics successfully renamed')

@app.callback(
    Output(component_id="image_wc", component_property="src"),
    [Input(component_id='topics_vis', component_property='hoverData')],
    [State(component_id='number_words', component_property='value')])
def display_terms_word_cloud(hoverData,number_wds):
    topics = lda_model.show_topics(formatted=False,num_words=number_wds)
    topic = hoverData['points'][0]['text']
    topic_words = dict(topics[topic][1])
    #print(topic_words)
    global stop_words
    global colors_list
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=300,
                      height=300,
                      #max_words=10,
                      colormap='tab10',
                      # color_func=lambda *args, **kwargs: colors_list[topic],
                      prefer_horizontal=1.0)
    #print(cloud)
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    cloud_image = cloud.to_image()
    buffer = BytesIO()
    cloud_image.save(buffer, format='PNG')
    wd_cloud = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64,{}".format(wd_cloud)


@app.callback(
    Output(component_id="alpha_output", component_property="children"),
    [Input(component_id='alpha_value', component_property='value')])
def set_alpha(alpha_val):
    return "ALPHA = "+ str(alpha_val)

@app.callback(
    Output(component_id="eta_output", component_property="children"),
    [Input(component_id='eta_value', component_property='value')])
def set_eta(eta_val):
    return "ETA = "+ str(eta_val)

@app.callback(
    Output(component_id="min_prob_output", component_property="children"),
    [Input(component_id='min_prob_value', component_property='value')])
def set_proba(prob_val):
    return "Topics with Probability < {} shall be rejected?".format(prob_val)

# @app.callback(
#     Output('hover-data', 'children'),
#     [Input('topics_vis', 'hoverData')])
# def callback_image(hoverData):
#     #return json.dumps(initial_token_table.to_json(), indent=2)
#     print(int(hoverData['points'][0]['text']))
#     return json.dumps(hoverData, indent=2)

@app.callback(
    [Output(component_id="advance_model_modal", component_property="is_open"),
    Output(component_id="advance_model_modal_heading", component_property="children"),
    Output(component_id='advance_model_modal_body', component_property="children")],
    [Input(component_id='pylda-model', component_property='n_clicks'),
    Input(component_id="advance_model_modal_close", component_property='n_clicks')],
    [State(component_id="advance_model_modal", component_property="is_open")]
)
def toggle_advance_model_modal(n1,n2, is_open):
    if n1 or n2:
        return not is_open, 'Advance Model Now Training', "The results will open shortly as \
                                                        an interactive visualisation in a new browser tab"
    return is_open, 'Advance Model Now Training', "The results will open shortly as \
                                                    a set of highly interactive visualisations in a new browser tab."

@app.callback(
    [Output(component_id="simple_model_modal", component_property="is_open"),
    Output(component_id="simple_model_modal_heading", component_property="children"),
    Output(component_id='simple_model_modal_body', component_property="children")],
    [Input(component_id='submit-model', component_property='n_clicks'),
    Input(component_id="simple_model_modal_close", component_property='n_clicks')],
    [State(component_id="simple_model_modal", component_property="is_open")]
)
def toggle_simple_model_modal(n1,n2, is_open):
    if n1 or n2:
        return not is_open, 'Simple Model Now Training', "The results will open shortly as an interactive \
                                                            bubble chart in one of the boxes on this page"
    return is_open, 'Simple Model Now Training', "The results will open shortly as an interactive \
                                                        bubble chart in one of the boxes on this page."

@app.callback(
    Output(component_id='topics_vis_div', component_property="children"),
    [Input(component_id='submit-model', component_property='n_clicks')],
    [State(component_id='number_topics', component_property='value'),
    State(component_id='alpha_value', component_property='value'),
    State(component_id='eta_value', component_property='value'),
    State(component_id='min_prob_value', component_property='value')])
def run_simple_model(n_clicks, n_topics, alpha_value, eta_value, min_prob_value):
    if (n_clicks >0 ):
        try:
            print('Start Simple LDA model training ...\n')
            global corpus
            global loaded_dict

            ldamodel = models.LdaMulticore(corpus=corpus,
                                             num_topics=n_topics,
                                             id2word=loaded_dict,
                                             passes=NUM_PASSES,
                                             alpha=alpha_value,
                                             eta=eta_value,
                                             minimum_probability = min_prob_value,
                                             random_state=49)
            # Save LDA model to file
            ldamodel.save(LDA_MODEL_FILE)
            global lda_model
            lda_model= ldamodel
            print ('LDA model saved\n')
            lda_data =  pyLDAvis.gensim.prepare(ldamodel, corpus, loaded_dict, mds='mmds')
            topic_coordinates = lda_data.topic_coordinates
            token_table = lda_data.token_table
            token_table.to_csv('token_table.csv', index=False)
            token_df =token_table
            result_df = pd.DataFrame(columns=['Term', 'Freq'])
            terms_df = token_df[['Term', 'Freq']]

            for idx, data in terms_df.groupby('Term'):
                freq = data['Freq'].sum()
                term = idx
                data_append = {result_df.columns[0]:term,
                               result_df.columns[1]:freq,
                              }
                append_df = pd.DataFrame(data=[data_append])
                result_df = pd.concat([result_df, append_df], sort=False)
            result_df.sort_values('Freq',ascending=False).to_csv('term_overall_freq.csv', index=False)

            #pyLDAvis.display(lda_data)
            pyLDAvis.save_html(lda_data, lda_out_file)
            print ('Topic visual saved\n')
            topic_coordinates.reset_index(inplace=True)
            topic_coordinates.sort_values('topic', ascending=True, inplace=True)
            topic_coordinates.to_csv('topic_coordinates.csv',index=False)
            figure = dict(
                    data = [go.Scatter(
                                x=topic_coordinates['x'],
                                y=topic_coordinates['y'],
                                text=topic_coordinates['topic'],  # use the new column for the hover text
                                mode='markers+text',
                                marker=dict(size=topic_coordinates['Freq'].mean()+(2*topic_coordinates['Freq'])),
                        )],
                    layout = go.Layout(
                                title='Intertopic Distance Map (via multidimensional scaling)',
                                hovermode='closest',
                                xaxis=dict(title='PC2'),
                                yaxis=dict(title='PC1')
                ))
            ####### Getting sentiment and prediction of topic
            global all_tweets_df
            print(all_tweets_df.columns)
            print('Trying to use spark..... -> In function run_simple_model')
            spark_df = spark.createDataFrame(all_tweets_df)
            polarity_df = spark_df.withColumn("polarity_label", udf_polarity_label(spark_df.text))
            sub_df = polarity_df.withColumn("sub_label", udf_subj_label(polarity_df.text))
            global polarity_topics_df
            polarity_topics_df = sub_df.toPandas()
            #print(polarity_topics_df.head())
            #global lda_model
            #global corpus
            predictions = lda_model[corpus]
            pred_topic = dict()
            #print('I am here')
            for i, pred in enumerate(predictions):
                if len(pred) == 1:
                    pred_topic[i] = pred[0][0]
                else:
                    pred_topic[i] = -1
            pred_list = list(pred_topic.values())
            #### If you consider applying the name to the topics you could create a global variable
            #### and store polarity_topics_df to it
            polarity_topics_df['predicted_topic'] = pred_list
            polarity_topics_df.to_csv('polarity_topic.csv', index=False)
            #print(polarity_topics_df.head())

            ####### Grouping polarity by topics
            pol_top_sel = polarity_topics_df[['polarity_label', 'predicted_topic']]
            pol_result_df = pd.DataFrame(columns=['predicted_topic','polarity_label', 'count'])
            for idx, data in pol_top_sel.groupby(['predicted_topic', 'polarity_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {pol_result_df.columns[0]:topic,
                               pol_result_df.columns[1]:label,
                               pol_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                pol_result_df = pd.concat([pol_result_df, append_df], sort=False)
            global topics_polarity_df
            topics_polarity_df = pol_result_df

            ###### Grouping subjectivity by topics
            subj_top_sel = polarity_topics_df[[ 'sub_label', 'predicted_topic']]
            subj_result_df = pd.DataFrame(columns=['predicted_topic','subj_label', 'count'])
            for idx, data in subj_top_sel.groupby(['predicted_topic', 'sub_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {subj_result_df.columns[0]:topic,
                               subj_result_df.columns[1]:label,
                               subj_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                subj_result_df = pd.concat([subj_result_df, append_df], sort=False)
            global topics_subj_df
            topics_subj_df = subj_result_df
            print ('Topic visual saved\n')
        except Exception as e:
            print('Error:-{}- in function run_simple_model'.format(e))
            figure = dict(
                    data = [go.Scatter(
                                x=[],
                                y=[],

                        )],
                    layout = go.Layout(
                                title='Error {} occured'.format(e),
                                hovermode='closest',
                                xaxis=dict(title='PC1'),
                                yaxis=dict(title='PC2')))
        graph = dcc.Graph(id='topics_vis', figure=figure)
        return graph

@app.callback(
    Output(component_id='pylda-status', component_property="children"),
    [Input(component_id='pylda-model', component_property='n_clicks')],
    [State(component_id='number_topics', component_property='value'),
    State(component_id='alpha_value', component_property='value'),
    State(component_id='eta_value', component_property='value'),
    State(component_id='min_prob_value', component_property='value')])
def run_general_model(n_clicks, n_topics, alpha_value, eta_value, min_prob_value):
    results = html.P('Results of running the Advanced Model will open in a new browser tab when completed', style=dict(color='red'))
    if (n_clicks >0 ):
        try:
            global corpus
            global loaded_dict
            print('Start General LDA model training ...\n')
            ldamodel = models.LdaMulticore(corpus=corpus,
                                                 num_topics=n_topics,
                                                 id2word=loaded_dict,
                                                 passes=NUM_PASSES,
                                                 alpha=alpha_value,
                                                 eta=eta_value,
                                                 minimum_probability = min_prob_value,
                                                 random_state=49)
            lda_data =  pyLDAvis.gensim.prepare(ldamodel, corpus, loaded_dict, mds='mmds')
            global lda_model
            lda_model = ldamodel
            print('Done Training General Model...')
            # Save LDA model to file
            ldamodel.save(LDA_MODEL_FILE)

            results = html.P('The model has just opened in another tab', style=dict(color='blue'))

            ####### Getting sentiment and prediction of topic
            global all_tweets_df
            print(all_tweets_df.columns)
            print('Trying to use spark.....-> In function run_general_model')
            spark_df = spark.createDataFrame(all_tweets_df)
            polarity_df = spark_df.withColumn("polarity_label", udf_polarity_label(spark_df.text))
            sub_df = polarity_df.withColumn("sub_label", udf_subj_label(polarity_df.text))
            global polarity_topics_df
            polarity_topics_df = sub_df.toPandas()
            print(polarity_topics_df.head())
            #global lda_model
            #global corpus
            predictions = lda_model[corpus]
            pred_topic = dict()
            #print('I am here')
            for i, pred in enumerate(predictions):
                if len(pred) == 1:
                    pred_topic[i] = pred[0][0]
                else:
                    pred_topic[i] = -1
            pred_list = list(pred_topic.values())
            #### If you consider applying the new names to the topics you could create a global variable
            #### and store polarity_topics_df to it
            polarity_topics_df['predicted_topic'] = pred_list
            polarity_topics_df.to_csv('polarity_topic.csv', index=False)
            print(polarity_topics_df.head())

            ####### Grouping polarity by topics
            pol_top_sel = polarity_topics_df[['polarity_label', 'predicted_topic']]
            pol_result_df = pd.DataFrame(columns=['predicted_topic','polarity_label', 'count'])
            for idx, data in pol_top_sel.groupby(['predicted_topic', 'polarity_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {pol_result_df.columns[0]:topic,
                               pol_result_df.columns[1]:label,
                               pol_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                pol_result_df = pd.concat([pol_result_df, append_df], sort=False)
            global topics_polarity_df
            topics_polarity_df = pol_result_df

            ###### Grouping subjectivity by topics
            subj_top_sel = polarity_topics_df[[ 'sub_label', 'predicted_topic']]
            subj_result_df = pd.DataFrame(columns=['predicted_topic','subj_label', 'count'])
            for idx, data in subj_top_sel.groupby(['predicted_topic', 'sub_label']):
                topic = idx[0]
                label = idx[1]
                count = data.shape[0]
                data_append = {subj_result_df.columns[0]:topic,
                               subj_result_df.columns[1]:label,
                               subj_result_df.columns[2]:count,
                              }
                append_df = pd.DataFrame(data=[data_append])
                subj_result_df = pd.concat([subj_result_df, append_df], sort=False)
            global topics_subj_df
            topics_subj_df = subj_result_df
            print ('LDA model saved\n')
            pyLDAvis.show(lda_data,port=8888)
        except Exception as e:
            print('Error:-{}- in callback run_general_model')
            print('We have some problems runnning the model: {}'.format(e))
            results = html.P('We have some problems runnning the model: {}'.format(e), style=dict(color='red'))
    return results

# @app.callback(
#     Output(component_id="topics_terms_vis", component_property="children"),
#     [Input(component_id='topics_vis', component_property='hoverData')])
# def display_topic_terms(hoverData):
#     topic = hoverData['points'][0]['text']
#
#     term_overall_freq_df = pd.read_csv('term_overall_freq.csv')
#     token_table_df = pd.read_csv('token_table.csv')
#
#
#     topic_df = token_table_df[token_table_df['Topic']==topic]
#
#     df1 = term_overall_freq_df
#     df2 = token_table_df
#     topic_i = df2[df2['Topic']==topic]
#     merged_df = pd.merge(df1,topic_i, on=['Term'], how='inner')
#
#     trace1 = go.Bar(
#                 x=merged_df['Term'],
#                 y=merged_df['Freq_y'],
#                 name='Term Frequency within the selected Topic',
#                 marker=dict(color='red'),
#                 cliponaxis=False,
#                 hoverinfo='y'
#                 #orientation='h'
#                 #marker=dict(size=2*topic_coordinates['Freq'])
#         )
#     trace2 = go.Bar(
#                 x=merged_df['Term'],
#                 y=merged_df['Freq_x'],
#                 name='Overall Term Frequency',
#                 marker=dict(color='blue'),
#                 hoverinfo='y'
#                 #cliponaxis=False,
#                 #orientation='h'
#                 #marker=dict(size=2*topic_coordinates['Freq'])
#         )
#     data = [trace1,trace2]
#     layout = go.Layout(
#             title='Frequency of terms in topic {}'.format(topic),
#             #hovermode='closest',
#             #xaxis=dict(title='Frequency', automargin=True),
#             margin=dict(l=160),
#             barmode='stack',
#     )
#     map = dcc.Graph(figure=dict(data=data, layout=layout))
#     return map

@app.callback(
    Output(component_id="sentiment_output", component_property="children"),
    [Input(component_id='sentiment_dropdown', component_property="value")]
)
def country_dropdown(country):
    if country == 'France':
        return html.P('No data to display for the moment. We are sorry!', style=dict(marginLeft=60, marginTop=10, color='red'))
    else:
        div1=  html.Div(
                [dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id='polarity_map',
                            style=dict( border='10'),
                            #config=dict(staticPlot=True),
                            figure = dict(
                                data = [go.Choropleth(
                                            locations=sentiment_df['loc'], # Spatial coordinates
                                            z = sentiment_df['polarity'].astype(float), # Data to be color-coded
                                            locationmode = 'USA-states', # set of locations match entries in `locations`
                                            colorscale = 'Reds',
                                            colorbar_title = "Tweets polarity",
                                            text=sentiment_df.apply(lambda row: f"{row['loc']}<br>{row['count']}<br>{row['polarity']}", axis=1),
                                            hoverinfo="text"
                                )],
                                layout = go.Layout(
                                        title = 'USA 05/03/2020 Corona Tweets polarity by State',
                                        geo_scope='usa', # limite map scope to USA
                                )))),
                    dbc.Col(
                        dcc.Graph(
                            id='subjectivity_map',
                            style=dict( border='10'),
                            #config=dict(staticPlot=True),
                            figure = dict(
                                data = [go.Choropleth(
                                            locations=sentiment_df['loc'], # Spatial coordinates
                                            z = sentiment_df['subjectivity'].astype(float), # Data to be color-coded
                                            locationmode = 'USA-states', # set of locations match entries in `locations`
                                            colorscale = 'Reds',
                                            colorbar_title = "Tweets subjectivity",
                                            text=sentiment_df.apply(lambda row: f"{row['loc']}<br>{row['count']}<br>{row['subjectivity']}", axis=1),
                                            hoverinfo="text"
                                )],
                                layout = go.Layout(
                                        title = 'USA 05/03/2020 Corona Tweets subjectivity by State',
                                        geo_scope='usa', # limite map scope to USA
                        ))))])])
        try:
            if "topic_name" in polarity_topics_df.columns:
                print("topic_name exist in the dataset")
                options = polarity_topics_df["topic_name"].unique()
                global topics_names_exist
                #topics_names_exist = True
            else:
                print("topic_name doesn't exist in the dataset")
                options = polarity_topics_df['predicted_topic'].unique()
            div2=  html.Div([
                        dcc.Dropdown(
                            id='sentiment_topic_dropdown',
                            options=[{'label': "Topic: {}".format(i), 'value': i} for i in options],
                            value=options[0],
                            style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}
                            ),
                        html.Div(id='sentiment_topic_output'),
                    ], style=dict( width="1500px", height="790px"))
            return div1,div2
        except Exception as e:
            print('Error {} --> In function country_dropdown'.format(e))

@app.callback(
    Output(component_id='sentiment_topic_output', component_property="children"),
    [Input(component_id='sentiment_topic_dropdown', component_property="value")]
)
def sentiment_topics_pie(topic):
    global renamed_topics_polarity_df
    global renamed_topics_subj_df
    if 'predicted_topic' in renamed_topics_polarity_df.columns:
        try:
            print(renamed_topics_polarity_df.head())
            print(renamed_topics_subj_df.head())
            pol_mask = renamed_topics_polarity_df['predicted_topic'] == topic
            subj_mask = renamed_topics_subj_df['predicted_topic'] == topic
            polDf = renamed_topics_polarity_df[pol_mask]
            subjDf = renamed_topics_subj_df[subj_mask]
            # Source: https://plotly.com/python/pie-charts/
            # Create subplots: use 'domain' type for Pie subplot
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
            fig.add_trace(go.Pie(labels=list(polDf['polarity_label'].values), values=list(polDf['count'].values), name="Polarity"),1, 1)
            fig.add_trace(go.Pie(labels=list(subjDf['subj_label'].values), values=list(subjDf['count'].values), name="Subjectivity"),1, 2)
            # Use `hole` to create a donut-like pie chart
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(title_text="Sentiment analysis of Topic:--> {}".format(topic),
            # Add annotations in the center of the donut pies.
                             annotations=[dict(text='Polarity', x=0.18, y=0.5, font_size=20, showarrow=False),
                                          dict(text='Subjectivity', x=0.82, y=0.5, font_size=20, showarrow=False)
                                         ])
            graph =  dcc.Graph(figure=fig)
        except Exception as e:
            print('Error: {} ---> In function to plot pie chart'.format(e))
            graph = html.P("Error: {}".format(e))
    else:
        try:
            print("topic name doesn't exist in dataset renamed_topics_polarity_df")
            pol_mask = topics_polarity_df['predicted_topic'] == topic
            subj_mask = topics_subj_df['predicted_topic'] == topic
            polDf = topics_polarity_df[pol_mask]
            subjDf = topics_subj_df[subj_mask]
            print('I am here in the function to plot the pie chart')
            # Source: https://plotly.com/python/pie-charts/
            # Create subplots: use 'domain' type for Pie subplot
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
            fig.add_trace(go.Pie(labels=list(polDf['polarity_label'].values), values=list(polDf['count'].values), name="Polarity"),1, 1)
            fig.add_trace(go.Pie(labels=list(subjDf['subj_label'].values), values=list(subjDf['count'].values), name="Subjectivity"),1, 2)
            # Use `hole` to create a donut-like pie chart
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(title_text="Sentiment analysis of Topic:--> {}".format(topic),
            # Add annotations in the center of the donut pies.
                             annotations=[dict(text='Polarity', x=0.18, y=0.5, font_size=20, showarrow=False),
                                          dict(text='Subjectivity', x=0.82, y=0.5, font_size=20, showarrow=False)
                                         ])
            graph =  dcc.Graph(figure=fig)
        except Exception as e:
            print('Error: {} ---> In function to plot pie chart'.format(e))
            graph = html.P("Error: {}".format(e))
    return graph


@app.callback(
    Output(component_id="user_mapping_output", component_property="children"),
    [Input(component_id='user_mapping_dropdown', component_property="value")]
)
def user_map_dropdown(country):
    if country == 'France':
        return html.P('No data to display for the moment. We are sorry!', style=dict(marginLeft=60, marginTop=10, color='red'))
    else:
        return dcc.Graph(
                        id='graph-2-tabs',
                        style=dict( border='10'),
                        #config=dict(staticPlot=True),
                        figure = dict(
                            data = [go.Choropleth(
                                        locations=locations['loc'], # Spatial coordinates
                                        z = locations['Count'].astype(float), # Data to be color-coded
                                        locationmode = 'USA-states', # set of locations match entries in `locations`
                                        colorscale = 'Reds',
                                        colorbar_title = "Tweets count",
                            )],
                            layout = go.Layout(
                                    title = 'USA 05/03/2020 Corona Tweets by State',
                                    geo_scope='usa', # limite map scope to USA
                            )))



if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    #app.run_server(debug=True,dev_tools_ui=False,dev_tools_props_check=False)
