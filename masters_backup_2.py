import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import flask
#import dash_dangerously_set_inner_html
#import dash_auth
import json
import plotly.graph_objs as go
from io import BytesIO
from wordcloud import WordCloud
import base64
import time

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
#Load previously saved bow_corpus
corpus = MmCorpus('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/models/clean_tweets_corpus.mm')
DICT_FILE = '/home/ulrich/Documents/dash-plotly/Masters2/dhifli/models/clean_tweets.dict'
#Load previously saved Dictionary of tokens
loaded_dict = corpora.Dictionary.load(DICT_FILE)

#Load previously saved LDA Model
LDA_MODEL_FILE = '/home/ulrich/Documents/dash-plotly/Masters2/dhifli/models/tweets_lda.model'
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


# def train_lda_model(n_topics):
#     print('Start LDA model training ...\n')
#     # # Build dictionary
#     # tweets_dict = corpora.Dictionary(token_tweets)
#     # # Remove words that occur in less than 10 documents,
#     # # or in more than 50% of the documents
#     # tweets_dict.filter_extremes(no_below=10, no_above=0.5)
#     # # Transform doc to a vectorized form by computing frequency of each word
#     # bow_corpus = [tweets_dict.doc2bow(doc) for doc in token_tweets]
#     # # Save corpus and dictionary to file
#     # MmCorpus.serialize(CORPUS_FILE, bow_corpus)
#     # tweets_dict.save(DICT_FILE)
#     #
#     # # Create tf-idf model and then apply transformation to the entire corpus
#     # tfidf = models.TfidfModel(bow_corpus)
#     # tfidf_corpus = tfidf[bow_corpus]
#
#     # Train LDA model
#     lda_model = models.ldamodel.LdaModel(corpus=corpus,
#                                          num_topics=n_topics,
#                                          id2word=loaded_dict,
#                                          passes=NUM_PASSES,
#                                          alpha=ALPHA,
#                                          eta=ETA,
#                                          random_state=49)
#     # Save LDA model to file
#     # lda_model.save(LDA_MODEL_FILE)
#     # print ('LDA model saved\n')
#     #
#     # # Save all generated topics to a file
#     # msg = ''
#     # for idx, topic in lda_model.print_topics(-1):
#     #     msg += 'Topic: {} \nWords: {}\n'.format(idx, topic)
#     # save_print_to_file(LDA_TOPICS_FILE, msg)
#
#     # Evaluate LDA model performance
#     #eval_lda (lda_model, tfidf_corpus, tweets_dict, token_tweets)
#     # Visualize topics
#     vis_topics(lda_model, corpus, loaded_dict)
#     return lda_model

# def vis_topics(lda_model, corpus, dict):
#     '''
#     Plot generated topics on an interactive graph
#     '''
#     lda_data =  pyLDAvis.gensim.prepare(lda_model, corpus, dict, mds='mmds')
#     #pyLDAvis.display(lda_data)
#     pyLDAvis.save_html(lda_data, lda_out_file)
#     print ('Topic visual saved\n')

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

app.layout = html.Div([
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Disclaimer", active=True, href="#")),
            dbc.NavItem(dbc.NavLink("Data Access", href="#")),
            dbc.NavItem(dbc.NavLink("Contact", href="#")),
            # dbc.NavItem(dbc.NavLink("Disabled", disabled=True, href="#")),
            # dbc.DropdownMenu(
            #     [dbc.DropdownMenuItem("Item 1"),
            #      dbc.DropdownMenuItem("Item 2")
            #     ],label="Dropdown", nav=True,
            # ),
        ], style={'float': 'right', 'marging':'100px'}),
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Data Summary',
                value='tab-1',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        dcc.Graph(
                            id='subjectivity_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x':sub_label_df['subj_label'],
                                    'y': sub_label_df['percent'],
                                    'type': 'bar',

                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Percentage of tweets'},
                                        xaxis = {'title': 'Subjectivity'},
                                )}),
                        dcc.Graph(
                            id='polarity_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x':polarity_label_df['polarity_label'],
                                    'y': polarity_label_df['percent'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Percentage of tweets'},
                                        xaxis = {'title': 'Polarity'},
                                )}),
                        dcc.Graph(
                            id='devices_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x': device_df['Source'],
                                    'y': device_df['percent'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Percentage of tweets'},
                                        xaxis = {'title': 'Device'},
                                )}),
                        dcc.Graph(
                            id='tweet_location_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x': locations['loc'],
                                    'y': locations['Count'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Number of tweets'},
                                        xaxis = {'title': 'State'},
                                )}),
                        dcc.Graph(
                            id='polarity_location_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x': sentiment_df['loc'],
                                    'y': sentiment_df['polarity'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Overall polarity'},
                                        xaxis = {'title': 'State'},
                                )}),
                        dcc.Graph(
                            id='sub_location_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x': sentiment_df['loc'],
                                    'y': sentiment_df['subjectivity'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Overall subjectivity'},
                                        xaxis = {'title': 'State'},
                                )}),
                        dcc.Graph(
                            id='ver_location_bar',
                            #style=dict(width='60'),
                            #config=dict(staticPlot=True),
                            figure={
                                'data': [{
                                    'x': verif_df['loc'],
                                    'y': verif_df['sum'],
                                    'type': 'bar'
                                }],
                                'layout' : go.Layout(
                                        yaxis = {'title': 'Number of verified accounts'},
                                        xaxis = {'title': 'State'},
                                )})], style=dict( width="1500px", height="790px", marginTop='20px'))]),
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
                        html.A(html.Button("Visualize Topics"), href="/lda_vis", target="_blank"),
                        html.P('How many topics to model?', style=dict(marginLeft=20, marginTop=10)),
                        dcc.Slider(
                            id='number_topics',
                            min=5,
                            max=30,
                            value=10,
                            step=1,
                            marks={i: str(i) for i in range(5, 31)}),
                            # )], style={'marginLeft':'50px', "marginBottom":"5px",'width':'100%'}),
                        html.Button(children='Run Model', n_clicks = 0, id='submit-model',style={'marginLeft':'20px', "marginBottom":"5px"}),
                        html.Div(id='tp_wd_output')
                    ], style=dict( width="1500px", height="790px"))]),
        dcc.Tab(label='Sentiments',
                value='tab-3',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        dcc.Dropdown(
                            id='sentiment_dropdown',
                            options=[{'label': i, 'value': i} for i in ['USA', 'France']],
                            value='USA',
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
                            style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}
                        ),
                        html.Div(id='user_mapping_output')], style=dict( width="1500px", height="790px"))]),
        dcc.Tab(label='Words Cloud',
                value='tab-5',
                #style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                width=700,
                                height=700,
                                style=dict(float='left', marginLeft=500)
                                )
                    ], style=dict( width="1500px", height="790px", marginTop='20px'))])
    ],  vertical=True, parent_style={'float': 'left', 'marginLeft':'5px', 'marginTop':90, "position":'absolute'}),

])

@app.server.route("/lda_vis")
def get_report():
    #lda_out_file = 'lda.html'
    return flask.send_from_directory(Path('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/'), 'lda.html')

@app.callback(
    Output(component_id="tp_wd_output", component_property="children"),
    [Input(component_id='submit-model', component_property='n_clicks')],
    [State(component_id='number_topics', component_property='value')])
def topics_button(n_clicks, n_topics):
    if (n_clicks >0 ):
        try:
            print('Start LDA model training ...\n')
            html.P('LDA model training')
            lda_model = models.ldamodel.LdaModel(corpus=corpus,
                                                 num_topics=n_topics,
                                                 id2word=loaded_dict,
                                                 passes=NUM_PASSES,
                                                 alpha=ALPHA,
                                                 eta=ETA,
                                                 random_state=49)
            lda_data =  pyLDAvis.gensim.prepare(lda_model, corpus, loaded_dict, mds='mmds')
            #pyLDAvis.display(lda_data)
            pyLDAvis.save_html(lda_data, lda_out_file)
            print ('Topic visual saved\n')
            #vis_topics(lda_model, corpus, loaded_dict)
            n_clicks = 0


            pyLDAvis.show(lda_data,port=8888)


            return html.Iframe(src=app.get_asset_url(lda_out_file),
                        style=dict( width="800px", height="800px", marginLeft=60))
        except Exception as e:
            return 'Error {} occured'.format(e)
# lda_out_file = 'lda.html'
# lda_html_file = 'lda.html'

# @app.callback(
#         Output(component_id="tp_wd_output", component_property="children"),
#         [Input(component_id="tp_wd", component_property="value")]
# )
# def tp_wd_dropdown(choice):
#     if choice == 'Word Cloud':
#         return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
#                 width=700,
#                 height=700,
#                 style=dict(float='left', marginLeft=500, marginTop=50)
#                 )
#     else:
#         return html.Iframe(src=app.get_asset_url(lda_html_file),
#                     style=dict( width="800px", height="800px", marginLeft=60)
#                 )

@app.callback(
    Output(component_id="sentiment_output", component_property="children"),
    [Input(component_id='sentiment_dropdown', component_property="value")]
)
def country_dropdown(country):
    if country == 'France':
        return html.P('No data to display for the moment. We are sorry!', style=dict(marginLeft=60, marginTop=10, color='red'))
    else:
        return dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id='polarity',
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
                            id='subjectivity',
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
                        ))))])
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
    app.run_server(debug=True)
