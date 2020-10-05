import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import dash_dangerously_set_inner_html
#import dash_auth
import json
import plotly.graph_objs as go
from io import BytesIO
from wordcloud import WordCloud
import base64


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
wordcloud = '/home/ulrich/Documents/dash-plotly/Masters2/dhifli/wordcloud.png' # replace with your own image
locations = pd.read_csv('locations.csv')
sentiment_df = pd.read_csv('sentiment.csv')
verif_df = pd.read_csv('verified_user.csv')
device_df = pd.read_csv('/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/device_spark.csv')
polarity_label_df = pd.read_csv("/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/polarity_label.csv")
sub_label_df = pd.read_csv("/home/ulrich/Documents/dash-plotly/Masters2/dhifli/agg/subj_label.csv")
encoded_image = base64.b64encode(open(wordcloud, 'rb').read())
lda_html_file = 'lda.html'

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
                        dcc.Dropdown(
                            id='tp_wd',
                            options=[{'label': i, 'value': i} for i in ['Word Cloud', 'Topics Analysis']],
                            value='Topics Analysis',
                            style={'marginLeft':'30px', "marginBottom":"5px",'width':'100%'}
                        ),
                        html.Div(id='tp_wd_output'),
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

@app.callback(
        Output(component_id="tp_wd_output", component_property="children"),
        [Input(component_id="tp_wd", component_property="value")]
)
def tp_wd_dropdown(choice):
    if choice == 'Word Cloud':
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                width=700,
                height=700,
                style=dict(float='left', marginLeft=500, marginTop=50)
                )
    else:
        return html.Iframe(src=app.get_asset_url(lda_html_file),
                    style=dict( width="1500px", height="800px", marginLeft=60)
                )
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
