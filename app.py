# -*- coding: utf-8 -*-
"""
Code for Schoolwijzer dashboard

author: Amor Frans
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd

from plotly import graph_objs as go
from plotly.graph_objs import Data
from dash.dependencies import Input, Output, State, Event
from math import radians, cos, sin, asin, sqrt
from random import randint

from sklearn.neighbors import KDTree
from sklearn import preprocessing
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
#######################################################################################

# app server
app = dash.Dash(__name__)
server = app.server
app.title = 'Schoolwijzer'

# API keys and datasets
mapbox_access_token = 'pk.eyJ1IjoiYW1vcnBmciIsImEiOiJjamx6NDVsNGUxc2M3M2tsODExOHNwbnZwIn0.POAeZ_S8K33WRnb1Wiikbw'
map_data = pd.read_excel("data_demo4a.xlsx")
map_data = map_data.sort_values(by='Gemeente')
postcodetabel = pd.read_excel("postcodetabel.xlsx")

# Boostrap / CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
app.css.append_css({'external_url': 'https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css'})
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

########################################################################################
#  Layouts

# layout table
layout_table = dict(
    autosize=True,
    #height=480,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=5,
        r=5,
        b=5,
        t=5
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
)    
layout_table['font-size'] = '12'
layout_table['margin-top'] = '20'


# tabs style
tab_style = {
    'color': 'white',
    #'backgroundColor': '#345E98'
    'backgroundColor': '#2a3f5f'
}

# layout kieshulp table


#########################################################################################
# functions
def gen_map(map_data, center, zoom):
    """
    Generates map based on map_data, center and zoom 
    """
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(map_data['Latitude']),
                "lon": list(map_data['Longitude']),
                "hoverinfo": "text",
                "hovertext": [["School: {} <br>Gemeente: {} <br>Aantal leerlingen: {}".format(i,j,k)]
                                for i,j,k in zip(map_data['Schoolnaam'], map_data['Gemeente'],map_data['Aantal leerlingen'])],
                "mode": "markers",
                "name": list(map_data['Schoolnaam']),
                "marker": {
                    "size": 10,
                    #"size": 40,
                    "opacity": 0.7
                }
        }],
        "layout": dict(
                                    autosize=True,
                                    height=480,
                                    font=dict(color="#191A1A"),
                                    titlefont=dict(color="#191A1A", size='14'),
                                    margin=dict(
                                        l=5,
                                        r=5,
                                        b=5,
                                        t=5
                                    ),
                                    hovermode="closest",
                                    plot_bgcolor='#fffcfc',
                                    #paper_bgcolor='#fffcfc',
                                    legend=dict(font=dict(size=10), orientation='h'),
                                    #title='Scholen',
                                    mapbox=dict(
                                        accesstoken=mapbox_access_token,
                                        style="light",
                                        center=center,
                                        zoom=zoom,
                                        pitch = 3.0,
                                        bearing=0
                                    )
                                )
    }

def gen_omgeving(map_data, center, zoom, factor):
    """
    Generates map based on map_data, center and zoom 
    """
    hover_string = "School: {} <br>Gemeente: {} <br>" + factor
    hover_string = hover_string + ": {}"
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(map_data['Latitude']),
                "lon": list(map_data['Longitude']),
                "hoverinfo": "text",
                "hovertext": [[hover_string.format(i,j,k)]
                                for i,j,k in zip(map_data['Schoolnaam'], map_data['Gemeente'],map_data[factor])],
                "mode": "markers",
                "name": list(map_data['Schoolnaam']),
                "marker": {
                    #"size": 6,
                    "size": 50,
                    "opacity": 0.4,
                    "cmax":map_data[factor].max(),
                    "cmin":map_data[factor].min()-1,
                    "color":map_data[factor].tolist(),
                    "colorbar":dict(
                        title=factor,
                        x= 0.01
                    ),
                    "colorscale":'Viridis'
          
                    }
        }],
        "layout": dict(
                                    autosize=True,
                                    #title = "Omgevingsfactoren",
                                    #width = 200,
                                    height=500,
                                    font=dict(color="#191A1A"),
                                    titlefont=dict(color="#191A1A", size='14'),
                                    margin=dict(
                                        l=5,
                                        r=5,
                                        b=5,
                                        t=5
                                    ),
                                    hovermode="closest",
                                    plot_bgcolor='#fffcfc',
                                    #paper_bgcolor='#fffcfc',
                                    legend=dict(font=dict(size=10), orientation='h'),
                                    #title='Scholen',
                                    mapbox=dict(
                                        accesstoken=mapbox_access_token,
                                        style="light",
                                        center=center,
                                        zoom=zoom,
                                        pitch = 3.0,
                                        bearing=0
                                    )
                                )
    }

      
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def lat(postalcode):
    """
    Convert postalcode to latitude
    """
    row = postcodetabel.loc[postcodetabel.PostCode == postalcode]
    if len(row)>0:
        return row['Latitude'].tolist()[0]
    else:
        return 0

def lon(postalcode):
    """
    Convert postalcode to longitude
    """
    row = postcodetabel.loc[postcodetabel.PostCode == postalcode]
    if len(row)>0:
        return row['Longitude'].tolist()[0]
    else:
        return 0
    
def check_range(lat_user, lon_user, lat_test, lon_test, radius):
    """
    Check whether coordinate of user falls into range given a coordinate and radius
    """
    distance = haversine(lon_user,lat_user, lon_test, lat_test)
    if distance <= radius:
        return True
    else:
        return False    

def layout_stacked_bar(title, color):
    """
    Function that creates the layout for the male/female plot
    """
    layout = go.Layout(
        title= title,
        bargap=0.25,
        bargroupgap=0.0,
        barmode='stack',
        showlegend=True,
        dragmode="select",
        xaxis=dict(
            showgrid=False,
            #nticks=50,
            fixedrange=False
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=False,
            fixedrange=False,
            rangemode='nonnegative',
            #zeroline='hidden'
        ),
        margin=go.Margin(
            l=620,
            r=50
        ),
        paper_bgcolor=color,
        plot_bgcolor=color,
    )
    return layout

def stacked_bar(x1,x2,y,title_x1, title_x2, layout): 
    """
    Creates the stacked bar plot used for the male/female plot, but can be used for other binary plots
    """
    trace1 = go.Bar(
            y= y,
            x= x1,
            name=title_x1,        
            text=x1,
            textposition = 'auto',
            orientation = 'h',
            opacity=0.9,
            marker = dict(
                color = 'rgba(246, 78, 139, 0.5)',
                line = dict(
                    color = 'rgba(246, 78, 139, 0.5)',
                    width = 1)
            )
    )
    trace2 = go.Bar(
            y=y,
            x=x2,
            name=title_x2,        
            text=x2,
            textposition = 'auto',
            orientation = 'h',
            opacity=0.9,
            marker = dict(
                color = 'rgba(158,202,225, 0.5)',
                line = dict(
                    color = 'rgba(158,202,225, 0.5)',
                    width = 1)
            )
    )
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    return fig

def layout_dotplot(title, color):
    """
    Creates the layout for the dotplot
    """
    layout = go.Layout(
        title=title,
        xaxis=dict(
            #title = 'Percentage in (%)',
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            titlefont=dict(
                color='black'
            ),
            tickfont=dict(
                color='rgb(102, 102, 102)',
            ),
            showticklabels=True,
            dtick=5,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
        ),
        yaxis = dict(showline=False),
        margin=dict(
            l=620,
            r=40,
            b=50,
            t=80
        ),
        legend=dict(orientation="h",
            font=dict(
                size=10,
            ),x=0.05, y=1.1
                    
        ),
        #width=800,
        #height=600,
        paper_bgcolor=color,
        plot_bgcolor=color,
        hovermode='closest',
    )
    return layout

def create_dotplot(feature_list, scholen, df, colors):
    """
    Creates dotplot based on the feature list, names of schools, dataframe and a list of colors
    """
    data = []
    for feature in feature_list:

        random_num = randint(0, len(colors)-1)
        color = colors[random_num]
        del colors[random_num]
        tmp = df[feature].tolist()
        trace = go.Scatter(
        x= tmp,
        y=scholen,
        mode='markers',
        name=feature,
        marker=dict(
            color=color,
            line=dict(
                color=color,
                width=1,
            ),
            symbol='circle',
            size=16,
         )
        )
        data.append(trace)
    return data


def get_ranking(input_vector,data, name):
    """
    Get similarity ranking given input vector and a featureset
    """
    # drop unnneeded columns
    #cols = [c for c in data.columns if 'distance' in c.lower() or 'rank' in c.lower()]
    
    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train= min_max_scaler.fit_transform(data.values)
    X_test = min_max_scaler.transform(np.array([input_vector.values]))
    
    # compute similarity
    output = cosine_similarity(X_test, X_train)
    data[name] = output[0]
    return data
    
# function that can tune the input feature vector
def tune_vector(column,plus_min, data_input, data_total):
    """
    Function that can tune the input feature vector based on preferences
    param column: features to be tuned
    param plus_min: whether the feauture has a positive or negative impace
    param data_input: input vector
    param data_total: total data
    """
    i=0
    for i in range(0, len(column)):
        if plus_min[i] == -1:
            data_input[column[i]] = data_total[column[i]].mean() -  data_total[column[i]].std()
            if data_input[column[i]] < 0:
                data_input[column[i]] =0
        elif plus_min[i] == 1:
            data_input[column[i]] = data_total[column[i]].mean() +  data_total[column[i]].std()
        else:
            data_input[column[i]] = data_total[column[i]].mean()
        
    return data_input

def convert_preferenes(preferences):
    """
    Convert clicked preferences/profiles into vectors to get the ranking
    """
    # init results
    i=0
    plus_min = []
    features = []
    
    # loop trhough clicked preferences
    for i in range(0,len(preferences)):
        voorkeur = preferences[i]
        
        if voorkeur == 'Kleine school':
            plus_min.append([-1,-1])
            features.append(["Aantal leerlingen","Aantal Personeel"])
        elif voorkeur == 'Grote school':
            plus_min.append([1,1])
            features.append(["Aantal leerlingen","Aantal Personeel"])
        elif voorkeur == 'Goede slagingspercentages':
            plus_min.append([1])
            features.append(["Slagingspercentage 2017"])
        elif voorkeur == 'Veel meisjes':
            plus_min.append([-1])
            features.append(["Percentage man"])
        elif voorkeur == 'Oudere leraren':
            plus_min.append([1])
            features.append(["Gem. Leeftijd personeel"])
        elif voorkeur == 'Veel kinderen in de omgeving':
            plus_min.append([1,1,1])
            features.append(['Percentage huishoudens met kinderen', 'Percentage leeftijd 0-15', 'Percentage leeftijd 15-25'])   
        elif voorkeur == 'Stedelijke omgeving':
            plus_min.append([1])
            features.append(["MateVanStedelijkheid"])
        else:
            "ERRROR"
    
    # merge lists in list
    features = [item for sublist in features for item in sublist]
    plus_min = [item for sublist in plus_min for item in sublist]
    
    return features, plus_min
        
            
            
        
#########################################################################################
# website html layout
app.layout = html.Div(
    html.Div([
            
        # header    
        html.Div([

            html.Span("Schoolwijzer", className='app-title', style={'font': 'Paladino'}),
            html.Span("           ", className='app-title',style={'fontSize' : 44, 'color' : 'silver', 'align' : 'right'}),
            html.Span("   Het online platform voor uw schoolkeuze", className='app-title', style={'fontSize' : 21, 'color' : 'silver', 'align' : 'right'}),
            html.Div(
                html.Img(src='https://i.postimg.cc/VNL01mC2/hippie-clipart-owl-6.png',height="100%")
                ,style={"float":"right","height":"100%"})
            ] ,className="row header"
        ),      

         # Selectors
        html.Div(
            [html.Div(id='global_postcode', style={'display': 'none'}),
                # level 1
                html.Div(
                [
                    # postcode
                    html.Div(
                        [
                            html.P('Postcode:'),
                                dcc.Input(id='postcode', type='text', placeholder = 'Typ postcode: 1234AB'),
                        ],
                        className='three columns',
                        style={'margin-top': '10'}
                    ),

                    # radius
                    html.Div(
                            [
                                html.P('Radius:'),
                                
                                    dcc.Slider(
                                        id= 'radius',
                                        min=0,
                                        max=25,
                                        step=0.5,
                                        value=5,
                                        marks={
                                            #0: '0 km',
                                            #25: '25 km'
                                        },
                                    ),
                                    html.Div(id='radius_output', style={'color': 'grey', 'fontSize': 12})
                            ],
                            className='three columns',
                            style={'margin-top': '10'}
                    ),       
                 ] ,style={'margin-top': '10'}
                ),
                        
                        

                # level 2
                html.Div(
                    [

                        
                        # niveau
                        html.Div(
                                [
                                html.P('Kies Niveau:'),
                                    dcc.Dropdown(
                                            id = 'niveau',
                                            multi=False,
                                            options=[{'label': 'PRO','value':'PRO'} ,
                                                      {'label': 'VBO', 'value':'VBO'},
                                                      {'label':  'MAVO', 'value':'MAVO'} ,
                                                      {'label' :'HAVO', 'value': 'HAVO'},
                                                      {'label': 'VWO', 'value':'VWO'}],
                                            #value= "HAVO",
                                            placeholder="Selecteer niveau")
                                            #labelStyle={'display': 'inline-block'}
                                    ],
                                className='three columns',
                                style={'margin-top': '10', 'margin-left': '80'}
                            ),                   
                        
                        #geloof
                        html.Div(
                            [                      
                                                     
                                html.P('Soort school:'),
                                dcc.Dropdown(
                                        id = 'geloof',
                                        multi=False,
                                        options=[{'label': 'Geen voorkeur', 'value':'Geen voorkeur'},
                                                      {'label': 'Gereformeerd','value':'Gereformeerd'} ,
                                                      {'label': 'Islamitisch', 'value':'Islamitisch'},
                                                      {'label':  'Joods orthodox', 'value':'Joods orthodox'} ,
                                                      {'label' :'Openbaar', 'value': 'Openbaar'},
                                                      {'label': 'Overige', 'value':'Overige'},
                                                      {'label': 'Protestants Christelijk', 'value':'Protestants Christelijk'},
                                                      {'label': 'Reformatorisch', 'value':'Reformatorisch'},
                                                      {'label': 'Rooms-Katholiek', 'value':'Rooms-Katholiek'},
                                                      {'label': 'Samenwerking Protestants-Katholiek', 'value':'Samenwerking Protestants-Katholiek'}], 
                                        #value= ["Geen voorkeur"],
                                        placeholder = "Selecteer denominatie school"
                                        #labelStyle={'display': 'inline-block'}
                                )
                            ],
                            className='three columns',
                            style={'margin-top': '10'}
                        ),                        
                    ],  style={'margin-top': '10', 'margin-left': '20'},
                    #className="six columns"
                )              
                    
                
            ],
            className='row', style={'margin-bottom': '20'}
        ),

        # Map + table 
        html.Div(
            [
                    
                # map
                html.Div(
                    [
                        dcc.Graph(id='map-graph',
                                  animate=True,
                                  style={'margin-top': '20'})
                    ], className = "six columns" #, style={'margin-right': '5'}
                ),
                    
                # table
                html.Div(
                    [
                        html.P('Selecteer hier scholen die u wilt vergelijken', style={'fontSize' : 19, 'textAlign': 'right'}),
                        dcc.RadioItems(
                            id = 'data_keuze',
                            options=[
                                {'label': 'Beknopt', 'value': 'Beknopt'},
                                {'label': 'Alle data', 'value': 'Alle data'},

                            ],
                            value='Beknopt',
                            labelStyle={'display': 'inline-block'},
                            style = {'textAlign': 'right'}
                        ), 
                        
                        # non-visivbe dt
                        dt.DataTable(
                            rows=map_data.to_dict('records'),
                            #style={'display': 'none'},
                            #rows  = [{}],
                            columns=['Schoolnaam', 'Niveau', 'Gemeente', 'Geloofsovertuiging'],
                            row_selectable=True,
                            #filterable=True,
                            sortable=True,
                            selected_row_indices=[],
                            max_rows_in_viewport = 10,
                            #title= "Selecteer scholen die uw wilt vergelijken",
                            id='datatable'),#'textAlign': 'left'

                    
                        html.P('De tabel is sorteerbaar en filterbaar per variabele', style={'fontSize' : 12, 'color' : ' #5d6d7e','textAlign': 'right'}),
                                                                                                 
                    ],
                            
                    style = layout_table,
                    className="six columns"#, style={'margin-left': '10'}
                ),
                            
                # tabs
                html.Div([
                        
                        dcc.Tabs(id="tabs", children=[
                                
                            # kieshulp
                            dcc.Tab(label='Kieshulp', children=[
                                html.Div([
                                        html.P('Welkom bij de Kieshulp! Klik hieronder uw preferenties aan en de kieshulp berekent hoeveel procent match uw heeft met de geselecteerde scholen. ', style={'fontSize' : 18}),
                                        dcc.Dropdown(
                                        id = 'preferenties',
                                        multi=True,
                                        options=[{'label': 'Kleine school','value': 'Kleine school'},
                                                 {'label': 'Grote school','value': 'Grote school'},
                                                 {'label': 'Goede slagingspercentages','value': 'Goede slagingspercentages'},
                                                 {'label': 'Veel meisjes','value': 'Veel meisjes'},
                                                 {'label': 'Oudere leraren','value': 'Oudere leraren'},
                                                 {'label': 'Veel kinderen in de omgeving','value': 'Veel kinderen in de omgeving'},
                                                 {'label': 'Stedelijke omgeving','value': 'Stedelijke omgeving'}
                                                 ],
                                        value= 'Goede slagingspercentages',
                                        #labelStyle={'display': 'inline-block'}
                                        ),dcc.Graph( id='figure_d1')
    
                                ],className= 'twelve columns' )
                            ], style=tab_style, selected_style=tab_style),
                                 
                            # leerlingen
                            dcc.Tab(label='Leerlingen', children=[
                                   
                                    html.Div([
                                        dcc.Graph( id='bar-graph'),
                                        dcc.Graph( id='figure_a2'),
                                        dcc.Graph( id='figure_a3')
                                    ])
                            ]),
                                        
                            # Personeel
                            dcc.Tab(label='Personeel', children=[
                                html.Div([
                                   dcc.Graph( id='figure_b1'),
                                   dcc.Graph( id='figure_b2')
                                   
                                ])
                            ]),
                                        
                            # Omgeving
                            dcc.Tab(label='Omgeving', children=[
                                html.Div([
                                    dcc.Graph(id='figure_c1',animate=False),
                                ], className = "nine columns",
                                style={'margin-top': '20'}),
                                
                            # Omgevingsfactoren
                            html.Div(
                            [                  
                                                     
                                html.P('Omgevingsfactor:'),
                                dcc.RadioItems(
                                        id = 'omgeving',
                                        #multi=False,
                                        options=[{'label': 'Statusscore','value': 'Statusscore'},
                                                 {'label': '% huishoudens met kinderen','value': 'Percentage huishoudens met kinderen'},
                                                 {'label': '% personen met bijstand','value': 'Percentage personen met bijstand'},
                                                 {'label': '% personen in leeftijd 0-15','value': 'Percentage leeftijd 0-15'},
                                                 {'label': '% personen in leeftijd 15-25','value': 'Percentage leeftijd 15-25'},
                                                 {'label': '% personen niet-westerse allochtonen','value': 'Percentage niet-westers'},
                                                 {'label': 'Aantal vernielingen en misdrijven','value': 'VernielingMisdrijfTegenOpenbareOrde'},
                                                 {'label': 'Aantal gewelds en seksuele misdrijven','value': 'GeweldsEnSeksueleMisdrijven'},
                                                 {'label': 'Bevolkingsdichtheid','value': 'Bevolkingsdichtheid'},
                                                 {'label': 'Gemiddelde woningwaarde (in €)','value': 'GemiddeldeWoningwaarde'},
                                                 {'label': 'Mate van stedelijkheid','value': 'MateVanStedelijkheid'},],
                                       #options= [{'label': str(item),
                                        #                  'value': str(item)}
                                        #                 for item in set(map_data['Niveau'])],
                                        value= "Statusscore",
                                        #labelStyle={'display': 'inline-block'}
                                ),
                                dcc.Markdown(id = 'uitleg' ,containerProps = {'style':{'fontSize' : 12, 'color' : '#5d6d7e', 'align' : 'right'}})
                            ],
                            className='three columns',
                            style={'margin-top': '20'}
                            )
                                
                            ]),

                        ])

                    ], className= 'twelve columns', style={'margin-top': '20'}
                    ),
                    

            ], className="row"
        )
    ], 
    className='ten columns offset-by-one'))
    #className='twelve columns'))
##############################################################################

# callbacks
    
# hidden global variable
@app.callback(Output('global_postcode', 'children'), [Input('postcode', 'value')])
def get_postcode(value):
    value = value.replace(" ", "")
    value = value.upper()
    return value

# radius
@app.callback(
    dash.dependencies.Output('radius_output', 'children'),
    [dash.dependencies.Input('radius', 'value')])
def update_output(value):
    return 'Maximale afstand {} km'.format(value)
 
# data table
@app.callback(
    Output('datatable', 'rows'),
    [Input('radius', 'value'),
     Input('global_postcode', 'children'),
     Input('niveau', 'value'),
     Input('geloof', 'value')])                                 
def update_selected_row_indices(radius, postcode, niveau, geloof):
    # combined niveaus also possible
    if niveau is None:
        niveau = "HAVO"
        niveau = [x for x in map_data['Niveau'].unique().tolist() if niveau in x]
        rows = map_data.to_dict('records')
        return rows
    else:
        niveau = [x for x in map_data['Niveau'].unique().tolist() if niveau in x]
    map_aux = map_data.copy()
    map_aux = map_aux[map_aux["Niveau"].isin(niveau)]
        
    # No preference geloof
    if geloof == "Geen voorkeur":
        map_aux = map_aux
    else:
        map_aux = map_aux.loc[map_aux["Geloofsovertuiging"] == geloof]
            
    # init radius data
    radius = radius # in kilometer
    result = []
    postal_input = postcode
    lon_user = lon(postal_input)
    lat_user = lat(postal_input)
    lats = map_aux['Latitude'].tolist()
    lons = map_aux['Longitude'].tolist()
        
    # Get all schools within radius
    for i in range(0,len(lats)):
        result.append(check_range(lat_user, lon_user, lats[i], lons[i], radius))
        
    # select the rows within the range
    map_aux = map_aux[result] 
           
    # convert to dict
    rows = map_aux.to_dict('records')
    return rows

# datatable columns, bekonopt/all data
@app.callback(
    Output('datatable', 'columns'),
    [Input('data_keuze', 'value'),
     Input('datatable', 'rows')])                                 
def update_selected_row_indices1(data_keuze, rows):
    if data_keuze == 'Beknopt':
        columns = ['Schoolnaam', 'Niveau', 'Gemeente', 'Geloofsovertuiging']
    else:
        columns = map_data.columns.tolist()
        
    return columns

# map
@app.callback(
    Output('map-graph', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])       
def map_selection(rows, indices):
    if (len(rows) == len(map_data)) and (len(indices) ==0):
        lats = map_data['Latitude'].tolist()
        longs = map_data['Longitude'].tolist()
        lats_center = sum(lats)/len(lats)
        longs_center = sum(longs)/len(longs)
        center = dict(lon=longs_center, lat=lats_center)
        return gen_map(map_data, center, 6.1)
    elif (len(rows) == len(map_data)) :
        aux = pd.DataFrame(rows)
        aux = aux.iloc[indices]
        lats = aux['Latitude'].tolist()
        longs = aux['Longitude'].tolist()
        lats_center = sum(lats)/len(lats)
        longs_center = sum(longs)/len(longs)
        center = dict(lon=longs_center, lat=lats_center)
        return gen_map(aux, center, 6.1)        
    else:
        aux = pd.DataFrame(rows)
        aux = aux.iloc[indices]
        lats = aux['Latitude'].tolist()
        longs = aux['Longitude'].tolist()
        lats_center = sum(lats)/len(lats)
        longs_center = sum(longs)/len(longs)
        center = dict(lon=longs_center, lat=lats_center)
        return gen_map(aux, center, 10.5)


# figures
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def figure_a1(rows, selected_row_indices):
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices]
    scholen = df['Schoolnaam'].tolist()
    vrouw = 1-(df['Percentage man']/100)
    vrouw = df['Aantal leerlingen']*vrouw
    vrouw=vrouw.astype(int)
    man = df['Aantal leerlingen']*(df['Percentage man']/100)
    man = man.astype(int)
    layout = layout_stacked_bar("Aantal leerlingen", "rgba(255, 255, 255, 1)")
    fig = stacked_bar(vrouw,man,scholen,'Meisjes','Jongens', layout)
    return fig
    
   
@app.callback(
    Output('figure_a2', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def figure_a2(rows, selected_row_indices):
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices]    
    scholen = df['Schoolnaam'].tolist()
    colors = ['rgba(156, 165, 196, 0.95)', 'rgba(247, 156, 76, 0.95)','rgba(255, 0, 0, 0.5)','rgba(255, 0, 0, 0.4)', 'rgba(204, 204, 204, 0.95)','rgba(16, 166, 150, 0.95)']
    data = create_dotplot(['Percentage zittenblijvers', 'Percentage opstromers', 'Percentage afstromers', 'Slagingspercentage 2017'], scholen, df, colors)
    layout= layout_dotplot("Performance leerlingen",'rgb(240, 240, 240)' )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output('figure_a3', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def figure_a3(rows, selected_row_indices):
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices] 
    scholen = df['Schoolnaam'].tolist()
    colors = ['rgba(156, 165, 196, 0.95)', 'rgba(247, 156, 76, 0.95)','rgba(255, 0, 0, 0.5)','rgba(255, 0, 0, 0.4)', 'rgba(204, 204, 204, 0.95)','rgba(16, 166, 150, 0.95)']
    data = create_dotplot(['Gem. Cijfer schoolexamen', 'Gem. Cijfer centraal examen', 'Gem. Cijfer cijferlijst'], scholen, df, colors)
    layout= layout_dotplot("Cijfers leerlingen",'rgb(220, 220, 220)' )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output('figure_b1', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def figure_b1(rows, selected_row_indices):
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices]
    scholen = df['Schoolnaam'].tolist()
    man = df['Percentage man personeel']/100
    vrouw = 1-man
    vrouw = df['Aantal Personeel']*vrouw
    vrouw=vrouw.astype(int)
    man = df['Aantal Personeel']*man
    man = man.astype(int)
    layout = layout_stacked_bar("Aantal personeel", "rgba(255, 255, 255, 1)")
    fig = stacked_bar(vrouw,man,scholen,'Vrouwen','Mannen', layout)
    return fig

@app.callback(
    Output('figure_b2', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def figure_b2(rows, selected_row_indices):
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices] 
    scholen = df['Schoolnaam'].tolist()
    colors = ['rgba(156, 165, 196, 0.95)', 'rgba(247, 156, 76, 0.95)','rgba(255, 0, 0, 0.5)','rgba(255, 0, 0, 0.4)', 'rgba(204, 204, 204, 0.95)','rgba(16, 166, 150, 0.95)']
    data = create_dotplot(['Aantal leerlingen per personeel', 'Gem. Leeftijd personeel', 'Gemiddelde FTE personeel', 'Percentage vaste dienst'], scholen, df, colors)
    layout= layout_dotplot("Cijfers personeel",'rgb(240, 240, 240)' )
    fig = go.Figure(data=data, layout=layout)
    return fig


# map
@app.callback(
    Output('figure_c1', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices'),
     Input('omgeving', 'value')])       
def figure_c1(rows, selected_row_indices, omgevingsfactor ):
    aux = pd.DataFrame(rows)
    aux = aux.iloc[selected_row_indices] 
    lats = aux['Latitude'].tolist()
    longs = aux['Longitude'].tolist()
    lats_center = sum(lats)/len(lats)
    longs_center = sum(longs)/len(longs)
    center = dict(lon=longs_center, lat=lats_center)
    return gen_omgeving(aux, center, 11.5, omgevingsfactor)

@app.callback(
    Output('uitleg', 'children'),
    [Input('omgeving', 'value')])       
def tekst_uitleg( omgevingsfactor ):
    if omgevingsfactor == 'Statusscore':
        text = "*Statusscores zijn scores die het SCP berekent en die aangeven hoe de sociale status van een wijk is, in vergelijking met andere wijken in Nederland. Met de sociale status bedoelen we hier niet het aanzien of de populariteit van een wijk. De sociale status van een wijk is afgeleid van een aantal kenmerken van de mensen die er wonen: hun opleiding, inkomen en positie op de arbeidsmarkt. "
    elif omgevingsfactor == 'Bevolkingsdichtheid':
        text = "*Bevolking op 1 januari gedeeld door de oppervlakte land in km²"
    elif omgevingsfactor == 'MateVanStedelijkheid':
        text = "*Een maatstaf voor de concentratie van menselijke activiteiten gebaseerd op de gemiddelde omgevingsadressendichtheid (oad). Hierbij zijn vijf categorieën onderscheiden: - (1) zeer sterk stedelijk: gemiddelde oad van 2500 of meer adressen per km2; -  (2) sterk stedelijk: gemiddelde oad van 1500 tot 2500 adressen per km2; - (3) matig stedelijk: gemiddelde oad van 1000 tot 1500 adressen per km2; - (4) weinig stedelijk: gemiddelde oad van 500 tot 1000 adressen per km2; - (5)  niet stedelijk: gemiddelde oad van minder dan 500 adressen per km2."
    else:
        text = ""
    return text


@app.callback(
    Output('figure_d1', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices'),
     Input('preferenties', 'value')])
def figure_d1(rows, selected_row_indices, voorkeuren):
    
    # make df
    df = pd.DataFrame(rows)
    df = df.iloc[selected_row_indices]
    
    # numeric data
    cols = ['Percentage man', 'Aantal Personeel', 'Gem. Leeftijd personeel',
       'Gemiddelde FTE personeel', 'Percentage vaste dienst',
       'Percentage man personeel', 'Aantal leerlingen',
       'Aantal leerlingen per personeel', 'Percentage zittenblijvers',
       'Percentage afstromers', 'Percentage opstromers',
       'Slagingspercentage 2017', 'Gem. Cijfer schoolexamen',
       'Gem. Cijfer centraal examen', 'Gem. Cijfer cijferlijst', 'Statusscore',
       'Percentage leeftijd 0-15', 'Percentage leeftijd 15-25',
       'Percentage niet-westers', 'Percentage huishoudens met kinderen',
       'Percentage personen met bijstand',
       'VernielingMisdrijfTegenOpenbareOrde', 'GeweldsEnSeksueleMisdrijven',
       'Bevolkingsdichtheid', 'GemiddeldeWoningwaarde',
       'MateVanStedelijkheid']
    df_numeric = df.loc[:,cols]
    
    # input vector
    input_vector = pd.Series()
    for x in df_numeric.columns:
        input_vector[x] = df_numeric[x].mean()
        
    # tune input
    if len(voorkeuren)>0:
        features, plus_min = convert_preferenes(voorkeuren)
        input_vector = tune_vector(features,plus_min, input_vector, df_numeric)
    
    # get ranking
    df_res = get_ranking(input_vector, df_numeric, 'Score Match')
    df['MatchScore'] = df_res['Score Match']
    df = df.sort_values(by= 'MatchScore')
    
    # graph input
    scholen = df['Schoolnaam'].tolist()
    ss = df['MatchScore']*100
    ss = ss.astype(int).tolist()
    teksts = [str(x) + "% match" for x in ss]
        
    # plot
    layout = go.Layout(
        xaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=False,
            zeroline=False,
            #title = "similarity in %"
            #domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            #zeroline=False,
            rangemode='nonnegative',
        ),
        barmode='stack',
        title = "Resultaten kieshulp",
        bargap=0.35,
        font=dict(family= 'Open Sans,sans-serif', size=14, color='#2a3f5f'),
        #dragmode = "select",
        paper_bgcolor='rgba(245, 246, 249, 1)',
        plot_bgcolor='rgba(245, 246, 249, 1)',
        margin=dict(
            l=620,
            r=10,
            t=70,
            b=80
        ),
        showlegend=False,
    )
    trace = go.Bar(
            y=scholen,
            x=ss,
            name="Name",        
            #text=ss,
            #textposition = 'auto',
            orientation = 'h',
            opacity=0.8,
            marker = dict(
                color = 'rgba(33, 49, 74, 0.7)',
                line = dict(
                    color = 'rgba(33, 49, 74, 1)',
                    width = 1)
            )
    )
    data = [trace]
    
    annotations = []

    for i in range(0, len(ss)):
        annotations.append(dict(x=ss[i]-8, y=scholen[i], text=teksts[i],
                                  font=dict(family='Open Sans,sans-serif', size=16,
                                  color='rgba(245, 246, 249, 1)'),
                                  showarrow=False,))
    layout['annotations'] = annotations
    
    fig = go.Figure(data=data, layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)