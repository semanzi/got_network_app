#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:42:31 2021

@author: sean
"""
import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import pandas as pd
import dash_cytoscape as cyto
import networkx as nx
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output

#cyto.load_extra_layouts()

#Graph creation code

nodes_1 = pd.read_csv('assets/data/got-s1-nodes.csv', low_memory=False)
edges_1 = pd.read_csv('assets/data/got-s1-edges.csv', low_memory=False)
nodes_2 = pd.read_csv('assets/data/got-s2-nodes.csv', low_memory=False)
edges_2 = pd.read_csv('assets/data/got-s2-edges.csv', low_memory=False)
nodes_3 = pd.read_csv('assets/data/got-s3-nodes.csv', low_memory=False)
edges_3 = pd.read_csv('assets/data/got-s3-edges.csv', low_memory=False)
nodes_4 = pd.read_csv('assets/data/got-s4-nodes.csv', low_memory=False)
edges_4 = pd.read_csv('assets/data/got-s4-edges.csv', low_memory=False)
nodes_5 = pd.read_csv('assets/data/got-s5-nodes.csv', low_memory=False)
edges_5 = pd.read_csv('assets/data/got-s5-edges.csv', low_memory=False)
nodes_6 = pd.read_csv('assets/data/got-s6-nodes.csv', low_memory=False)
edges_6 = pd.read_csv('assets/data/got-s6-edges.csv', low_memory=False)
nodes_7 = pd.read_csv('assets/data/got-s7-nodes.csv', low_memory=False)
edges_7 = pd.read_csv('assets/data/got-s7-edges.csv', low_memory=False)
nodes_8 = pd.read_csv('assets/data/got-s8-nodes.csv', low_memory=False)
edges_8 = pd.read_csv('assets/data/got-s8-edges.csv', low_memory=False)

def create_nx_graph(nodeData,edgeData):
    ## Initiate the graph object
    G = nx.Graph()
    
    ## Tranform the data into the correct format for use with NetworkX
    # Node tuples (ID, dict of attributes)
    idList = nodeData['Id'].tolist()
    labels =  pd.DataFrame(nodeData['Label'])
    labelDicts = labels.to_dict(orient='records')
    nodeTuples = [tuple(r) for r in zip(idList,labelDicts)]
    
    # Edge tuples (Source, Target, dict of attributes)
    sourceList = edgeData['Source'].tolist()
    targetList = edgeData['Target'].tolist()
    weights = pd.DataFrame(edgeData['Weight'])
    weightDicts = weights.to_dict(orient='records')
    edgeTuples = [tuple(r) for r in zip(sourceList,targetList,weightDicts)]
    
    ## Add the nodes and edges to the graph
    G.add_nodes_from(nodeTuples)
    G.add_edges_from(edgeTuples)
    
    return G

def create_analysis(G,nodes):
    #Graph metrics
    g_met_dict = dict()
    g_met_dict['num_chars'] = G.number_of_nodes()
    g_met_dict['num_inter'] = G.number_of_edges()
    g_met_dict['density'] = nx.density(G)
    
    #Node metrics
    e_cent = nx.eigenvector_centrality(G)
    page_rank = nx.pagerank(G)
    degree = nx.degree(G)
    between = nx.betweenness_centrality(G)
    
    # Extract the analysis output and convert to a suitable scale and format
    e_cent_size = pd.DataFrame.from_dict(e_cent, orient='index',
                                         columns=['cent_value'])
    e_cent_size.reset_index(drop=True, inplace=True)
    #e_cent_size = e_cent_size*100
    page_rank_size = pd.DataFrame.from_dict(page_rank, orient='index',
                                            columns=['rank_value'])
    page_rank_size.reset_index(drop=True, inplace=True)
    #page_rank_size = page_rank_size*1000
    degree_list = list(degree)
    degree_dict = dict(degree_list)
    degree_size = pd.DataFrame.from_dict(degree_dict, orient='index',
                                         columns=['deg_value'])
    degree_size.reset_index(drop=True, inplace=True)
    g_met_dict['avg_deg'] = degree_size.iloc[:,0].mean()
    between_size = pd.DataFrame.from_dict(between, orient='index',
                                          columns=['betw_value'])
    between_size.reset_index(drop=True, inplace=True)
    
    dfs = [e_cent_size,page_rank_size,degree_size,between_size]
    analysis_df = pd.concat(dfs, axis=1)
    cols = list(analysis_df.columns)
    an_arr = analysis_df.to_numpy(copy=True)
    scaler = StandardScaler()
    an_scaled = scaler.fit_transform(an_arr)
    an_df = pd.DataFrame(an_scaled)
    an_st = an_df.copy(deep=True)
    an_st.columns = cols
    an_df.columns = cols
    an_mins = list(an_df.min())
    for i in range(len(an_mins)):
        an_df[cols[i]] -= an_mins[i] - 1
        an_df[cols[i]] *= 6
    
    colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
               '#42d4f4','#4363d8','#911eb4','#f032e6']
    others = ['#c7c3b9']*(len(an_st)-10)
    all_colours = colours + others
    names = ['cent_col', 'rank_col', 'deg_col', 'betw_col']
    names_two = ['cent_lab', 'rank_lab', 'deg_lab', 'betw_lab']
    an_df['id'] = nodes['Id']
    an_df['lab'] = nodes['Label']
    top_id = pd.DataFrame()
    for j in range(len(names)):
        an_df.sort_values(cols[j], ascending=False, inplace=True)
        an_df[names[j]] = all_colours
        top = list(an_df.iloc[0:10,5])
        top_id[j] = list(an_df.iloc[0:10,4])
        blanks = ['']*(len(an_df)-10)
        labs = top + blanks
        an_df[names_two[j]] = labs
        an_df.sort_index(inplace=True)
    
    return an_df, an_st, top_id, g_met_dict

def edge_styling(edges, top_id):
    colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
               '#42d4f4','#4363d8','#911eb4','#f032e6']
    match_s = pd.DataFrame()
    match_t = pd.DataFrame()
    for i in range(len(top_id)):
        match_s[i] = np.where(edges['Source'] == top_id.iloc[i,0], colours[i],
                              '')
        match_t[i] = np.where(edges['Target'] == top_id.iloc[i,0], colours[i],
                              '')
    edges['col_source'] = match_s[0] + match_s[1] + match_s[2] + match_s[3] + \
                        match_s[4] + match_s[5] + match_s[6] + match_s[7] + \
                        match_s[8] + match_s[9]
    edges['col_target'] = match_t[0] + match_t[1] + match_t[2] + match_t[3] + \
                        match_t[4] + match_t[5] + match_t[6] + match_t[7] + \
                        match_t[8] + match_t[9]
    l = list()
    for i in range(len(edges)):
        if edges['col_source'][i] != '':
            l.append(edges['col_source'][i])
        elif edges['col_target'][i] != '':
            l.append(edges['col_target'][i])
        else:
            l.append('#c7c3b9')
    edges['col_all'] = l
    
    return edges

def create_barchart(an_adj,num_col,lab_col,col_col):
    an_sorted = an_adj.sort_values(num_col, ascending=False).copy(deep=True)
    an_top = an_sorted.iloc[0:10,:].copy(deep=True)
    an_top.sort_values(num_col, ascending=True, inplace=True)
    #fig = [px.bar(an_top, x='cent_value', y='cent_lab',orientation='h')]
    fig_1 = [go.Bar(x=an_top[num_col],y=an_top[lab_col],orientation='h',
                    marker=dict(color=an_top[col_col]))]
    
    return fig_1

def line_plot_data(an_adj):
    l_data = pd.DataFrame(columns=list(an_adj.columns))
    for i in range(len(an_adj)):
        if an_adj.iloc[i,5] == an_adj.iloc[i,7] or \
        an_adj.iloc[i,5] == an_adj.iloc[i,9] or \
        an_adj.iloc[i,5] == an_adj.iloc[i,11] or \
        an_adj.iloc[i,5] == an_adj.iloc[i,13]:
            l_data = l_data.append(an_adj.iloc[i,:],ignore_index=True)
   
    l_dict_list = list()
    for i in range(len(l_data)):
        lab = l_data.iloc[i,5]
        val = l_data.iloc[i,4]
        l_dict_list.append({'label': lab, 'value': val})
    
    return l_data, l_dict_list

def net_analysis(nodes_1, edges_1):
    G_1 = create_nx_graph(nodes_1,edges_1)
    an_adj, an_st, top_id, g_met_dict  = create_analysis(G_1,nodes_1)
    l_data, l_dict = line_plot_data(an_adj)
    edges_1 = edge_styling(edges_1, top_id)
    bar_1 = create_barchart(an_adj,'cent_value','cent_lab','cent_col')
    bar_2 = create_barchart(an_adj,'rank_value','rank_lab','rank_col')
    bar_3 = create_barchart(an_adj,'deg_value','deg_lab','deg_col')
    bar_4 = create_barchart(an_adj,'betw_value','betw_lab','betw_col')
    bar_dict = {1: bar_1, 2: bar_2, 3: bar_3, 4: bar_4}
    
    nodes_list = list()
    for i in range(len(nodes_1)):
        c_node = {
                "data": {"id": nodes_1.iloc[i,0], 
                         "label": an_adj.iloc[i,7],
                         "e_cent": an_adj.iloc[i,0], 
                         "e_col": an_adj.iloc[i,6],
                         "rank": an_adj.iloc[i,1], 
                         "rank_col": an_adj.iloc[i,8],
                         "deg": an_adj.iloc[i,2], 
                         "deg_col": an_adj.iloc[i,10],
                         "betw": an_adj.iloc[i,3], 
                         "betw_col": an_adj.iloc[i,12]}
                
            }
        nodes_list.append(c_node)
    
    edges_list = list()
    for j in range(len(edges_1)):
        c_edge = {
                "data": {"source": edges_1.iloc[j,0], 
                         "target": edges_1.iloc[j,1],
                         "weight": edges_1.iloc[j,2], 
                         "color_all": edges_1.iloc[j,6]}
            }
        edges_list.append(c_edge)
    
    elements = nodes_list + edges_list
    
    return bar_dict, elements, g_met_dict, l_data, l_dict

def character_plot_data(l_plot_list):
    c_df = pd.DataFrame()
    for i in range(len(l_plot_list)):
        a = l_plot_list[i]
        c_df = pd.concat([c_df,a.iloc[:,4]], ignore_index=True, axis=0)
    u_arr = c_df[0].unique()
    
    char_met_dict = dict()
    for j in range(len(u_arr)):
        char_met = pd.DataFrame()
        b = u_arr[j]
        for k in range(len(l_plot_list)):
            c = l_plot_list[k]
            c['season'] = [(k+1)] * len(c)
            for m in range(len(c)):
                if c.iloc[m,4] == b:
                    char_met = char_met.append(c.iloc[m,[0,1,2,3,4,14]])
        char_met = char_met[['id', 'season', 'cent_value', 'rank_value', 
                             'deg_value', 'betw_value']]
        char_met.reset_index(inplace=True)
        del char_met['index']
        char_met_dict[b] = char_met
        
    return char_met_dict         

node_data = [nodes_1, nodes_2, nodes_3, nodes_4, nodes_5, nodes_6, nodes_7, 
             nodes_8]
edge_data = [edges_1, edges_2, edges_3, edges_4, edges_5, edges_6, edges_7, 
             edges_8]
#seasons = [1,2,3,4,5,6,7,8]
elements_list = list()
bar_list = list()
g_met_list = list()
l_plot_list = list()
l_char_list = list()
for i in range(len(node_data)):
    #bar_name = 'bars_s_' + str(seasons[i])
    bars, elements, g_met_dict, l_data, l_dict = net_analysis(node_data[i],
                                                              edge_data[i])
    elements_list.append(elements)
    bar_list.append(bars)
    g_met_list.append(g_met_dict)
    l_plot_list.append(l_data)
    l_char_list.append(l_dict)

l_char_vals = [y for x in l_char_list for y in x]
char_vals_uni = list({v['value']:v for v in l_char_vals}.values())

char_met_dict = character_plot_data(l_plot_list)    

app = dash.Dash(__name__)

#app layout here
app.layout = html.Div([
    html.Div(className='site-title text-center', children=[
        html.H1(className='padheader', children=[
            'Network of Thrones'
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='col-3', children=[
            html.Div(className='center-items', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Select season']),
                    ]),
                html.Div(className='slider-container', children=[
                    dcc.Slider(className='slider', id='season-slider',
                               updatemode='drag',
                               vertical=True,
                               marks={
                                   1: {'label': 'Season 1', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   2: {'label': 'Season 2', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   3: {'label': 'Season 3', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   4: {'label': 'Season 4', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   5: {'label': 'Season 5', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   6: {'label': 'Season 6', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   7: {'label': 'Season 7', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   8: {'label': 'Season 8', 
                                       'style': {'color': '#34373b', 
                                                 'font-size': '18px'}},
                                   },
                               min=1,
                               max=8,
                               step=1,
                               value=1
                               )
                    ]),
                ]),
            ]),
        html.Div(className='col-4',children=[
            cyto.Cytoscape(
                id='cyto-test',
                className='net-obj',
                elements=elements,
                style={'width':'100%', 'height':'600px'},
                layout={'name': 'cose',
                        'padding': '200',
                        #'quality': 'proof',
                        'nodeRepulsion': '7000',
                        #'gravity': '0.01',
                        'gravityRange': '6.0',
                        'nestingFactor': '0.8',
                        'edgeElasticity': '50',
                        'idealEdgeLength': '200',
                        'nodeDimensionsIncludeLabels': 'true',
                        'numIter': '6000',
                        },
                stylesheet=[
                        {'selector': 'node',
                         'style': {
                                 'width': 'data(e_cent)',
                                 'height': 'data(e_cent)',
                                 'background-color': 'data(e_col)',
                                 'content': 'data(label)',
                                 'font-size': '40px',
                                 'text-outline-color': 'white',
                                 'text-outline-opacity': '1',
                                 'text-outline-width': '8px',
                                 # 'text-background-color': 'white',
                                 # 'text-background-opacity': '1',
                                 # 'text-background-shape': 'round-rectangle',
                                 # 'text-background-padding': '20px'
                             }
                         },
                        {'selector': 'edge',
                         'style': {
                                 'line-color': 'data(color_all)'
                             }
                         }
                    ]
                )
            ]),
        html.Div(className='col-3',children=[
            html.Div(className='text-center', children=[
                html.H3(children=['Character data']),
                ]),
            html.Div(className='data-container', children=[
                html.Ul(id='node-data', className='list-text')
                ]),
            html.Div(className='text-center', children=[
                html.H3(children=['Season data']),
                ]),
            html.Div(className='data-container', children=[
                html.Ul(id='season-data', className='list-text')
                ]),
            ]),
        html.Div(className='col-2',children=[
            ]),
        ]),
    html.Div(className='padtop', children=[
        html.Div(className='row', children=[
            html.Div(className='l-graph-container', children=[
                html.H3(children=[('Top 10 characters central to the story '
                                   'by season')]),
                ]),
            ]),
        html.Div(className='row', children=[
            html.Div(className='l-graph-container', children=[
                html.P(children=[('The metrics below are measures of '
                                  'centrality. They provide a measure of the '
                                  'prominance of the character in furthering '
                                  'the story line. Detailed descriptions of '
                                  'what each metric means is given below each '
                                  'graph.')]),
                ]),
            ]),
        ]),
    html.Div(className='padtop', children=[
        html.Div(className='row', children=[
            html.Div(className='col-3', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Eigenvector centrality'])
                    ]),
                ]),
            html.Div(className='col-3', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Page rank'])
                    ]),
                ]),
            html.Div(className='col-3', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Degree'])
                    ]),
                ]),
            html.Div(className='col-3', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Betweenness'])
                    ]),
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='col-3', children=[
            html.Div(className='center-items', children=[
                dcc.Graph(id='bar_1',
                      figure={'layout': go.Layout(margin={'t': 0})},
                      config={'displayModeBar': False},
                      style={'width': '90%', 'height': '100%'}
                      ),    
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='center-items', children=[
                dcc.Graph(id='bar_2',
                      figure={'layout': go.Layout(margin={'t': 0})},
                      config={'displayModeBar': False},
                      style={'width': '90%', 'height': '100%'}
                      ),    
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='center-items', children=[
                dcc.Graph(id='bar_3',
                      figure={'layout': go.Layout(margin={'t': 0})},
                      config={'displayModeBar': False},
                      style={'width': '90%', 'height': '100%'}
                      ),    
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='center-items', children=[
                dcc.Graph(id='bar_4',
                      figure={'layout': go.Layout(margin={'t': 0})},
                      config={'displayModeBar': False},
                      style={'width': '90%', 'height': '100%'}
                      ),    
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='col-3', children=[
            html.Div(className='text-center', children=[
                html.P(className='pad-paragraph', 
                       children=[("Eigenvector centrality - provides a "
                                  "measure of a nodes' (characters') "
                                  "importance in facilitating connections "
                                  "(edges) across the whole of the network")])
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='text-center', children=[
                html.P(className='pad-paragraph', 
                       children=['Page rank - orders the nodes (characters) '
                                 'based on the number of incoming connections '
                                 '(edges)'])
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='text-center', children=[
                html.P(className='pad-paragraph', 
                       children=['Degree - the total number of incoming and '
                                 'outgoing connections (edges)'])
                ]),
            ]),
        html.Div(className='col-3', children=[
            html.Div(className='text-center', children=[
                html.P(className='pad-paragraph', 
                       children=['Betweenness - measures how many paths '
                                 'between any two characters (nodes) pass '
                                 'through a given character (node)'])
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='padsection', children=[
            html.Div(className='l-graph-container', children=[
                html.H3(children=['Whole network metrics by season']),
                ]),
            html.Div(className='l-graph-container', children=[
                html.P(children=[('Metrics describing the whole network for '
                                  'each season can be compared. Select one or '
                                  'more metrics from the dropdown below')]),
                ]),
            html.Div(className='l-graph-container', children=[
                html.Label(className='drop-label', children=[
                    'Select metric to compare on',
                    dcc.Dropdown(className='drops',
                    id='met-drop',
                    options=[
                        {'label':'Number of characters', 'value':'num_chars'},
                        {'label':'Number of interactions', 'value':'num_inter'},
                        {'label':'Density of interactions', 'value':'density'},
                        {'label':'Average interactions', 'value':'avg_deg'},
                        ],
                    multi=True,
                    value=['num_chars']
                    )
                    ]),
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='padsection', children=[
            html.Div(className='l-graph-container', children=[
                dcc.Graph(className='line-graph', id='season-line')
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='padsection', children=[
            html.Div(className='l-graph-container', children=[
                html.H3(children=[('Comparison of character metrics across '
                                   'all seasons')]),
                ]),
            html.Div(className='l-graph-container', children=[
                html.P(children=[('Here you can compare the metrics for '
                                  'different characters across all 8 seasons. '
                                  'Select one or more characters from the '
                                  'dropdown below and a metric to compare')]),
                ]),
            html.Div(className='l-graph-container', children=[
                html.Label(className='drop-label', children=[
                    'Select character(s)',
                    dcc.Dropdown(className='drops',
                    id='char-drop',
                    options=char_vals_uni,
                    multi=True,
                    value=['ARYA']
                    )
                    ]),
                html.Label(className='drop-label', children=[
                    'Select metric to compare on',
                    dcc.Dropdown(className='drops',
                    id='char-met-drop',
                    options=[
                        {'label': 'Eigenvector centrality',
                         'value': 'cent_value'},
                        {'label': 'Page rank', 'value': 'rank_value'},
                        {'label': 'Degree', 'value': 'deg_value'},
                        {'label': 'Betweenness', 'value': 'betw_value'},
                        ],
                    clearable=False,
                    value='cent_value',
                    style={'width': '300px'})
                    ]),
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='padsection', children=[
            html.Div(className='l-graph-container', children=[
                dcc.Graph(className='line-graph', id='char-line')
                ]),
            ]),
        ]),
    ])

@app.callback(Output('season-line', 'figure'),
              [Input('met-drop', 'value')])
def updateSeasonChart(metric):
    df = pd.DataFrame(g_met_list)
    df['Season'] = [1,2,3,4,5,6,7,8]
    yaxis_name = {'num_chars':'Number of characters',
                    'num_inter':'Number of interactions',
                    'density':'Density of interactions',
                    'avg_deg':'Average interactions'}
    fig = go.Figure()
    for i in range(len(metric)):
        fig.add_trace(go.Scatter(x=df['Season'], y=df[metric[i]], name=yaxis_name[metric[i]]))
    fig.update_layout(xaxis_title='Season', yaxis_title='Metric value',
                      showlegend=True, legend_title_text='Metric')
    #fig = px.line(df, x='Season', y=metric)
    return fig

@app.callback(Output('char-line', 'figure'),
              [Input('char-drop', 'value'),
               Input('char-met-drop', 'value')])
def updateCharacterChart(character, metric):
    yaxis_name = {'cent_value': 'Eigenvector centrality',
                   'rank_value': 'Page rank', 'deg_value': 'Degree',
                   'betw_value': 'Betweenness'}
    fig = go.Figure()
    for i in range(len(character)):
        df = char_met_dict[character[i]]
        fig.add_trace(go.Scatter(x=df['season'],
                                 y=df[metric],name=character[i]))
    fig.update_layout(xaxis_title='Season', yaxis_title=yaxis_name[metric],
                      showlegend=True, legend_title_text='Characters')
    return fig
    
@app.callback(Output('node-data', 'children'),
              [Input('cyto-test', 'tapNodeData')])
def displayNodeData(data):
    if data:
        info_list = []
        info_list.append(html.Li('Name: ' + data['id']))
        info_list.append(html.Li('Eignvector: ' + \
                                 str(round(data['e_cent'],2))))
        info_list.append(html.Li('Page rank: ' + \
                                 str(round(data['rank'],2))))
        info_list.append(html.Li('Degree: ' + str(round(data['deg'],2))))
        info_list.append(html.Li('Betweenness: ' + \
                                 str(round(data['betw'],2))))
        
        return info_list
    
@app.callback(Output('season-data', 'children'),
              [Input('season-slider', 'value')])
def displaySeasonData(season):
    met_dict = g_met_list[season-1]
    if season:
        s_info_list = []
        s_info_list.append(html.Li('Season: ' + str(season)))
        s_info_list.append(html.Li('Characters: ' + \
                                   str(met_dict['num_chars'])))
        s_info_list.append(html.Li('Interactions: ' + \
                                   str(met_dict['num_inter'])))
        s_info_list.append(html.Li('Density: ' + \
                                   str(round(met_dict['density'],2))))
        s_info_list.append(html.Li('Avg interactions: ' \
                                   + str(round(met_dict['avg_deg'],2))))
        
        return s_info_list

@app.callback([Output('cyto-test', 'elements'),
               Output('bar_1', 'figure'),
               Output('bar_2', 'figure'),
               Output('bar_3', 'figure'),
               Output('bar_4', 'figure')],
              [Input('season-slider', 'value')])
def changeSeason(season):
    s = season - 1
    net = elements_list[s]
    bars = bar_list[s]
    b_1 = bars[1]
    b_2 = bars[2]
    b_3 = bars[3]
    b_4 = bars[4]
    
    return net, {'data': b_1}, {'data': b_2}, {'data': b_3}, {'data': b_4}


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050)