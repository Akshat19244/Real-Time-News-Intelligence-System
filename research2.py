import dash
from dash import html, dcc, Input, Output
import dash_cytoscape as cyto
import networkx as nx
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import vstack
import feedparser
import time
from collections import Counter
from bs4 import BeautifulSoup
import plotly.graph_objs as go
import random
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import shap

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

rss_url = "https://news.google.com/rss"

app = dash.Dash(__name__)
app.title = "Live Graph Streaming with Communities"

vectorizer = TfidfVectorizer(stop_words='english')
G = nx.Graph()
tfidf_matrix = None
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)

WINDOW_SIZE = 50
SIMILARITY_THRESHOLD = 0.3
most_central_post = None
most_viral_community = None
previous_partition = {}
community_colors = {}

# Clean text from HTML tags
def clean_text(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Fetch RSS feed data
def fetch_rss_feed(url):
    print(f"Fetching RSS feed from: {url}")
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            raise ValueError(f"Error parsing feed: {feed.bozo_exception}")
        articles = []
        for entry in feed.entries:
            articles.append({
                "title": clean_text(entry.title),
                "link": entry.link,
                "published": entry.published if "published" in entry else None,
                "summary": clean_text(entry.summary) if "summary" in entry else ""
            })
        if not articles:
            print("No articles found in the feed.")
        return articles
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return []

# Stream RSS feed articles with dynamic intervals
def stream_rss_data(url, initial_interval=10):
    seen_articles = set()
    interval = initial_interval
    while True:
        articles = fetch_rss_feed(url)
        if not articles:
            print("No valid articles retrieved. Retrying...")
            interval = min(interval * 2, 600) 
        else:
            interval = max(initial_interval, interval // 2) 
        for article in articles:
            if article["link"] not in seen_articles:
                seen_articles.add(article["link"])
                yield article
        time.sleep(interval)

# Enhanced sentiment analysis using transformers
def analyze_sentiment_enhanced(text):
    sentiment = sentiment_pipeline(text)[0]
    return {
        'polarity': sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'],
        'subjectivity': None,  
        'vader': None  
    }

# Topic modeling for node summaries
def extract_topics(tfidf_vector):
    if tfidf_vector is not None and tfidf_vector.shape[0] > 0:
        topic_distribution = lda_model.transform(tfidf_vector)
        dominant_topic = topic_distribution.argmax()
        return f"Topic {dominant_topic}"
    return "No Topic"

# Add new posts as nodes and compute edges
def add_edges_to_graph(new_post, tfidf_matrix, G, threshold=SIMILARITY_THRESHOLD):
    if not new_post.strip():
        print("Skipping empty or null post.")
        return tfidf_matrix

    new_tfidf_vector = vectorizer.transform([new_post])
    topic = extract_topics(new_tfidf_vector)
    new_node = f"Post {len(G.nodes) + 1}"
    
    # Perform enhanced sentiment analysis
    sentiment = analyze_sentiment_enhanced(new_post)
    timestamp = time.time()
    
    G.add_node(new_node, label=topic, sentiment=sentiment, timestamp=timestamp)
    
    if tfidf_matrix is not None:
        similarities = cosine_similarity(new_tfidf_vector, tfidf_matrix).flatten()
        for i, sim in enumerate(similarities):
            if sim > threshold:
                existing_node = f"Post {i + 1}"
                G.add_edge(new_node, existing_node, weight =sim, timestamp=timestamp)
    else:
        tfidf_matrix = new_tfidf_vector
    return tfidf_matrix

# Remove old nodes based on sliding window size
def remove_old_nodes(G):
    if len(G.nodes) > WINDOW_SIZE:
        nodes_by_time = sorted(G.nodes(data=True), key=lambda x: x[1].get('timestamp', 0))
        oldest_node = nodes_by_time[0][0]
        G.remove_node(oldest_node)

# Assign random colors to communities
def assign_community_colors(partition):
    community_counts = Counter(partition.values())
    return {comm: "#" + ''.join(random.choices("0123456789ABCDEF", k=6)) for comm in community_counts}

# Convert NetworkX graph to Cytoscape elements
def nx_to_cytoscape_elements(G, partition, centrality):
    global most_central_post, most_viral_community, community_colors

    most_central_post = max(centrality, key=centrality.get, default=None)
    community_sizes = Counter(partition.values())
    most_viral_community = max(community_sizes, key=community_sizes.get, default=None)

    community_colors = assign_community_colors(partition)

    elements = []
    for node in G.nodes:
        comm_id = partition.get(node, -1)
        node_color = community_colors.get(comm_id, "#FFFFFF")

        elements.append({
            'data': {
                'id': node,
                'label': G.nodes[node].get('label', 'Unknown'),
                'community': comm_id,
                'sentiment': G.nodes[node].get('sentiment', 0.0),
                'centrality': centrality.get(node, 0.0),
                'polarity': G.nodes[node].get('sentiment', {}).get('polarity', 0.0),
                'subjectivity': G.nodes[node].get('sentiment', {}).get('subjectivity', None),
                'vader': G.nodes[node].get('sentiment', {}).get('vader', None)
            },
            'style': {
                'background-color': node_color,
                'border-color': '#000000',
                'border-width': 1
            }
        })

    for edge in G.edges(data=True):
        elements.append({
            'data': {
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2].get('weight', 1.0),
                'timestamp': edge[2].get('timestamp', 0)
            }
        })
    return elements


# Define the generate_timeline function
def generate_timeline(G):
    timestamps = [ts for ts in [G.nodes[node].get('timestamp', None) for node in G.nodes] if ts is not None]
    
    counts = Counter(timestamps)
    timeline_data = sorted(counts.items())
    x_data = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) for ts, count in timeline_data]
    y_data = [count for ts, count in timeline_data]

    return {
        'data': [go.Scatter(x=x_data, y=y_data, mode='lines+markers')],
        'layout': go.Layout(title='Timeline of Posts', xaxis_title='Time', yaxis_title='Number of Posts')
    }


# Fetch initial articles and fit the vectorizer
initial_articles = fetch_rss_feed(rss_url)
initial_summaries = [article["summary"] for article in initial_articles if "summary" in article]

if initial_summaries:
    vectorizer.fit(initial_summaries)
    tfidf_matrix = vectorizer.transform(initial_summaries)
    lda_model.fit(tfidf_matrix)
else:
    print("No valid summaries found to fit the vectorizer.")

# Dash app layout
app.layout = html.Div([
    html.H1("Live Graph Streaming and Community Detection", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='layout-dropdown',
        options=[
            {'label': 'Grid', 'value': 'grid'},
            {'label': 'Circle', 'value': 'circle'},
            {'label': 'Cose', 'value': 'cose'},
            {'label': 'Breadthfirst', 'value': 'breadthfirst'},
            {'label': 'Concentric', 'value': 'concentric'},
            {'label': 'Random', 'value': 'random'},
            {'label': 'Euler', 'value': 'euler'},
            {'label': 'Ecosystem', 'value': 'ecosystem'},
        ],
        value='cose',
        style={'width': '50%', 'margin': 'auto'}
    ),
    cyto.Cytoscape(
        id='cyto-graph',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'},
        elements=[]
 ),
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  
        n_intervals=0
    ),
    html.Div(id='live-update-text'),
    dcc.Graph(id='timeline-graph'),
    html.Div(id='community-info', style={'textAlign': 'center'}),
    html.Div(id='influential-node-info', style={'textAlign': 'center'}),
    
    # New components for additional information
    html.Div(id='community-list', style={'textAlign': 'center'}),
    dcc.Graph(id='topic-distribution-graph'),
    html.Div(id='recent-articles', style={'textAlign': 'center'}),
    html.Div(id='sentiment-summary', style={'textAlign': 'center'}),
    
    # User feedback section
    html.Div([
        dcc.Input(id='feedback-input', type='text', placeholder='Provide your feedback'),
        html.Button('Submit', id='submit-feedback'),
        html.Div(id='feedback-output')
    ])
])

# Callback to update the Cytoscape graph and community information
@app.callback(
    Output('cyto-graph', 'elements'),
    Output('community-info', 'children'),
    Output('influential-node-info', 'children'),
    Output('community-list', 'children'),
    Output('topic-distribution-graph', 'figure'),
    Output('recent-articles', 'children'),
    Output('sentiment-summary', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('layout-dropdown', 'value')
)
def update_cytoscape_elements(n_intervals, layout):
    global tfidf_matrix
    articles = fetch_rss_feed(rss_url)
    for article in articles:
        tfidf_matrix = add_edges_to_graph(article["summary"], tfidf_matrix, G)
    
    partition = community_louvain.best_partition(G)
    centrality = nx.degree_centrality(G)
    
    # Calculate modularity and clustering coefficient
    modularity = community_louvain.modularity(partition, G)
    clustering_coefficient = nx.average_clustering(G)

    # Prepare community information
    num_communities = len(set(partition.values()))
    community_info = f"Number of Communities: {num_communities}, Modularity: {modularity:.4f}, Average Clustering Coefficient: {clustering_coefficient:.4f}"

    # Identify the most influential node
    most_influential_node = max(centrality, key=centrality.get, default=None)
    influential_node_info = f"Most Influential Node: {most_influential_node}, Centrality: {centrality[most_influential_node]:.4f}" if most_influential_node else "No Influential Node Found"

    # Community List and Sentiment Analysis
    community_list = []
    community_sentiments = {}
    for comm_id in set(partition.values()):
        community_nodes = [node for node in G.nodes if partition[node] == comm_id]
        avg_sentiment = sum(G.nodes[node].get('sentiment', {'polarity': 0})['polarity'] for node in community_nodes) / len(community_nodes) if community_nodes else 0
        community_sentiments[comm_id] = avg_sentiment
        community_list.append(f"Community {comm_id}: Average Sentiment = {avg_sentiment:.4f}")

    community_list_display = html.Ul([html.Li(item) for item in community_list])

    # Topic Distribution
    topic_distribution = Counter([G.nodes[node].get('label', 'Unknown') for node in G.nodes])
    topic_labels = list(topic_distribution.keys())
    topic_values = list(topic_distribution.values())
    topic_distribution_fig = {
        'data': [go.Bar(x=topic_labels, y=topic_values)],
        'layout': go.Layout(title='Topic Distribution', xaxis_title='Topics', yaxis_title='Count')
    }

    # Recent Articles
    recent_articles_display = html.Ul([html.Li(f"{article['title']} - {article['link']}") for article in articles])

    # Sentiment Summary
    overall_sentiment = sum(G.nodes[node].get('sentiment', {'polarity': 0})['polarity'] for node in G.nodes) / len(G.nodes) if G.nodes else 0
    sentiment_summary = f"Overall Sentiment: {'Positive' if overall_sentiment > 0 else 'Negative' if overall_sentiment < 0 else 'Neutral'}"

    print("Graph G:", G)
    print("Partition:", partition)
    print("Centrality:", centrality)
    
    return nx_to_cytoscape_elements(G, partition, centrality), community_info, influential_node_info, community_list_display, topic_distribution_fig, recent_articles_display, sentiment_summary

# Callback to update the timeline graph
@app.callback(
    Output('timeline-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_timeline(n_intervals):
    return generate_timeline(G)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)