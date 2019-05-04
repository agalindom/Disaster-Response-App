import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Pie
from plotly.graph_objs import Bar
import plotly.plotly as py
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('clean_tweet_with_cats', engine)

# load model
model = joblib.load("models/Random_Forest_cv.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    columns_only = df.iloc[:, 5:]
    categories = columns_only.sum().sort_values(ascending= False).head(5).index
    cat_values = list(columns_only.sum().sort_values(ascending= False).head(5).values)
    
    df1 = df.drop(['original', 'id', 'message', 'related'], axis = 1)
    df1_cols = df1.iloc[:, 1:].columns
    df_melt = pd.melt(df, id_vars = ['genre'], value_vars = df1_cols)
    ordered_df = df_melt.groupby(['genre', 'variable'])['value'].sum().reset_index().\
            sort_values(['genre', 'value'], ascending = False)
    social = ordered_df.query('genre == "social"').reset_index(drop = True)
    direct = ordered_df.query('genre == "direct"').reset_index(drop = True)
    news = ordered_df.query('genre == "news"').reset_index(drop = True)
    social_cats = list(social.iloc[:5].variable)
    social_val = list(social.iloc[:5].value)
    news_cats = list(news.iloc[:5].variable)
    news_val = list(news.iloc[:5].value)
    direct_cats = list(direct.iloc[:5].variable)
    direct_val = list(direct.iloc[:5].value)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    opacity = 0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=categories, 
                    values=cat_values
                )
            ],
            'layout': {
                'title': 'Top Ten Message Categories All',
            }
        },
        {
            'data': [
                Bar(
                    x=social_cats,
                    y=social_val,
                    opacity = 0.5,
                    marker=dict(
                    color='rgb(70, 50, 255)')
                )
            ],

            'layout': {
                'title': 'Social Genre Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_cats,
                    y=news_val,
                    opacity = 0.5,
                    marker=dict(
                    color='rgb(255, 0, 0)')
                )
            ],

            'layout': {
                'title': 'News Genre Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_cats,
                    y=direct_val,
                    opacity = 0.5,
                    marker=dict(
                    color='rgb(0, 205, 0)')
                )
            ],

            'layout': {
                'title': 'Direct Genre Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()