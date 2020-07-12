import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def plot_genre_distr():
    """
    Returns disctionary of plotly plot for genre distribution

    :return: dictionary of plotly plot
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    return {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts
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
    }


def plot_response_distr():
    """
    Returns disctionary of plotly plot for response distribution

    :return: dictionary of plotly plot
    """
    category_names = [c for c in df if c not in ('id', 'message', 'original', 'genre')]
    response_counts = (
        df[category_names].melt(var_name='response_type')
            .groupby('response_type').sum()
            .reset_index().rename(columns={'value': 'count'})
            .sort_values('count', ascending=False)
    )

    return {
        'data': [
            Bar(
                x=response_counts['response_type'],
                y=response_counts['count']
            )
        ],

        'layout': {
            'title': 'Distribution of Response Types',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Response Type",
                'automargin': True
            }
        }
    }


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    genre_distr = plot_genre_distr()
    response_distr = plot_response_distr()

    # create visuals
    graphs = [genre_distr, response_distr]

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
