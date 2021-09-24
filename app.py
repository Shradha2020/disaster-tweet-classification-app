from flask import Flask, render_template, request
from classify_tweets import tweet_classifier

app = Flask(__name__)


@app.route("/classify_tweets", methods=["POST"])
def classify_tweets():
    tweet = request.form['tweet']
    tweet, classification = tweet_classifier.predictTweet(tweet)
    return render_template("index.html", title="Tweets Classification", tweet=tweet, classification=classification)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run()
