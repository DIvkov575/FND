import joblib
import string
import re

RFC = joblib.load('weights/random_forest_model.pkl')
vectorization = joblib.load('weights/tfidf_vectorizer.pkl')


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def predict_news(news):
    """
    :param news: text content to be used for prediction
    :return:
    0 - fake news
    1 - real news

    """
    assert news

    news = wordopt(news)
    news_vectorized = vectorization.transform([news])
    return RFC.predict(news_vectorized)[0]
