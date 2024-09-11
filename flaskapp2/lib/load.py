import joblib
import string
import re

RFC = joblib.load('../out/random_forest_model.pkl')
# LR = joblib.load('logistic_regression_model.pkl')
# DT = joblib.load('../out/decision_tree_model.pkl')
# GBC = joblib.load('gradient_boosting_model.pkl')

# Load the vectorizer
vectorization = joblib.load('../out/tfidf_vectorizer.pkl')

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


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


def predict_news(news):
    if not news:
        return "No news provided!"

    news = wordopt(news)
    news_vectorized = vectorization.transform([news])

    # Make predictions with all models
    # pred_LR = LR.predict(news_vectorized)[0]
    # pred_DT = DT.predict(news_vectorized)[0]
    # pred_GBC = GBC.predict(news_vectorized)[0]
    pred_RFC = RFC.predict(news_vectorized)[0]

    # Format the predictions
    predictions = {
        # "Logistic Regression": output_lable(pred_LR),
        # "Decision Tree": output_lable(pred_DT),
        # "Gradient Boosting": output_lable(pred_GBC),
        "Random Forest": output_lable(pred_RFC)
    }

    print("###################")
    print(predictions)

    # return predictions