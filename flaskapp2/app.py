from flask import Flask, render_template, request, redirect, url_for
from lib.soup import fetch_webpage_contents
from lib.load import predict_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_text', methods=['POST'])
def submit_text():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        predict_news(user_input)
        return redirect(url_for('index', result=""))

@app.route('/submit_url', methods=['POST'])
def submit_url():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        print(user_input)
        a = fetch_webpage_contents(user_input)
        print(a)
        predict_news(a)
        return redirect(url_for('index', result=""))

# @app.route('/result')
# def result():
#     result = request.args.get('result', '')
#     return f"Result: {result}"

if __name__ == '__main__':
    app.run(debug=True)