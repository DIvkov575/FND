from flask import Flask, render_template, request, redirect, url_for
import logging
from lib.soup import fetch
from lib.load import predict_news

app = Flask(__name__)


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/result_page')
def result_page():
    return render_template('result.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        url = request.form.get('url')
        text = request.form.get('text')
        contents = ""

        try:
            if url:
                contents = fetch(url)
            elif text:
                contents = text
            else:
                raise AssertionError("No prompt entered")

            prediction = predict_news(contents)
            logger.info(f"{contents}\n{prediction}")
            return redirect(url_for('result_page', result=prediction))

        except Exception as e:
            logger.error(e)
            return redirect(url_for('index_page'))

    else:
        logger.warning("unreachable GET request on /submit")
        return redirect(url_for('index_page'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,  # Set the logging level
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
                        handlers=[logging.StreamHandler()])  # Output to console

    logger = logging.getLogger(__name__)

    from waitress import serve
    # app.run(debug=True)
    serve(app, host='0.0.0.0', port=8080)