import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def fetch_webpage_contents(url):
    try:
        assert is_valid_url(url)

        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            return text
        else:
            return f"Failed to retrieve the webpage. Status code: {response.status_code}"

    except Exception as e:
        return f"An error occurred: {e}"


