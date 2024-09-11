import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def get_page(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response
    else:
        return False


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def fetch_webpage_contents(url):
    try:
        assert is_valid_url(url)

        response = get_page(url)

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()

        return text

    except Exception as e:
        return f"An error occurred: {e}"



# if __name__ == '__main__':
    # a = fetch_webpage_contents("https://www.politico.com/live-updates/2024/09/10/trump-harris-presidential-debate-tonight/donald-trump-ukraine-russia-war-00178515")
    # print(a)
    # a = "https://www.politico.com/live-updates/2024/09/10/trump-harris-presidential-debate-tonight/donald-trump-ukraine-russia-war-00178515"
    # url = 'http://worldagnetwork.com/'

    # print(result)
