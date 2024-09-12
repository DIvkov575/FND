import requests
from bs4 import BeautifulSoup


def fetch(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(url, headers=headers)

    assert response.status_code == 200, f"url request returned {response.status_code}"

    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()
