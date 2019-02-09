
from bs4 import BeautifulSoup
import urllib.request
import os

#created by saria goudarzvand
def search_spider(sea, lim):
    url = "https://en.wikipedia.org/w/index.php?limit="+lim+"&offset=0&search="+sea
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    result_list = soup.findAll('div', {'class': "mw-search-result-heading"})
    print(result_list)
    for div in result_list:
        link = div.find('a')
        href = "https://en.wikipedia.org"+link.get('href')
        if (link.get('href').startswith("http")):
            href=link.get('href')
        get_data(href)


def get_data(url):
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    body = soup.find('div', {'class': 'mw-parser-output'})
    file2.write(str(body.text))
    print(body.text)

search = input('type something to search in wiki: ')
limit = input('how many results do you want to get?: ')

if not os.path.exists(search):
    print("Creating folder " + search)
    file2 = open(search+'.txt','a+',encoding='utf-8')

search_spider(search, limit)