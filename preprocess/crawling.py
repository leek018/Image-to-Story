import requests
from bs4 import BeautifulSoup

def crawling(write, url, page):

    f = open(write, 'w', -1, "utf-8")

    # 1 페이지 크롤링
    response = requests.get(url).text
    soup = BeautifulSoup(response, "html.parser")
    posts = soup.select(".entry-content")
    for post in posts:
        writer = post.select_one('figure > span')
        title = post.select_one('.post_title > a')
        if writer and '관리자' not in writer.text and '장원' not in title.text and '안녕하세요' not in title.text:
            url2 = title['href']
            detail_res = requests.get(url2).text
            detail_soup = BeautifulSoup(detail_res, "html.parser")
            details = detail_soup.select(".entry-content > p")
            for detail in details:
                if len(detail.text)>3:
                    f.write('\n'+detail.text)


    # 전체 페이지 크롤링
    for i in range(2, page):
        response = requests.get(url+'/page/'+str(i)).text
        soup = BeautifulSoup(response, "html.parser")
        posts = soup.select(".entry-content")
        for post in posts:
            writer = post.select_one('figure > span')
            title = post.select_one('.post_title > a')
            # 필터링
            if writer and '관리자' not in writer.text and '장원' not in title.text and '안녕하세요' not in title.text:
                url2 = title['href']
                detail_res = requests.get(url2).text
                detail_soup = BeautifulSoup(detail_res, "html.parser")
                details = detail_soup.select(".entry-content > p")
                f.write('\n')
                for detail in details:
                    if len(detail.text)>3:
                        f.write('\n'+detail.text)
                        print(detail.text)
    f.close()

crawling('life_data.txt', 'https://teen.munjang.or.kr/archives/category/write/life', 316)
# crawling('story_data.txt', 'https://teen.munjang.or.kr/archives/category/write/story', 480)