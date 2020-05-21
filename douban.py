from bs4 import BeautifulSoup as bs
import requests as rq
import urllib.request
import time
import re
import os
import random
import base64
import csv
from urllib.parse import quote
import string
from urllib import request

book_result = open('/Users/dadaqingjian/Documents/6.Xcode_Projects/scrapper_text/result.txt',
                   'a+')
web_site = 'http://www.bookschina.com'
All_categories = 'http://www.bookschina.com/books/kinder/'  # 一级和二级目录所在页面

proxies = {
      "http":"http://H7S66VGXZ2ZY7M3P:2192100224F4082C@http-pro.abuyun.com:9010"
      #"https": "https://H7S66VGXZ2ZY7M3P:2192100224F4082C@http-pro.abuyun.com:9010"
   }

# 定义函数下载并保存图片
import urllib
import urllib.request
import requests


def get_image(url, img_name):
    d = '/Users/dadaqingjian/Documents/6.Xcode_Projects/scrapper_text/book_cover/'

    path = d + str(img_name) + '.jpg'
    try:
        if not os.path.exists(d):
            os.mkdir(d)
        if not os.path.exists(path):
            r = requests.get(url)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)
                f.close()
                print("图片保存成功")
        else:
            print("图片已存在")
    except:
        print("图片获取失败")


def get_books_from(listpage, first_directory_name, second_directory_name):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'

    headers = {
        "Origin": "https://book.douban.com",
        "Accept": "text/event-stream",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        # "Cookie": cookie,
        "User-Agent": user_agent}
    r = rq.get(listpage, headers=headers, proxies=proxies,timeout=5).text  # headers=headers,

    soup = bs(r, 'html.parser')

    max_page = int(soup.find('div', class_="pagination").find_all('b')[0].text.strip())

    cat = soup.select('head > title ')[0].text
    cat = cat[4:cat.find('书籍')]
    print(cat)
    print(max_page)
    urls = [listpage + '_0_0_11_0_1_{}_0_0/'.format(str(i)) for i in range(1, max_page + 1)]
    book_detail_url = ''
    for url in urls:
        try:
            r = rq.get(url, headers=headers,proxies=proxies, timeout=5).text
            soup = bs(r, 'html.parser')

            books = soup.select('div.bookList > ul > li')
            for bk in books:
                nt = bk.select('div.cover > a')[0]
                # 进入、书籍详细页面
                book_detail_url = web_site + nt['href']

                r2 = rq.get(book_detail_url,proxies=proxies, timeout=5).text  # headers=headers,
                soup2 = bs(r2, 'html.parser')
                try:
                    book_special_detail = soup2.select('div.specialist > p')[0].text  # 在图书详细页面上获取图书内容
                except:
                    try:
                        book_special_detail = soup2.select('div.brief > p')[0].text  # 在图书详细页面上获取图书内容
                    except:
                        try:
                            book_special_detail = soup2.select('div.excerpt > p')[0].text
                        except:
                            pass

                book_cover_image_url = soup2.select('div.coverImg > a > img')[0]['src']
                # 读取并存储图片
                book_name = nt['title']


                get_image(url=book_cover_image_url, img_name=book_name)
                desc = bk.select('p.recoLagu')[0].text
                desc = re.sub('\s+', ' ', desc)
                # label表示该文本的标签
                data = [str(first_directory_name), str(second_directory_name), nt['title'], desc, cat,
                        book_special_detail]
                print(data[0], data[1], data[2])

                book_result.write("\t".join(data) + '\n')

            # ra = random.uniform(0, 3)
            # print('hi休眠等待：%f秒' % ra)
            # time.sleep(ra)
        except:
            print('there no items in this {} page '.format(book_detail_url))


# 定义一个函数用于遍历一级目录和二级目录
def get_directory(links):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',
                   UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')
    headers = {
        "Origin": "https://book.douban.com",
        "Accept": "text/event-stream",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        # "Cookie": cookies,
        "User-Agent": user_agent}
    r = rq.get(links, headers=headers,proxies=proxies, timeout=5).text  # headers=headers,
    soup02 = bs(r, 'html.parser')
    first_directory_list = soup02.select('div.w1200>h2>a')  # 获取一级目录分类
    # first_directory_name = first_directory_list[0].text  # 表示第一级目录的名字
    second_directory_list = soup02.select('div.w1200>ul')  # 获取二级目录类别
    count01 = 0
    for directory in second_directory_list:  # 遍历所有的二级目录
        first_directory_name = first_directory_list[count01].text  # 表示第一级目录的名字
        for directory02 in list(directory):
            try:
                nt = directory02.select('a')[0]
            except:
                continue;
            # 获取二级目录名称
            second_directory_name = directory02.text

            # 进入、书籍详细页面
            second_class_detail_url = web_site + nt['href'][:-1]  # 获取每一个二级类别，然后从跳转之后的页面获取类别信息
            # 调用get_books_from(listpage, label)
            get_books_from(listpage=second_class_detail_url, first_directory_name=first_directory_name,
                           second_directory_name=second_directory_name)
        count01 = count01+1


# 第一步，从构造好的链接列表中依次取出，抽取书的详情页的网址，保存到数据库。
# 每一个连接表示一个类别 标签从0开始，0，1，2，...
Label = 0
if __name__ == '__main__':
    get_directory(links=All_categories)

