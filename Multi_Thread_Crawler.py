import requests
import time
from threading import Thread
from queue import Queue
import json
from bs4 import BeautifulSoup as bs
import requests as rq
import urllib.request
import time
import threading
from threading import Thread
from queue import Queue
import re
import os


def run_time(func):
    def wrapper(*args, **kw):
        start = time.time()
        func(*args, **kw)
        end = time.time()
        print('running', end - start, 's')

    return wrapper


class Spider():

    def __init__(self):
        self.qurl = Queue()
        self.data = list()
        # self.page_num = 171
        self.max_page = 1
        self.thread_num = 50
        self.cat = ''
        self.second_class_detail_url, self.first_directory_name = '', ''
        self.book_detail_url = ''
        self.book_special_detail = ''
        self.desc = ''
        self.web_site = 'https://www.zhihu.com/topic/19551275/hot'
        self.All_categories = 'http://www.bookschina.com/books/kinder/'  # 一级和二级目录所在页面
        self.book_result = open('/Users/dadaqingjian/Documents/6.Xcode_Projects/scrapper_text/result.txt',
                                'a+')
        self.img_directory = '/Users/dadaqingjian/Documents/6.Xcode_Projects/scrapper_text/book_cover'

        self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
        self.headers = {
            "Origin": "https://book.douban.com",
            "Accept": "text/event-stream",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Connection": "keep-alive",
            # "Cookie": cookies,
            "User-Agent": self.user_agent}
        self.proxies = {
            "http": "http://H29W337D920H7YSP:6209E0012E0A9356@http-pro.abuyun.com:9010"
        }

    def get_image(self, url, img_name):
        path = self.img_directory + str(img_name) + '.jpg'
        try:
            if not os.path.exists(self.img_directory):
                os.mkdir(self.img_directory)
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

    def produce_url(self, directory):
        count01 = 0
        self.first_directory_name = self.first_directory_list[count01].text  # 表示第一级目录的名字
        for directory02 in list(directory):
            try:
                nt = directory02.select('a')[0]
            except:
                continue;
            # 获取二级目录名称
            self.second_directory_name = directory02.text
            if self.second_directory_name not in ['纪实文学', '民间文学', '外国诗歌', '外国随笔', '文集', '文学理论', '戏剧', '中国古代随笔',
                                        '中国古诗词', '中国现当代诗歌', '中国现当代随笔', '中国当代小说', '中国古典小说', '中国近现代小说', '外国小说', '财经',
                                        '都市', '港澳台小说', '官场', '惊悚/恐怖', '军事', '科幻', '历史', '魔幻', '情感',
                                        '社会', '武侠', '乡土', '影视小说', '侦探/悬疑/推理', '职场', '作品集', '四大名著', '世界名著',
                                        '爱情/情感', '爆笑/无厘头', '叛逆/成长', '校园', '玄幻/新武侠/魔幻/科幻', '悬疑/惊悚', '娱乐/偶像', '大陆原创',
                                        '港台青春文学', '韩国青春文学', '其他国外青春文学',
                                        '0-2岁', '7岁及以上', '3-6岁', '动漫/卡通', '儿童文学', '中国儿童文学', '外国儿童文学', '科普/百科', '励志/成长',
                                        '少儿英语', '绘本', '艺术课堂', '益智游戏', '幼儿启蒙', '传统文化',
                                        '地震', '动物', '海洋生物', '昆虫', '鸟类', '植物', '河流', '海洋', '环境保护与治理', '历史', '气象', '人类',
                                        '生态', '星体观测', '自然灾害', '其他科普知识',
                                        '大陆漫画', '动漫学堂', '港台漫画', '日韩漫画', '欧美漫画', '其他国外漫画', '世界经典漫画集', '小说/名著漫画版',
                                        '幽默/笑话集',
                                        '教育', '社会科学类教材', '社会科学总论', '社会学', '图书馆学档案学', '文化人类学', '心理学', '新闻传播出版', '语言文字',
                                        '地方史志'
                                        ] and self.second_directory_name != ', ':
                # 进入、书籍详细页面
                second_class_detail_url = self.web_site + nt['href'][:-1]  # 获取每一个二级类别，然后从跳转之后的页面获取类别信息
                # 调用get_books_from(listpage, label)
                self.get_books_from(
                    listpage=second_class_detail_url)  # , first_directory_name=first_directory_name,second_directory_name=second_directory_name
            count01 = count01+1
    def get_books_from(self, listpage):

        r = rq.get(listpage, headers=self.headers,  timeout=5).text  # proxies=self.proxies,
        soup = bs(r, 'html.parser')
        print(listpage)
        try:
            self.max_page = int(soup.find('div', class_="pagination").find_all('b')[0].text.strip())
        except:
            print("do not have max page")
        urls = [listpage + '_0_0_11_0_1_{}_0_0/'.format(str(i)) for i in range(1, self.max_page + 1)]
        book_detail_url = ''
        for url in urls:
            self.qurl.put(url)  # 生成URL存入队列，等待其他线程提取

    def get_info(self):

        while not self.qurl.empty():  # 保证url遍历结束后能退出线程
            url = self.qurl.get()  # 从队列中获取URL
            try:
                r = rq.get(url, headers=self.headers,  timeout=5).text
                soup = bs(r, 'html.parser')
                try:
                    books = soup.select('div.bookList > ul > li')
                except:
                    print("there is no booklist")
                for bk in books:
                    nt = bk.select('div.cover > a')[0]
                    # 进入、书籍详细页面
                    self.book_detail_url = self.web_site + nt['href']

                    r2 = rq.get(self.book_detail_url, timeout=5).text  # headers=headers,proxies=self.proxies,
                    soup2 = bs(r2, 'html.parser')
                    self.cat = (soup2.find_all('div', class_='crumsItem')[2].find_all('a')[-1]).text  # 获取二级目录

                    try:
                        self.book_special_detail = soup2.select('div.specialist > p')[0].text  # 在图书详细页面上获取图书内容
                    except:
                        try:
                            self.book_special_detail = soup2.select('div.brief > p')[0].text  # 在图书详细页面上获取图书内容
                        except:
                            try:
                                self.book_special_detail = soup2.select('div.excerpt > p')[0].text
                            except:
                                pass

                    book_cover_image_url = soup2.select('div.coverImg > a > img')[0]['src']
                    # 读取并存储图片
                    book_name = nt['title']

                    self.get_image(url=book_cover_image_url, img_name=book_name)
                    self.desc = bk.select('p.recoLagu')[0].text
                    self.desc = re.sub('\s+', ' ', self.desc)
                    # label表示该文本的标签
                    data = [self.first_directory_name, self.cat, nt['title'], self.desc, self.cat,
                            self.book_special_detail]
                    print(data[0], data[1], data[2])

                    self.book_result.write("\t".join(data) + '\n')
            except:
                print('there no items in this {} page '.format(self.book_detail_url))

    @run_time
    def run(self):
        r = rq.get(self.All_categories, headers=self.headers, timeout=5).text  # headers=headers,
        soup02 = bs(r, 'html.parser')
        self.first_directory_list = soup02.select('div.w1200>h2>a')  # 获取一级目录分类
        self.second_directory_list = soup02.select('div.w1200>ul')  # 获取二级目录类别

        for directory in self.second_directory_list:  # 遍历所有的二级目录
            self.produce_url(directory)

            ths = []
            for _ in range(self.thread_num):
                th = Thread(target=self.get_info)
                th.start()
                ths.append(th)
            for th in ths:
                th.join()
            print('Data crawling is finished.')


if __name__ == '__main__':
    Spider().run()

