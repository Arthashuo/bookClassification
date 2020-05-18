from bs4 import BeautifulSoup as bs
import requests as  rq
import urllib.request
import time
import  re
import random
import base64
import csv
from urllib.parse import quote
import string
from urllib import request

csvfile =  open('/Users/leonjiang/Downloads/dialogue/data/book/book.csv', 'a+', encoding='utf8', newline='')
f = csv.writer(csvfile)
book_urls = open('/Users/leonjiang/Downloads/dialogue/data/book/url.txt', 'a+')
links = [
        # "http://www.bookschina.com/kinder/53000000/", "http://www.bookschina.com/kinder/54000000",
    #    "http://www.bookschina.com/kinder/46000000", "http://www.bookschina.com/kinder/47000000",
    #    "http://www.bookschina.com/kinder/36000000", "http://www.bookschina.com/kinder/17000000",
    #    "http://www.bookschina.com/kinder/48000000", "http://www.bookschina.com/kinder/37000000",
    #    "http://www.bookschina.com/kinder/14000000", "http://www.bookschina.com/kinder/52000000",
    #    "http://www.bookschina.com/kinder/57000000", "http://www.bookschina.com/kinder/49000000",
    #    "http://www.bookschina.com/kinder/23000000", "http://www.bookschina.com/kinder/16000000",
    #    "http://www.bookschina.com/kinder/29000000", "http://www.bookschina.com/kinder/39000000",
    #    "http://www.bookschina.com/kinder/43000000", "http://www.bookschina.com/kinder/42000000",
    #    "http://www.bookschina.com/kinder/38000000", "http://www.bookschina.com/kinder/60000000",
    #    "http://www.bookschina.com/kinder/21000000", "http://www.bookschina.com/kinder/34000000",
    #    "http://www.bookschina.com/kinder/24000000", "http://www.bookschina.com/kinder/20000000",
    #    "http://www.bookschina.com/kinder/12000000", "http://www.bookschina.com/kinder/11000000",
    #    "http://www.bookschina.com/kinder/51000000", "http://www.bookschina.com/kinder/18000000",
    #    "http://www.bookschina.com/kinder/61000000", "http://www.bookschina.com/kinder/62000000",
       "http://www.bookschina.com/kinder/64000000", "http://www.bookschina.com/kinder/27000000",
       "http://www.bookschina.com/kinder/30000000", "http://www.bookschina.com/kinder/22000000",
       "http://www.bookschina.com/kinder/56000000", "http://www.bookschina.com/kinder/45000000",
       "http://www.bookschina.com/kinder/50000000", "http://www.bookschina.com/kinder/35000000",
       "http://www.bookschina.com/kinder/31000000", "http://www.bookschina.com/kinder/63000000"]
# testP =  'http://category.dangdang.com/cp01.54.00.00.00.00.html'  #大的分类页面，用于测试get_books_from()函数。
# #根据页码构造页码列表。可以作为get_books_from()的参数。
# links = ['https://book.douban.com/tag/{}'.format(j) for j in cat]
# links = [j + '?start={}&type=T'.format(str(i * 20)) for i in range(383) for j in links]
user_agent_list = []
f = open('/Users/leonjiang/Downloads/dialogue/douban/user_agent.txt', 'r')
for date_line in f:
    user_agent_list.append(date_line.replace('\r\n','').replace('\n',''))
proxyHost = "http-dyn.abuyun.com"
proxyPort = "9020"
proxyUser = 'H6LU025735440VOD'
proxyPass = 'C2AA06CFDEBBF8FD'
proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
    "host" : proxyHost,
    "port" : proxyPort,
    "user" : proxyUser,
    "pass" : proxyPass,
}

proxy_handler = request.ProxyHandler({
    "http"  : proxyMeta,
    "https" : proxyMeta,
})
opener = request.build_opener(proxy_handler)
request.install_opener(opener)
proxies = { "http": proxyMeta,  "https": proxyMeta }
# cookie = '_ga=GA1.2.250175391.1555641484; _octo=GH1.1.1313679600.1555641484; _device_id=4a8ed8a88943d308b85d69d8bc45c08b; experiment:homepage_signup_flow=eyJ2ZXJzaW9uIjoiMSIsInJvbGxPdXRQbGFjZW1lbnQiOjc5LjA1NzI4NjkwNzY5OTI1LCJzdWJncm91cCI6bnVsbCwiY3JlYXRlZEF0IjoiMjAyMC0wMy0xMlQwOTozNjowNC45OTFaIiwidXBkYXRlZEF0IjoiMjAyMC0wMy0xMlQwOTozNjowNC45OTFaIn0=; user_session=5a8ntUa_Qw3cWTJ976DFcTVJHoF2nkVxhD05mAOUVHO0ZHQP; __Host-user_session_same_site=5a8ntUa_Qw3cWTJ976DFcTVJHoF2nkVxhD05mAOUVHO0ZHQP; logged_in=yes; dotcom_user=EricAugust; tz=Asia%2FShanghai; has_recent_activity=1; _gat=1; _gh_sess=tL22xKdOVGaFd8RwKCpTFCicyh3OaxoaS6qNbIGhDOJjQjnGBQWXa2WjeHvbDtASxXk4HZ9Cv9pjemTojlt10KyKKtXWNI0sa3CWL9oj2x6bDto0MAnw4PtbF6nUIeyJ1lgIjv0ZWvfps7azksWbbiBswYupZd3v%2FgzhGBiZJSy8ypUOJyrcnTS634IbWjgkyb4U5cueQDaJ5Klss2OHgw3b3YEjbq5mKt06EFDW7nYy%2BCVmWW3%2F7SKW5V4KhbVznyPTI4hjjREvsn9%2FyHfMTFDriFMwPWAN0uzmas3S8doKyBT56pM6g2RwC06lNsmxbm8cW8Gzszll7iffeahOwWo7fKZRMyiRoN0f6ysa5AcvExXVvqxYmuoCzDpFAmk0fQjio3CFgJbXORb1E17GC40EJAQ105FAfLPJog6%2BGmt%2FlzLZh%2BZAn%2FkpZ%2FAH8x5yCrdkFAS7zt5k3sP9URUd%2FcrLLihylu3FAW8FgCh070wJSDeNsw7bH8aXVchNg6Ts%2Fu1wldFhBl%2BD9Qb6OhfoTVx6WojfOjkgQpnmTUH%2F63Pyrjlfeae9oI0LkhsT6%2FPJEC%2B6xZzPqaiZGOtkhhyTKkOmulopZphtdh46PkTxXS4GacNJhD9qyU3OBRbG7zzdYSOxuiRAxAF8TxNjO8EmIcx2gtm%2FMYbH%2Bppnghkh6aakCWX%2By9TvAENv%2BecnO%2BLRcjyJuZnl5ee9zmfbRBSvIVWdYnArbTxBLQmnE%2FL0tWdxa6o2O9RvZrGtu2A3IuNHCJtgFp%2BnwtIFY88aivGueOJsOuwZyzHl9qdyQ7EVmsjrLAhBDpn8Vo8LlVj7bUdtRWsY5y7dW5%2Bybl9dLUN20YlXIigqsMyL7aMdGJDGUprlHV9HG8Hgfr2IEak1Os%2BjanhtHKWQfjM2pfJjkY%2F2rep9rqzqBvJHg1v9z%2BKl%2FLa20s%2FEGNbTKdIKNtFQU4rdM14JqPbPGHHN8sAMl5LoTq0%2FoE6x0ZRs9VU8tYTU7KTlcysoEQn54SuqPNyj2rcWZLRv6YS9uQmIpgeXP07P3td1sE1OvVb83D%2FkE54JOY6zldraNEvfq3LsYMYvlasZa%2Fn7ntge1un%2FjYJI1yiZPA3Gd6Bw4wZjw9OrTN0lBRMtb9zydaEqmfMWZ9TRtktzfAfvWxnp03tUowj%2F6pc%2BiJARq0Ex%2BgP2rgyrLNEIzrTAbmfbh6bdDtz4MQ%2FXMFSp6bSVB%2FygWkdRm%2FoufeXPZm6iCKj7uVpNy4YP7qjY3dFMZ5An9DFyu9h9rkuzij849ZrdpmowTQ9PJA8mNqKS3LxUe0ItEdLLuSiWMwXzFsoowmUIPrJEuM%2FBi3RfHD4kY7jjjh5ctLhW%2FsemEunitMFO75jNG2mTSosfxULD%2B2MfGkyADBMW7NpxK1EvD4tWhVIDJJ90rS9t8%2BbHvPwP227M1TfSLLZVfMjtp%2FM30bOUOAIOUOhyJO%2F%2BwXZ6AcS69QrjfeK0XyJb8cLcJg4zPb%2FIe8oTzM8MRwjrbZDeDyWQXToaMqpvXbrVxGqoN6Op7KYslq9JMX4aAqdkvQBQnkFsOu3C4dD2PxFgiu7hloNJoUGnJWEEbq6VBym%2BKObA5ZiCa61i%2Biq1EmRI%2FNFQWckyywaX5upB6JU9R4QjKpNeDZpl0JYZsJACC3jg%2FpM8WMz5eHi9uNa37yuhPJ47OKs3yZZP9xjHBB1SrrH%2F83cTREbz9FgKWIMIEKCa6cChlAp9ZEwww9E%2FgxxwJJVsxoxwIrJ%2F6pk22J56mpIrtIPoBqZQ%2FtjVOf3noG8wmy2PpARjPQaR1gCIReh5%2Fuk7uLFnVRUneQpUKI2RZ4f%2BPEVaUe5XrYdixLyTDeecu5MX9lhFLqKH15Io0vGn61%2FAmQ%3D%3D--kEa16PolgHfbNMtu--kmhP2S3DMjxca7wwI2RS8w%3D%3D'

# def cookies2dict(cookies):
#     items = cookies.split(';')
#     d = {}
#     for item in items:
#         kv = item.split('=',1)
#         k = kv[0]
#         v = kv[1]
#         d[k] = v
#     return d
# cookie = cookies2dict(cookie)


def  get_books_from(listpage):
    # proxy = random.choice(proxy_list)
    user_agent = random.choice(user_agent_list)
    # if len(user_agent) < 10:
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'

    headers = {
        "Origin":"https://book.douban.com",
        "Accept": "text/event-stream",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        # "Cookie": cookie,
        "User-Agent": user_agent}
    r = rq.get(listpage, headers=headers, timeout=5).text
    # s = quote(listpage, safe=string.printable)
    # req = urllib.request.Request(s, headers=headers, method='GET')
    # r = urllib.request.urlopen(req).read().decode("gbk")
    soup = bs(r, 'html.parser')
    max_page = int(soup.find('div', class_="pagination").find_all('b')[0].text.strip())
    cat = soup.select('head > title ')[0].text
    cat = cat[4:cat.find('书籍')]
    print(cat)
    print(max_page)
    urls = [listpage+'_0_0_11_0_1_{}_0_0/'.format(str(i)) for i in range(1, max_page +1 )]
    for  url in urls:
        try:
            # r = request.urlopen(url).read().decode("gbk")
            r = rq.get(url, headers=headers, timeout=5).text
            soup = bs(r, 'html.parser')
            books = soup.select('div.bookList > ul > li')
            for bk in books:
                nt = bk.select('div.cover > a')[0]
                desc = bk.select('p.recoLagu')[0].text
                desc = re.sub('\s+', ' ', desc)
                data = ['http://www.bookschina.com' + nt['href'], nt['title'], desc, cat]
                print(data[1])
                book_urls.write("\t".join(data) + '\n')

            ra = random.uniform(0,1)
            print('休眠等待：%.3f秒'%ra)
            time.sleep(ra)
        except:
            print('there no items in this {} page '.format(url))


def get_book_info(bookpage):        #特殊情况需要判断是否为404找不到，这里不存在这个问题。所以省略。
    # r = rq.get(bookpage, headers={'Connection':'close'})
    r = urllib.request.urlopen(bookpage).read().decode("gbk")
    soup = bs(r, 'html.parser')
    name = soup.find('div', class_='name_info').find('h1')['title']  # 找到书的名称所在位置
    info = soup.find('div',class_='messbox_info').find_all('span')
    author = info[0].text
    press = info[1].text
    pubdate = info[2].text
    descrip = soup.find('ul',class_='key clearfix').text
    # bookid = bookpage.split('.')[-2].split('/')[1]          # 后面要用书id号构造js请求。
    # price_url = 'http://product.dangdang.com/index.php?r=callback%2Fproduct-info&productId={}' \
    #             '&isCatalog=0&shopId=0&productType=0'.format(bookid)
    time.sleep(0.1)  # 比第1次页面请求推迟0.1秒
    # res = rq.get(price_url)
    # pj = res.json()
    # newprice = pj['data']['spu']['price']['salePrice']
    # oldprice = pj['data']['spu']['price']['originalPrice']

    list = soup.find_all('a',{ 'name':'__Breadcrumb_pub'})
    target_1 = list[-2].text
    target_2 = list[-1].text
    print(target_1, target_2)
    cate = list[-1]['href'].split('/')[-1][2:20]          #得到图书所在分类的字符串。构造JS请求获取评论数
    # cateurl = 'http://product.dangdang.com/index.php?r=comment%2Flist&productId={}&categoryPath={}' \
    #           '&mainProductId={}&mediumId=0&pageIndex=1&sortType=1&filterType=1&isSystem=1&tagId=0&' \
    #           'tagFilterCount=0'.format(bookid,cate,bookid)
    time.sleep(0.1)          #比获取价格的JS请求再推迟0.1秒
    # cRes =  rq.get(cateurl)
    # cj = cRes.json()
    # totalcmt = cj['data']['summary']['total_comment_nm']   #得到评论总数
    # goodRate = cj['data']['summary']['goodRate']            #得到好评率，不带%。

    bkdict = {'name':name, 'author':author[3:],'pubDate':pubdate[5:-1], 'Press':press[4:],
              'target_1':target_1,  'target_2':target_2,'detail':descrip[1:]}
    print('write {} to file'.format(bkdict['name']))
    itemlist = [bookpage, ((bkdict['name'])).replace('\n','').replace(' ','').replace('当当','**'),
         ((bkdict['author'])).replace('\n','').replace(' ',''),
          ((bkdict['pubDate'])).replace('\n','').replace(' ',''),
           ((bkdict['Press'])).replace('\n','').replace(' ',''),
            ((bkdict['target_1'])).replace('\n','').replace(' ',''),
             ((bkdict['target_2'])).replace('\n','').replace(' ',''),
              ((bkdict['detail'])).replace('\n','').replace(' ','')]
    f.writerow(itemlist)
    # book_info.insert_one(dict)


#第一步，从构造好的链接列表中依次取出，抽取书的详情页的网址，保存到数据库。
for  link in links:
    get_books_from(link)
    ra = random.uniform(2,4)
    print('休眠等待：%.3f秒'%ra)
    time.sleep(ra)
print('经济——门类全部收录完毕！')
#第二步，循环打开详情页，爬取图书的详细信息。在此仅用1个商品页面作为示例。
#发现收集到的图书链接有重复的，所以后面批量抓取时，必须进行去重。
# get_book_info('http://product.dangdang.com/23274638.html')