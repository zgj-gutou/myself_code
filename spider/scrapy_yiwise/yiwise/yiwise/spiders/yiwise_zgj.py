# -*- coding: utf-8 -*-
import scrapy
from pyquery import PyQuery as pq
from scrapy import Request,Spider
# flag = 0

class YiwiseZgjSpider(scrapy.Spider):
    name = 'yiwise_zgj'
    allowed_domains = ['www.yiwise.com']
    start_urls = 'https://www.yiwise.com/'   # 注意这里是字符串，如果是默认生成的文件则是一个list
    flag =0

    def start_requests(self):   # 用于发送请求给DownloadMiddleware(在middlewares.py文件中）
        for i in range(2):
            url = self.start_urls
            print("nihao")
            yield Request(url = url,callback  = self.parse,meta={'i':i},dont_filter=True)

    def parse(self, response):   # parse函数里直接把存数据的操作也写进去了，这样就不用去管items.py和pipelines.py了。。
        doc = pq(response.css('*').extract()[0])
        print("flag:",self.flag)
        if self.flag == 0:
            items = doc("#content .section-3 .list .item").items()
            file = open('functions.txt', 'a', encoding='utf-8')
            for item in items:
                # print(i)
                title = item('.item').find('.title.gradient-text').text()
                title = title + ": "
                # print(title)
                contents = item('.item').children().children()  # 不知道这里为什么不能根据节点名称ul选子节点。。
                # print(contents)
                con = [title]
                for content in contents.items():  # 记得要加items()
                    con.append(content.text())
                    # print(content)
                all_contents = ""
                for i in range(len(con)):
                    all_contents = all_contents + con[i]
                print(all_contents)
                file.write(all_contents)
                file.write("\n")
            file.close()
            self.flag = 1  # 说明这里爬取完成，到下一个网页爬取
        else:
            items = doc('#content .page-core .section-list.section .list-ul .list-li').items()  # 都是list-li开头，所以用遍历即可
            file = open('articles.txt', 'a', encoding='utf-8')
            for item in items:
                item_articles = item(
                    '.up-card .item-ul').children()  # 这里只能用children,而不能用(li)(不知道为什么不能这么用,可能是因为li没有class)
                for item_article in item_articles.items():
                    item_names = item_article(
                        ".strong").items()  # 有的strong的地方有多个名字，所以这里重新再选一下，不能直接text(),否则多个名字会合成一个字符串
                    # print(item_names)
                    names = []
                    for j in item_names:
                        names.append(j.text())
                    # print(names)
                    text = ""
                    parts = item_article.text().split('\n')
                    for i in range(len(parts)):
                        for k in range(len(names)):
                            if names[k] == parts[i]:
                                parts[i] = " " + parts[i] + ", "  # 有的名字后面没有逗号，所以这里加上逗号
                        text = text + parts[i]
                    print(text)
                    file.write(text)
                    file.write("\n")
            file.close()


