# -*- coding: utf-8 -*-
from scrapy import Request,Spider
from pyquery import PyQuery as pq
from ..items import DangdangItem

class DangdangZgjSpider(Spider):
    name = 'dangdang_zgj'
    allowed_domains = ['www.dangdang.com']
    base_url_1 = 'http://search.dangdang.com/?key='
    base_url_2 = '&act=input&page_index='

    def start_requests(self):
        for page in range(1,self.settings.get("MAX_PAGE")+1):
            url = self.base_url_1 + self.settings.get("KEYWORDS")+self.base_url_2+str(page)
            yield Request(url = url, callback=self.parse, meta={"page":page},dont_filter = True)

    def parse(self, response):
        doc = pq(response.css('*').extract()[0])
        items = doc('#bd .con #12808 #12810 .con #search_nature_rg .bigimg').children().items()
        for product in items:
            # print(item)
            item = DangdangItem()
            # product = {
            item['title']=product.find('.pic').attr('title'),
            item['price']=product.find('.price .search_now_price').text(),
            item['comment']= product.find('.search_star_line .search_comment_num').text(),
            # }
            yield item
            # print(product)
            # save_to_mongo(product)
