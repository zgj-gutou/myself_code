# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import item,Field


class ImageItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # pass
    collection = table = 'images'
    # collection代表mongodb的collecion，则images就是collecion名称，
    # table表示mysql的表，则images就是表名称
    id = Field()
    url = Field()
    title = Field()
    thumb = Field()
