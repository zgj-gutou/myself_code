# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymongo

class DangdangPipeline(object):
    def process_item(self, item, spider):
        return item

class MongoPipeline(object):
    def __init__(self,mongo_uri,mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls,crawler):
        return cls(mongo_uri = crawler.settings.get('MONGO_URI'),mongo_db = crawler.settings.get('MONGO_DB'))

    def open_spider(self,spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def process_item(self,item,spider):
        self.db[item.collection].insert(dict(item))
        return item

    def close_spider(self,spider):
        self.client.close()

class TextPipeline(object):
    def __init__(self,file_name):
        self.file_name = file_name

    @classmethod
    def from_crawler(cls,crawler):
        return cls(file_name = crawler.settings.get('FILE_NAME'))

    def process_item(self,item,spider):
        with open(self.file_name,'a',encoding='utf-8') as f:
            f.write("title: "+str(item['title'])+" price: "+str(item['price'])+" comment: "+str(item['comment']))
            f.write('\n')





