# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


# class Image360Pipeline(object):
#     def process_item(self, item, spider):
#         return item

import pymongo
import pymysql
from scrapy import Request
from scrapy.exceptions import DropItem
from scrapy.pipelines.images import ImagesPipeline


class MongoPipeline():
    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DB')
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def process_item(self, item, spider):
        self.db[item.collection].insert(dict(item))  # item.collection就是image,在items.py的ImageItem类中有定义
        return item

    def close_spider(self, spider):
        self.client.close()

class MysqlPipeline():
    def __init__(self, host, database, user, password, port):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port

    @classmethod
    def from_crawler(cls,crawler):
        return cls(
            host=crawler.settings.get("MYSQL_HOST"),
            database=crawler.settings.get('MYSQL_DATABASE'),
            user=crawler.settings.get('MYSQL_USER'),
            password=crawler.settings.get('MYSQL_PASSWORD'),
            port = crawler.settings.get('MYSQL_PORT')
        )

    def open_spider(self,spider):
        self.db = pymysql.connect(self.host, self.user, self.password, self.database,
                                  self.port, charset='utf8')
        self.cursor = self.db.cursor()

    def close_spider(self,spider):
        self.db.close()

    def process_item(self,item,spider):
        data = dict(item)
        keys = ','.join(data.keys())
        values = ','.join(['%s']*len(data))
        sql = 'insert into %s (%s) values (%s)'%(item.table,keys,values)
        self.cursor.execute(sql, tuple(data.values()))
        self.db.commit()
        return item


