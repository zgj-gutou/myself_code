import requests
from urllib.parse import urlencode

def get_page(offset):
    params = {
        "offset":offset,
        "format":"json",
        # "keyword":"%E8%B7%AF%E4%BA%BA",
        "keyword":"路人",  #　注意这里直接写中文就可以
        "autoload":"true",
        "count":'20',
        "cur_tab":"3",
    }
    url = 'https://www.toutiao.com/search_content/?'+urlencode(params)+'&from=gallery&pd='
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # print(response.json())
            return response.json()
    except requests.ConnectionError:
        print("error")
        return None

def get_images(json):
    if json.get('data'):
        for item in json.get('data'):
            title = item.get('title')
            images = item.get('image_list')
            for image in images:
                yield {   # 返回图片链接和图片所属的标题
                    'image':"http:"+image.get('url'),
                    'title':title
                }

import os
from hashlib import md5
def save_image(item):
    if not os.path.exists(item.get('title')):
        os.mkdir(item.get('title'))
    try:
        response = requests.get(item.get('image'))  # 请求图片的网址链接
        if response.status_code == 200:
            file_path = '{0}/{1}.{2}'.format(item.get('title'),md5(response.content).hexdigest(),'jpg')
            # 0表示文件夹名，1表示文件名，用图片内容的md5值，2表示jpg格式
            if not os.path.exists(file_path):
                with open(file_path,'wb') as f:
                    f.write(response.content)
            else:
                print("Already downloaded",file_path)
    except requests.ConnectionError:
        print('Failed to Save Image')

from multiprocessing.pool import Pool  # python进程池

def main(offset):
    json = get_page(offset)
    for item in get_images(json):
        print(item)
        save_image(item)

GROUP_START = 0
GROUP_END = 20

if __name__ == '__main__':
    pool = Pool()
    groups = ([x * 20 for x in range(GROUP_START, GROUP_END+1)])
    pool.map(main,groups)
    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用