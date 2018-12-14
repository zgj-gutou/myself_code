from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from urllib.parse import quote
from pyquery import PyQuery as pq

browser = webdriver.Chrome()
wait = WebDriverWait(browser,10)
flag = 0

def skip_page():
    global flag
    try:
        url = 'https://www.yiwise.com/'
        browser.get(url)
        get_functions()  # 爬取第一个网页
        if flag == 1:
            submit = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '#header .header-wrap .index-header_entrance [href="core"]')))
            submit.click()
            get_articles()  # 爬取第二个网页
    except TimeoutException:
        skip_page()

def get_functions():
    html = browser.page_source
    doc = pq(html)
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
        for content in contents.items():   # 记得要加items()
            con.append(content.text())
            # print(content)
        all_contents = ""
        for i in range(len(con)):
            all_contents = all_contents + con[i]
        print(all_contents)
        file.write(all_contents)
        file.write("\n")
    file.close()
    global flag
    flag = 1  # 说明这里爬取完成，到下一个网页爬取

def get_articles():
    html = browser.page_source
    doc = pq(html)
    # print(doc('#content .page-core .section-list.section .list-ul .list-li'))
    items = doc('#content .page-core .section-list.section .list-ul .list-li').items()  # 都是list-li开头，所以用遍历即可
    file = open('articles.txt', 'a', encoding='utf-8')
    for item in items:
        item_articles = item('.up-card .item-ul').children()  # 这里只能用children,而不能用(li)(不知道为什么不能这么用,可能是因为li没有class)
        for item_article in item_articles.items():
            item_names = item_article(".strong").items()  # 有的strong的地方有多个名字，所以这里重新再选一下，不能直接text(),否则多个名字会合成一个字符串
            # print(item_names)
            names = []
            for j in item_names:
                names.append(j.text())
            # print(names)
            text = ""
            parts = item_article.text().split('\n')
            for i in range(len(parts)):
                for k in range(len(names)):
                    if names[k]==parts[i]:
                        parts[i] = " "+ parts[i] + ", "  # 有的名字后面没有逗号，所以这里加上逗号
                text=text+parts[i]
            print(text)
            file.write(text)
            file.write("\n")
    file.close()

def main():
    skip_page()
    browser.close()

if __name__ == '__main__':
    main()