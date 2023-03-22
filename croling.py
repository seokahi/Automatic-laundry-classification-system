import urllib.request
from bs4 import BeautifulSoup
from PIL import Image
import os

# 접근할 페이지 번호
pageNum = 1

# 저장할 이미지 경로 및 이름  (download/무신사socks폴더 안에 socks이름으로..)
imageNum = 492
new_path = 'C:/Users/Seokahi/Desktop/system/shirt'  # 이동할 경로
if not os.path.exists(new_path):  # 이동할 경로가 없으면 생성
    os.makedirs(new_path)

new_fname = "shirt"
imageStr = os.path.join(new_path, new_fname)

# 몇페이지까지 넘어갈건지 정하고, url넣어주기
while pageNum < 9:
    # url.. 원하는 페이지 링크 복사해오고,페이지에 해당하는 부분 찾아서 pageNum넣어주기
    url = "https://www.musinsa.com/categories/item/001002?d_cat_cd=001002&brand=filluminate%2Clafudgestore%2Cthomasmore%2Cdiamondlayla%2Cpartimento%2Ccovernat%2Ctrillion%2Csuare%2Cromanticcrown%2Cmahagrid%2Cgoodlifeworks%2Csoup&list_kind=big&sort=pop_category&sub_sort=&page=" + \
        str(pageNum)+"&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure="
    fp = urllib.request.urlopen(url)
    source = fp.read()
    fp.close()

    soup = BeautifulSoup(source, 'html.parser')
    # 크롬에서 링크 열고 F12눌러서 저장할 이미지 있는 부분 찾아내기
    soup = soup.findAll("a", class_="img-block")

    # 이미지 경로를 받아 로컬에 저장한다.
    for i in soup:
        img = i.find("img")
        if "data-original" in img.attrs:
            imageNum += 1
            http = "https:"
            imgURL = img["data-original"]
            fullURL = http + imgURL
            fname = "{}_{:04d}.jpg".format(imageStr, imageNum)
            urllib.request.urlretrieve(fullURL, fname)
            print(imgURL)
            print(imageNum)
        else:
            print("data-original not found, skipping...")
    pageNum += 1
