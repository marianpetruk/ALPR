import sys
import requests
import urllib.request
import os
from bs4 import BeautifulSoup
from PIL import Image
import io


def parse_page(page_url):
    r = requests.get(page_url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')

    imgs_srcs = []
    plate_nums = []
    relevant_blocks = soup.find_all("a", class_="com-car-num2-withoutbg")
    for block in relevant_blocks:
        img = block.find_all("img")
        if img:
            img = img[0]
            imgs_srcs.append(img['src'])
            plate_nums.append(img['title'])
    return imgs_srcs, plate_nums


def image_downloader(image_url, filename):
    r = requests.get(image_url)
    bio = io.BytesIO(r.content)
    if bio.getbuffer().nbytes < 1000:
        return
    with open(filename, 'wb') as f:
        f.write(r.content)


if __name__ == "__main__":
    # com-car-num2-

    image_downloader("http://media.autocarma.ua/p/plates/AP/94/51/AE/140.png", "img.jpg")
    all_imgs_sources = []
    all_plates_nums = []
    page = int(sys.argv[1])
    print("START PARSING")
    while page < int(sys.argv[2]):
        print("PAGE:", page)
        imgs_srcs, plate_nums = parse_page(f"http://autocarma.ua/comments/page/{page}")
        all_imgs_sources.extend(imgs_srcs)
        all_plates_nums.extend(plate_nums)
        page += 48

    print("START SAVING")
    PLATES_FOLDER = "plates"
    EXT = "png"
    if PLATES_FOLDER not in os.listdir("./"):
        os.mkdir("plates")

    for index in range(len(all_imgs_sources)):
        print("IMAGE:", index+1)
        src = all_imgs_sources[index]
        plate_num = all_plates_nums[index]
        image_downloader(src, f"{PLATES_FOLDER}/{plate_num}.{EXT}")

