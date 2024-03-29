from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

import flickrapi

# APIキーの情報
key = "3d90ad55c94617f185bbdc8830dff836"
secret = "60b6fc89a34d306d"
wait_time = 1

# 保存フォルダの指定
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format="parsed-json")
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_q, licence"
)

photos = result['photos']
# pprint(photos) # 返り値を表示する

for i, photo in enumerate(photos['photo']):
    url_q = photo["url_q"]
    filepath = savedir + "/" + photo['id'] + ".jpg"

    if os.path.exists(filepath):continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)


