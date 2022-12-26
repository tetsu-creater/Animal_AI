
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# 画像の読み込み

X = []
Y = []
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")

    for i, file in enumerate(files):
        if i > 200: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size)) #リサイズする
        data = np.asarray(image) 
        X.append(data)
        Y.append(index)

X = np.array(X) #リスト形式からNumpy形式に
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y) #データを3:1で分割
xy = (X_train, X_test, Y_train, Y_test)
np.save("./animal.npy", xy) #画像をNumpy形式にして出力
