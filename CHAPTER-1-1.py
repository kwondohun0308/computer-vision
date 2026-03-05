import pandas as pd
import numpy as np
import cv2 as cv
import sys

img = cv.imread('soccer.jpg')
if img is None:
    print('Image load failed!')
    sys.exit()

img_small = cv.resize(img, (0, 0), fx=0.5, fy=0.5) #이미지 크기를 반으로 감소
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #이미지를 흑백으로 변환
gray_small = cv.resize(gray, (0, 0), fx=0.5, fy=0.5) #흑백으로 변환한 이미지 크기를 반으로 감소

# 원본 이미지는 3채널이므로, 그레이스케일 이미지를 3채널로 변환하여 원본과 연결
gray_3ch = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)
combined = np.hstack((img_small, gray_3ch))

cv.imshow('combined', combined)
cv.waitKey()
cv.destroyAllWindows()

