#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import io
import tensorflow.keras
import sklearn
import pydot
import graphviz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import visualkeras

from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import IPython
import scikitplot as skplt


# In[20]:


#!pip install opencv-python
#!pip install opencv-contrib-python
#!pip install Pillow
#pip3 install bpmll


# ## csv 데이터 불러오기

# In[2]:


train = pd.read_csv("train.csv")


# In[3]:


train.head()


# In[4]:


train.info()


# In[51]:


sample_1= train[(train["J"] > 1)]
len(sample_1)


# In[5]:


block_1= train[(train['A'] == 1) & (train['B'] == 0)
               & (train['C'] == 0) & (train['D'] == 0)
               & (train['E'] == 0) & (train['F'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_1)


# In[29]:


block_1


# In[8]:


block_2= train[(train['B'] == 1) & (train['A'] == 0)
               & (train['C'] == 0) & (train['D'] == 0)
               & (train['E'] == 0) & (train['F'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_2)


# In[10]:


block_3= train[(train['C'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['D'] == 0)
               & (train['E'] == 0) & (train['F'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_3)


# In[21]:


block_4= train[(train['D'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['E'] == 0) & (train['F'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_4)


# In[27]:


block_5= train[(train['E'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['F'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_5)


# In[15]:


block_6= train[(train['F'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['E'] == 0)
               & (train['G'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_6)


# In[29]:


block_7= train[(train['G'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['E'] == 0)
               & (train['F'] == 0) & (train['H'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_7)


# In[30]:


block_8= train[(train['H'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['E'] == 0)
               & (train['F'] == 0) & (train['G'] == 0)
               & (train['I'] == 0) & (train['J'] == 0)
              ]

len(block_8)


# In[31]:


block_9= train[(train['I'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['E'] == 0)
               & (train['F'] == 0) & (train['G'] == 0)
               & (train['H'] == 0) & (train['J'] == 0)
              ]

len(block_9)


# In[32]:


block_10= train[(train['J'] == 1) & (train['A'] == 0)
               & (train['B'] == 0) & (train['C'] == 0)
               & (train['D'] == 0) & (train['E'] == 0)
               & (train['F'] == 0) & (train['G'] == 0)
               & (train['H'] == 0) & (train['I'] == 0)
              ]

len(block_10)


# In[3]:


drop_train = train.drop_duplicates(subset=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                   keep='first', inplace=False, ignore_index=False)


# In[4]:


pd.options.display.max_rows = 964
drop_train


# In[72]:


len(drop_train)


# In[73]:


df = pd.concat([train, drop_train])
df = df.reset_index(drop = True)
df


# In[74]:


col = list(df.columns)[2:]
col


# In[79]:


df_grp = df.groupby(col)
df_di = df_grp.groups
print(type(df_di))
df_di


# In[78]:


len(df_di)


# In[9]:


groups = train.groupby(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
groups.size()


# In[13]:


print(groups.size == 1)


# In[18]:


#1110011111
#1111111011

img_con1= train[(train['A'] == 1) & (train['B'] == 1)
               & (train['C'] == 1) & (train['D'] == 0)
               & (train['E'] == 0) & (train['F'] == 1)
               & (train['G'] == 1) & (train['H'] == 1)
               & (train['I'] == 1) & (train['J'] == 1)
              ]
img_con1

img_con2= train[(train['A'] == 1) & (train['B'] == 1)
               & (train['C'] == 1) & (train['D'] == 1)
               & (train['E'] == 1) & (train['F'] == 1)
               & (train['G'] == 1) & (train['H'] == 0)
               & (train['I'] == 1) & (train['J'] == 1)
              ]
img_con2


# In[82]:


len(groups.size())


# In[92]:


len(train.loc[train['A'] == 1])


# In[93]:


len(train.loc[train['B'] == 1])


# In[94]:


len(train.loc[train['C'] == 1])


# In[95]:


len(train.loc[train['D'] == 1])


# In[96]:


len(train.loc[train['E'] == 1])


# In[97]:


len(train.loc[train['F'] == 1])


# In[98]:


len(train.loc[train['G'] == 1])


# In[99]:


len(train.loc[train['H'] == 1])


# In[100]:


len(train.loc[train['I'] == 1])


# In[101]:


len(train.loc[train['J'] == 1])


# ### 이미지 데이터 살펴보기

# In[3]:


img_files = glob.glob('.//train//*.jpg')


# In[7]:


index = 0

image_bgr = cv2.imread(img_files[index], cv2.IMREAD_COLOR)
image_bgr[0,0] # 픽셀을 확인
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # RGB로 변환
plt.imshow(image_rgb), plt.axis("off") # 이미지를 출력
plt.show()


# ### 이미지 크기 변경(reshape)

# In[4]:


image_re = cv2.resize(image_rgb, (224, 224))
plt.imshow(image_re), plt.axis("off") # 이미지를 출력
plt.show()


# In[102]:


image_yolo = cv2.resize(image_rgb, (416, 416))
plt.imshow(image_yolo), plt.axis("off") # 이미지를 출력
plt.show()


# ### Blurry 기법

# In[15]:


image_blurry = cv2.blur(image_re, (15,15))

plt.imshow(image_blurry, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()


# In[17]:


kernel = np.ones((15,15)) / 225.0 # 커널을 만듭니다.
kernel # 커널을 확인
image_kernel = cv2.filter2D(image_re, -1, kernel) # 커널을 적용
plt.imshow(image_kernel), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()


# In[9]:


image_very_blurry = cv2.GaussianBlur(image_re, (15,15), 0) # 가우시안 블러를 적용
plt.imshow(image_very_blurry), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()


# In[21]:


kernel = np.array([[-6, -6, -6],
                   [-6,49,-6],
                   [-6, -6, -6]]) # 커널을 만듭니다.

# 이미지를 선명하게 만듭니다.
image_sharp = cv2.filter2D(image_very_blurry, -1, kernel)

plt.imshow(image_sharp), plt.axis("off") # 이미지 출력
plt.show()


# ### 배경 제거

# In[26]:


# 사각형 좌표: 시작점의 x,y  ,넢이, 너비
rectangle = (0, 56, 256, 150)

# 초기 마스크 생성
mask = np.zeros(image_sharp.shape[:2], np.uint8)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(image_sharp, # 원본 이미지
           mask,       # 마스크
           rectangle,  # 사각형
           bgdModel,   # 배경을 위한 임시 배열
           fgdModel,   # 전경을 위한 임시 배열 
           5,          # 반복 횟수
           cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화

# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱하여 배경을 제거
image_sharp_nobg = image_sharp * mask_2[:, :, np.newaxis]

# plot
plt.imshow(image_sharp_nobg)
plt.show()


# ### 경계선 감지

# In[28]:


plt.imshow(image_sharp_nobg), plt.axis("off") # 이미지 출력
plt.show()

median_intensity = np.median(image_sharp_nobg) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정합니다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용합니다.
image_canny = cv2.Canny(image_sharp_nobg, lower_threshold, upper_threshold)

plt.imshow(image_canny), plt.axis("off") # 이미지 출력
plt.show()


# ### 복합 블록 구조에 대한 전처리 연구

# In[163]:


index = 17036

image_bgr_17036 = cv2.imread(img_files[17036], cv2.IMREAD_COLOR)
image_rgb_17036 = cv2.cvtColor(image_bgr_17036, cv2.COLOR_BGR2RGB) # RGB로 변환
plt.imshow(image_rgb_17036), plt.axis("off") # 이미지를 출력
plt.show()


# In[165]:


# 컬러 영상의 히스토그램 평활화
src = cv2.imread('field.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()
    
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
ycrcb_planes = cv2.split(src_ycrcb)


# 밝기 성분에 대해서만 히스토그램 평활화 수행
ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])

dst_ycrcb = cv2.merge(ycrcb_planes)
dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()


# In[ ]:





# In[68]:


image_re_17036 = cv2.resize(image_rgb_17036, (200, 200))
plt.imshow(image_re_17036), plt.axis("off") # 이미지를 출력
plt.show()

# 사각형 좌표: 시작점의 x,y  ,넢이, 너비
rectangle = (0, 56, 256, 150)

# 초기 마스크 생성
mask_17036 = np.zeros(image_re_17036.shape[:2], np.uint8)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(image_re_17036, # 원본 이미지
            mask_17036,       # 마스크
            rectangle,  # 사각형
            bgdModel,   # 배경을 위한 임시 배열
            fgdModel,   # 전경을 위한 임시 배열 
            20,          # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화

# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_17036_2 = np.where((mask_17036==2) | (mask_17036==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱하여 배경을 제거
image_17036_sharp_nobg1 = image_re_17036 * mask_17036_2[:, :, np.newaxis]

# plot
plt.imshow(image_17036_sharp_nobg1)
plt.show()


# In[64]:


plt.imshow(image_rgb_17036), plt.axis("off") # 이미지 출력
plt.show()

median_intensity = np.median(image_rgb_17036) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정합니다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용합니다.
image_canny = cv2.Canny((image_rgb_17036), lower_threshold, upper_threshold)

plt.imshow(image_canny), plt.axis("off") # 이미지 출력
plt.show()


# In[39]:


image_re_17036 = cv2.resize(image_rgb_17036, (200, 200))
plt.imshow(image_re_17036), plt.axis("off") # 이미지를 출력
plt.show()


# In[56]:


kernel = np.array([[-4, -4, -4],
                   [-4,33,-4],
                   [-4, -4, -4]]) # 커널을 만듭니다.

# 이미지를 선명하게 만듭니다.
image_17036_sharp = cv2.filter2D(image_re_17036, -1, kernel)

plt.imshow(image_17036_sharp), plt.axis("off") # 이미지 출력
plt.show()


# In[61]:


image_17036_very_blurry = cv2.GaussianBlur(image_17036_sharp, (3,3), 0) # 가우시안 블러를 적용
plt.imshow(image_17036_very_blurry), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()


# In[62]:


# 사각형 좌표: 시작점의 x,y  ,넢이, 너비
rectangle = (0, 56, 256, 150)

# 초기 마스크 생성
mask_17036 = np.zeros(image_17036_sharp.shape[:2], np.uint8)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(image_17036_sharp, # 원본 이미지
            mask_17036,       # 마스크
            rectangle,  # 사각형
            bgdModel,   # 배경을 위한 임시 배열
            fgdModel,   # 전경을 위한 임시 배열 
            5,          # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화

# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_17036_2 = np.where((mask_17036==2) | (mask_17036==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱하여 배경을 제거
image_17036_sharp_nobg = image_17036_sharp * mask_17036_2[:, :, np.newaxis]

# plot
plt.imshow(image_17036_sharp_nobg)
plt.show()


# In[63]:


plt.imshow(image_17036_very_blurry), plt.axis("off") # 이미지 출력
plt.show()

median_intensity = np.median(image_17036_very_blurry) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정합니다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용합니다.
image_canny = cv2.Canny(image_17036_very_blurry, lower_threshold, upper_threshold)

plt.imshow(image_canny), plt.axis("off") # 이미지 출력
plt.show()


# ### 흑백으로 로드

# In[157]:


image = cv2.imread(img_files[index], cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
plt.imshow(image, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
type(image) # 데이터 타입을 확인
image # 이미지 데이터를 확인
image.shape # 차원을 확인 (해상도)


# In[158]:


hist1 = cv2.calcHist([image], [0], None, [
    256], [0, 256])
plt.subplot(3, 1, 1)
plt.title('hist1')
plt.plot(hist1)
dst1 = cv2.equalizeHist(image)

hist2 = cv2.calcHist([dst1], [0], None, [
    256], [0, 256])
plt.subplot(3, 1, 2)
plt.title('equal')
plt.plot(hist2)


# In[159]:


cv2.imshow('dst1', dst1)
plt.show()
cv2.waitKey(0)


# ### Yolo-v3

# In[125]:


weights_path = 'C:/Users/shoya/yolov3.weights'
config_path = 'C:/Users/shoya/yolov3.cfg'

cv_net_yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)


# In[126]:


layer_names = cv_net_yolo.getLayerNames()
outlayer_names = [layer_names[i-1] for i in cv_net_yolo.getUnconnectedOutLayers()]


# In[127]:


[layer_names[i-1] for i in cv_net_yolo.getUnconnectedOutLayers()]


# In[128]:


img = cv2.imread('C:/Users/shoya/train/TRAIN_17035.jpg')

# 해당 모델은 416 x 416이미지를 받음으로 img 크기정해줘야함.
cv_net_yolo.setInput(cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))

cv_outs = cv_net_yolo.forward(outlayer_names) # layer이름을 넣어주면 그 layer이름에 해당하는 output을 return 하게 됨


# In[129]:


print(cv_outs[0].shape, cv_outs[1].shape, cv_outs[2].shape) # detection에 필요한 정보를 추출


# In[130]:


# rows , cols는 1/255.0 스케일링된 값을 추후 복원시키는 용도로 사용
rows = img.shape[0]
cols = img.shape[1]

conf_threshold = 0.5 # confidence score threshold

# for loop를 돌면서 추출된 값을 담을 리스트 세팅
class_ids = [] 
confidences = []
boxes = []

for _, out in enumerate(cv_outs):
    for _, detection in enumerate(out): # detection =>  4(bounding box) + 1(objectness_score) + 10(class confidence))
        class_scores = detection[5:] # 인덱스 5음 부터는 10개의 score 값
        class_id = np.argmax(class_scores) # 10개중에 최대값이 있는 index 값
        confidence = class_scores[class_id] # 최대값 score
        
        if confidence > conf_threshold:# 바운딩 박스 중심 좌표 and 박스 크기
            cx = int(detection[0] * cols) # 0~1 사이로 리사이즈 되어있으니 입력영상에 맞는 좌표계산을 위해 곱해줌.
            cy = int(detection[1] * rows)
            bw = int(detection[2] * cols)
            bh = int(detection[3] * rows)
            
            # 바운딩 박스를 그리기 위해선 좌상단 점이 필요함(아래는 그 공식)
            
            left = int(cx - bw / 2) 
            top = int(cy - bh / 2)
            
            class_ids.append(class_id) # class id값 담기
            confidences.append(float(confidence)) # confidence score담기
            boxes.append([left, top, bw, bh]) # 바운딩박스 정보 담기


# In[131]:


labels_to_names_seq = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}


# In[132]:


import matplotlib.pyplot as plt

nms_threshold = 0.4  # nonmaxsuppression threshold 

# opencv 제공하는 nonmaxsuppression 함수를 통해 가장 확률 높은 바운딩 박스 추출
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold) 

draw_img = img.copy() #그림그리기 위한 도화지

# 같은 라벨별로 같은 컬러를 사용하기 위함/ 사용할때마다 랜덤하게 컬러 설정
colors = np.random.uniform(0, 255, size=(len(labels_to_names_seq),3))

if len(idxs) > 0: # 바운딩박스가 아예 없을 경우를 대비하여 1개 이상일때만 실행하도록 세팅
    for i in idxs:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        caption = f"{labels_to_names_seq[class_ids[i]]}: {confidences[i]:.2})" 
        label = colors[class_ids[i]]
        cv2.rectangle(draw_img, (int(left), int(top), int(width), int(height)), color=label, thickness=2)
        cv2.putText(draw_img, caption, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label, 2, cv2.LINE_AA)
        
img_yolo_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB) #
plt.figure(figsize=(12, 12))
plt.imshow(img_yolo_rgb)


# In[59]:


imgfile = 'C:/Users/shoya/train\TRAIN_17036.jpg'
    
    #원본 이미지
img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)


# In[ ]:





# ### Pretrained Resnet50 사용

# In[ ]:


from torchvision import models
import torch

resnet18_pretrained = models.resnet18(pretrained=True)

print(resnet18_pretrained)

# change the output layer to 10 classes
num_classes = 10
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0')
resnet18_pretrained.to(device)

# get the model summary
from torchsummary import summary
summary(resnet18_pretrained, input_size=(3, 224, 224), device=device.type)


# In[ ]:


# visualize the filters of the first CNN layer
for w in resnet18_pretrained.parameters():
    w = w.data.cpu()
    print(w.shape)
    break

# normalize weights
min_w = torch.min(w)
w1 = (-1/(2 * min_w)) * w + 0.5

# make grid to display it
grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = utils.make_grid(x_grid, nrow=8, padding=1)

plt.figure(figsize=(10, 10))
show(x_grid)


# In[ ]:


def contour():
    #이미지
    imgfile = '파일 경로'

    #원본 이미지
    img = cv2.imread(imgfile)

    #흑백 이미지
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #원본 이미지와 흑백이미지가 따로 존재

	#이미지 이진화(TRESH_TOSU)
    ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# In[ ]:




