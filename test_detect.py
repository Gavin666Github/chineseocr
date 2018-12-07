import opencv_dnn_detect
#import darknet_detect
from PIL import Image
import numpy as np
import cv2
##

def to_box(r):
    boxes = []
    scores = []
    for rc in r:
        if rc[0]==b'text':
            cx,cy,w,h = rc[-1]
            scores.append(rc[1])
            prob  = rc[1]
            xmin,ymin,xmax,ymax = cx-w/2,cy-h/2,cx+w/2,cy+h/2
            boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
    return boxes,scores

def result_to_box(result):
    boxes = []
    for res in result:

        w,cx,h,cy = res['w'],res['cx'],res['h'],res['cy']
        xmin,ymin,xmax,ymax = cx-w/2,cy-h/2,cx+w/2,cy+h/2
        boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
    return boxes


def result_to_bbox(result,angle):
    bbox = []
    for res in result:
        w, cx, h, cy,degree = res['w'], res['cx'], res['h'], res['cy'],res['degree']
        box = model.xy_rotate_box(cx, cy, w, h, degree)
        #box = model.box_rotate(box,angle,h,w)
        bbox.append(box)
    return bbox

'''

img = cv2.imread('/home/gavin/Desktop/demo-card-1.jpeg')

boxes, scores = opencv_dnn_detect.text_detect(np.array(img))

for bbox in boxes:
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=1)

print(len(boxes))
cv2.imshow('tested_1',img)
cv2.waitKey(0)
cv2.imwrite('/home/gavin/Desktop/tested_demo_card.jpg',img)


'''

import model
from   config import DETECTANGLE

path = '/home/gavin/Desktop/demo-card-1.jpeg'

img = Image.open(path).convert("RGB")
img_res, result, angle,new_boxes = model.model(img,
                                     detectAngle=DETECTANGLE,  ##是否进行文字方向检测
                                     config=dict(MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并80
                                                 MIN_V_OVERLAPS=0.6,
                                                 MIN_SIZE_SIM=0.6,
                                                 TEXT_PROPOSALS_MIN_SCORE=0.2,
                                                 TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                 TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                                 MIN_RATIO=1.0,
                                                 LINE_MIN_SCORE=0.2,
                                                 TEXT_PROPOSALS_WIDTH=0,
                                                 MIN_NUM_PROPOSALS=0,
                                                 ),
                                     leftAdjust=True,  ##对检测的文本行进行向左延伸
                                     rightAdjust=True,  ##对检测的文本行进行向右延伸
                                     alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                                     ifadjustDegree=True
                                     )
print(result)
print("*"*50)
print(angle)
print("="*50)
res = map(lambda x:{'w':x['w'],'h':x['h'],'cx':x['cx'],'cy':x['cy'],'degree':x['degree'],'text':x['text']}, result)
res = list(res)


img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)


#不带旋转角度，框不住正确的文字
boxes = result_to_box(res)
print(len(boxes))

for bbox in boxes:
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=1)
'''
# x1,y1,x2,y2,x3,y3,x4,y4
bboxes = result_to_bbox(res,angle)

for bbox in bboxes:
    print("."*40)
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[4]), int(bbox[5])), color=(0, 255, 0), thickness=1)
    tmpImg = img.rotate(degree_, center=(x_center, y_center))
    #cv2.rectangle(img, (x1, y1), (x3,y3), color=(0, 255, 0), thickness=1)
'''
cv2.imshow('tested_1',img)
cv2.waitKey(0)
cv2.imwrite('/home/gavin/Desktop/tested_demo_card.jpg',img)

