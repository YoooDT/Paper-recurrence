import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

"""
NMS步骤:
0:NMS是对每一类别进行处理，所以是for object in all_obejcts,设当前类别框数组为Fate
1.取当前类别Fate中得分最高的框A,加入到keep数组中
2.取当前类别Fate中除了A的框分别与A进行IOU计算，如果计算值大于设置阈值(一般为0.3~0.5),则删除当前框
3.重复上述操作直至Fate为空，此时keep数组就为nms处理的所有框
"""

def display(src,det,filename):
    for i in range(len(det)):
        test = det[i]
        x1 = int(test[0])
        y1 = int(test[1])
        x2 = int(test[2])
        y2 = int(test[3])
        # print(x1,y1,x2,y2)
        cv2.rectangle(src,(x1,y1),(x2,y2),(45,52,115),3)
        #右上角坐标
        word_x = x2 + 5 
        word_y = y1 + 10
        cv2.putText(src,"score %lf" % test[4],(word_x,word_y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(55,255,155))
    cv2.imwrite(filename,src)

def iou(d1,d2):
    a_x1 , a_y1, a_x2, a_y2 = d1[0], d1[1], d1[2], d1[3]
    b_x1 , b_y1, b_x2, b_y2 = d2[0], d2[1], d2[2], d2[3]
    
    aera_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    aera_b = (b_x2 - b_x1) * (b_y2 - b_y1)

    w = min(a_x2, b_x2) - max(a_x1, b_x1)
    h = min(a_y2, b_y2) - max(a_y1, b_y1)

    if w <= 0 or h <= 0 : #无相交
        return 0 
    
    area = w * h 

    return area / (aera_a + aera_b - area)

def NMS(det):
    keep = []
    times = 3 
    det = det.tolist()
    det.sort(key=lambda x:x[4],reverse=True) #按照得分大小，对框进行排序
    while len(det) != 0:
        keep.append(det[0]) #加入当前值最大的框，并在det删除
        det.pop(0)
        cur_len = len(det)
        i = 0 
        while cur_len:
            cur_len -= 1 
            # print(i,iou(keep[-1],det[i]))
            if iou(keep[-1],det[i]) > 0.3 : #如果新加入的框与其他框的Iou大于阈值，则删除
                det.pop(i)
            else:
                i += 1 #如果Iou小于阈值，则保留这个框，所以进行下一个框查看
    return keep

            
if __name__ == "__main__":

    img_path = "catdog.jpg"
    
    #[x1,y2,x2,y2,score]
    det  = np.array(
        [
        [[65.0,30.0,324.0,472.0,0.99], #狗准确位置
        [150.0,90.0,330.0,470.0,0.97], #狗偏移1
        [20.0,10.0,322.0,466.0,0.98]], #狗偏移2
        
        [[334.0,184.0,552.0,470.0,0.98], #猫准确位置
        [433.0,270.0,546.0,466.0,0.97], #猫偏移1
        [280.0,134.0,550.0,465.0,0.96]] #猫偏移2
        ]
    )
    src = cv2.imread(img_path)
    for i in range(len(det)):
        display(src,det[i],"before_nms.jpg")
    src = cv2.imread(img_path)
    for i in range(len(det)):
        keep = NMS(det[i])
        display(src,keep,"after_nms.jpg")
    