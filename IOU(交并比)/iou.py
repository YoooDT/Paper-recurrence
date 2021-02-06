import numpy as np 
import cv2

def display(src,det):
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
    cv2.imwrite("iou.jpg",src)


"""
计算两个矩形框的iou
"""
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


if __name__ == "__main__":
    
    # img_path = "catdog.jpg"

    det = np.array(
        [
        [65.0,30.0,324.0,472.0,0.99], 
        [334.0,184.0,552.0,470.0,0.98], 
        [150.0,90.0,400.0,350.0,0.97], 
        ]
    )
    # src = cv2.imread(img_path)
    # display(src,det) #显示det

    for i in range(len(det)):
        for j in range(len(det)):
            if i == j: continue  
            print("%d和%d的iou为:%lf"%(i+1,j+1,iou(det[i],det[j])))