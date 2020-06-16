import numpy as np


def nums(dets,thresh):
    #thresh IOU阈值

    # det:[x1,y1,x2,y2,scores] 每个框都有4个坐标，一个score
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]

    scores = dets[:,4]

    areas = (x2-x1+1)*(y2-y1+1) # x,y初始值有0 0~2应为3

    order = scores.argsort()[::-1]  #按照scores大小排序 -1表示倒叙，从大到小

    keep = []

    while len(order)!=0:
        bb0= order[0] #选取score最大的
        keep.append(bb0)

        xx1 = np.maximum(x1[bb0], x1[order[1:]])
        yy1 = np.maximum(y1[bb0], y1[order[1:]])
        xx2 = np.minimum(x2[bb0], x2[order[1:]])
        yy2 = np.minimum(y2[bb0], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[bb0] + areas[order[1:]] - inter) #计算bb0与bbi的IOU

        inds = np.where(ovr <= thresh)[0] #保留IOU不大于阈值的bbox
        order = order[inds + 1]  #删除IOU大于阈值的bbox

    return keep