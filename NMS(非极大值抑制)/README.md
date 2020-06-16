NMS:   
非极大值抑制

作用:   
在目标检测中，对于一个需要预测的对象，会预测出很多个回归框(bounding box)，NMS的作用是来去掉多余的回归框,保留最近姐GT框的回归框

实现思路:   
1.分类器会给每个bounding box一个score，表示这个bbox属于某一个class的概率.首先按照score对bbox排序(0-n个bbox) 选出最高得分为bb0    
2.遍历其余的bbox ，如果bbi(i:1-n)与bb0的IOU大于设置的某个阈值，则删除该bbi框(同一对象选择score大的那个)   
3.在未删除的bbox中继续重复1,2过程，直到找到全部保留的bbox   
4.最后剩余的bbox应为每个类别score最高的   

结果:   
去掉了多余的bbox    
