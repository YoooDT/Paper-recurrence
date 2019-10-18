import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoiddifferential(y):
    return y*(1-y)


def demo_backward_p(x,y,epoch):

    #demo

    alpha = 0.5

    x1 = np.float(x[0])
    x2 = np.float(x[1])

    y1 = np.float(y[0])
    y2 = np.float(y[1])

    #初始权重
    b = [0.35,0.6]
    w = [0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]


    for e in range(epoch):
    # forward_p 前向传播
        z1 = w[0]*x1+w[1]*x2+b[0]
        z2 = w[2]*x1+w[3]*x2+b[0]

        a1 = sigmoid(z1)
        a2 = sigmoid(z2)

        z3 = w[4]*a1+w[5]*a2+b[1]
        z4 = w[6]*a1+w[7]*a2+b[1]

        a3 = sigmoid(z3)
        a4 = sigmoid(z4)


    #backward_p 反向传播

        error = 1/2*np.square(y1-a3) + 1/2*np.square(y2-a4)


        print("第 %d 轮的误差为 %f \n" %(e,error))

        diff_err_to_a3 = -(y1-a3)
        diff_err_to_a4 = -(y2-a4)

        diff_a3_to_z3 = sigmoiddifferential(a3)
        diff_a4_to_z4 = sigmoiddifferential(a4)

        diff_a1_to_z1 = sigmoiddifferential(a1)
        diff_a2_to_z2 = sigmoiddifferential(a2)


        diffw1 = (diff_err_to_a3*diff_a3_to_z3*w[4] + diff_err_to_a4*diff_a4_to_z4*w[6])*diff_a1_to_z1*x1
        diffw2 = (diff_err_to_a3*diff_a3_to_z3*w[4] + diff_err_to_a4*diff_a4_to_z4*w[6])*diff_a1_to_z1*x2
        diffw3 = (diff_err_to_a3*diff_a3_to_z3*w[5] + diff_err_to_a3*diff_a3_to_z3*w[7])*diff_a2_to_z2*x1
        diffw4 = (diff_err_to_a3*diff_a3_to_z3*w[5] + diff_err_to_a3*diff_a3_to_z3*w[7])*diff_a2_to_z2*x2

        diffw5 = diff_err_to_a3*diff_a3_to_z3*a1
        diffw6 = diff_err_to_a3*diff_a3_to_z3*a2
        diffw7 = diff_err_to_a4*diff_a4_to_z4*a1
        diffw8 = diff_err_to_a4*diff_a4_to_z4*a2


        w[0] = w[0] - alpha*diffw1
        w[1] = w[1] - alpha*diffw2
        w[2] = w[2] - alpha*diffw3
        w[3] = w[3] - alpha*diffw4
        w[4] = w[4] - alpha*diffw5
        w[5] = w[5] - alpha*diffw6
        w[6] = w[6] - alpha*diffw7
        w[7] = w[7] - alpha*diffw8



if __name__ == '__main__':

    x = [0.05,0.10]
    y = [0.01,0.99]
    epoch = 10
    demo_backward_p(x,y,epoch)