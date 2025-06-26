import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import make_interp_spline

# 读取数据
origin_data = np.loadtxt("data/zara02/test/crowds_zara02.txt")
predict_data = np.loadtxt("predictions.txt")


H = np.array(
    [[0.02104651,   0,              0           ],
     [0,            -0.02386598,    13.74680446 ],
     [0,            0,              1           ]]
)

H = np.linalg.inv(H)

# 选一段轨迹，投影到像素平面
start_world = origin_data[origin_data[:,1]==3, 2:4][0:9]   # 取行人 id==3 的所有 (x,y)
gt_world = origin_data[origin_data[:,1]==3, 2:4][9:]   # 取行人 id==3 的所有 (x,y)
predict_world = predict_data[predict_data[:,1]==3, 2:4][12:]   # 取行人 id==3 的所有 (x,y)

# 转齐次并映射
def mapping(traj_world):
    ones      = np.ones((len(traj_world),1))
    homo_w    = np.hstack([traj_world, ones])
    homo_img  = (H @ homo_w.T).T
    traj_img  = homo_img[:,:2] / homo_img[:,2:3]
    return traj_img

start_img = mapping(start_world)
gt_img = mapping(gt_world)
predict_img = mapping(predict_world)

# 生成平滑的贝塞尔样条
def spline(traj_img):
    t = np.arange(len(traj_img))  # 时间序列
    spl_x = make_interp_spline(t, traj_img[:,0])  # 对x坐标插值
    spl_y = make_interp_spline(t, traj_img[:,1])  # 对y坐标插值
    t_new = np.linspace(0, len(traj_img)-1, 100)  # 生成100个平滑点
    x_smooth = spl_x(t_new)  # 平滑后的x坐标
    y_smooth = spl_y(t_new)  # 平滑后的y坐标
    return x_smooth, y_smooth

# 画到背景图上
img = np.array(Image.open("crowds_zara02/frame_0085.jpg"))
plt.figure(figsize=(8,6))
plt.imshow(img)
# plt.plot(traj_img[:,0], traj_img[:,1], '-o', color='red', label='Predicted')  # 原始轨迹
# plt.plot(x_smooth, y_smooth, '-', color='blue', label='Smoothed')  # 平滑轨迹
plt.plot(start_img[:,0], start_img[:,1], '-o', color='red', label='Start')
plt.plot(gt_img[:,0], gt_img[:,1], '-o', color='green', label='Ground Truth')
plt.plot(predict_img[:,0], predict_img[:,1], '-o', color='blue', label='Predicted')
plt.legend()
plt.axis('off')
plt.show()