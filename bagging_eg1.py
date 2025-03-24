import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 避免坐标轴不能正常的显示负号
def LR(d):
    x=np.array([point[0] for point in d]).reshape(-1, 1)  # 参数(样本数量,1)，前者可填-1
    y=np.array([point[1] for point in d])
    model = LinearRegression()  # 建立线性回归模型
    model.fit(x, y)             # 拟合数据
    w=model.coef_[0]            # 斜率
    b=model.intercept_          # 截距
    arg.append([w,b])

arg=[]
d=[[[4.2,8600],[7.1,6100],[6.3,6700],[1.1,12000],[0.2,14200],[4.0,8500],[3.5,8900],[8,6200],[2.3,11200]]]
x,y= zip(*d[0])
plt.scatter(x,y,color='b',marker='.',label='原始数据点')

d.append([d[0][j] for j in [0,0,0,2,3,4,6,6,8]])
d.append([d[0][j] for j in [0,0,1,1,3,5,5,6,8]])
d.append([d[0][j] for j in [0,3,4,5,5,5,5,6,7]])

for i in range(0,4):
    LR(d[i])
    print(f"线性回归L{i}(X)={arg[-1][0]:.4f}x+{arg[-1][1]:.4f}")

w=sum([row[0] for row in arg[1:4]])/3
b=sum([row[1] for row in arg[1:4]])/3

print(f"简单平均法集成L(X)={w:.4f}x+{b:.4f}")
ypre=w*5.5+b
print(f"求得L(5.5)={ypre:.0f}（元/m²）")

x1=np.array([0,10])
y1=w*x1+b
plt.scatter(5.5,ypre,color='g',marker='*',label='预测数据点')
plt.plot(x1,y1,color='r',label=f'L(X)')
plt.title("某市房屋价格与房屋位置 线性回归集成模型")
plt.xlabel("X(km)")
plt.ylabel("y(元/m²)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()