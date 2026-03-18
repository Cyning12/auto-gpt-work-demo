# 使用代码让 LLM 帮你自动拟合一组现实数据，并画出趋势图

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
from pathlib import Path

# 1. 现实数据 (第一步：收集)
csv_path = Path(__file__).parent / "data" / "爬行动物体型数据.csv"

lengths = []
weights = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lengths.append(float(row["身长(米)"]))
        weights.append(float(row["体重(公斤)"]))

x = np.array(lengths)
y = np.array(weights)


# 2. 设定数学模型 (第二步：假设这是一个线性规律 y = ax + b)
def linear_model(x, a, b):
    return a * x + b


# 3. 执行拟合 (第三步：找最优参数 a 和 b)
# curve_fit 会自动通过“最小二乘法”找到让误差最小的参数
# 还有其他的，比如梯度下降法、极大似然估计法等，这里只是最简单的一种
params, covariance = curve_fit(linear_model, x, y)

# 4. 输出结果 (第四步：解读 a 和 b)
a, b = params
print(f"拟合结果：a  = {a:.2f}, b = {b:.2f}")
print(f"预测公式：y = {a:.2f}x + ({b:.2f})")

# 5. 画图 (第五步：展示拟合效果)
plt.scatter(x, y, label="观测数据")
plt.plot(x, linear_model(x, a, b), "r-", label="拟合曲线")
plt.legend()
plt.show()
