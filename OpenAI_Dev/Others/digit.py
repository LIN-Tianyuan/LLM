"""
加载手写数字数据 → 划分训练/测试 → 训练随机森林模型 → 用模型对测试数据进行预测 → 输出预测结果
"""
from sklearn.model_selection import train_test_split  # 用于将数据集随机划分为训练集和测试集
from sklearn.ensemble import RandomForestClassifier   # 一个集成学习的分类器，它用多棵决策树来做预测
from sklearn.datasets import load_digits              # 载入一个包含 0~9 手写数字图片的小数据集（每张图片是 8x8 像素）

# 加载数据集
digits = load_digits()
print(digits.data.shape)    # (1797, 64)
X = digits.data             # 图像数据，大小为 (1797, 64)，表示1797张图，每张图是8x8=64个像素点的灰度值（已经扁平化为1维）
y = digits.target           # 图像对应的数字标签（也就是图片上写的是几）
print('X: ', X)
print('y: ', y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 把 20% 的数据当作测试集（test_size=0.2），其余 80% 是训练集
# random_state=42 是为了保证每次划分一致，方便复现。
# 训练模型
model = RandomForestClassifier()    # 创建一个随机森林模型（这里没有设置参数，使用默认配置）
model.fit(X_train, y_train)         # 用训练数据 X_train 和标签 y_train 来训练模型
# 预测
predictions = model.predict(X_test) # 用训练好的模型，对测试集 X_test 进行预测
print(predictions)                  # 输出的 predictions 是模型预测的数字（长度和 y_test 一样）

# 准确率
from sklearn.metrics import accuracy_score
print("准确率:", accuracy_score(y_test, predictions))
"""
准确率: 0.9777777777777777
"""