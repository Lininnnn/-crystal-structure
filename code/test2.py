import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read
import matplotlib.pyplot as plt
import matplotlib  # 添加此行以导入 matplotlib
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

zhfont1=matplotlib.font_manager.FontProperties(fname=r"SourceHanSansSC-Bold.otf")
# 读取数据
with open('supecon.csv', 'r') as file:
    data = pd.read_csv(file)
    pattern = r'([A-Z][a-z]?)(\d*)'
    elements = data['Formula'].str.findall(pattern).apply(lambda x: [match[0] for match in x])
    all_elements = set()
    all_elements.update(elements.explode().unique())
    all_elements = list(all_elements)

    cifs = data['cif']
    Tc_AD = data['Tc_AD']  # 目标变量
    features = []

# 创建标准化器对象
scaler = StandardScaler()

# 创建SOAP描述符对象
soap = SOAP(
    species=all_elements,
    r_cut=5,
    n_max=2,
    l_max=1,
    sigma=0.125,
    compression={"mode": "off", "species_weighting": None},
    sparse=False,
    dtype='float32'
)

# 生成SOAP描述符并标准化
for i in tqdm(range(len(cifs)), desc="读取并用ase解析cif文件,生成SOAP描述符"):
    cif_file = data.loc[i, "cif"]
    with open('temp_file.cif', 'w') as cif_output:
        cif_output.write(cif_file)
    atoms = read('temp_file.cif')
    soap_descriptors = soap.create(atoms)
    soap_descriptors = scaler.fit_transform(np.array(soap_descriptors, dtype=np.float32))
    features.append(soap_descriptors)

labels = np.array(data['Tc_AD'])  # 将标签转换为数组

# 计算每个描述符的最大长度
max_length = max(feature.shape[0] for feature in features)
max_width = max(feature.shape[1] for feature in features)

# 填充特征
padded_features = np.array([
    np.pad(f, ((0, max_length - f.shape[0]), (0, max_width - f.shape[1])), mode='constant') for f in features
])

# 将三维数据展平为二维数据
padded_features = padded_features.reshape(len(features), -1)  # reshape为(n_samples, n_features)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(padded_features, labels, test_size=0.3, random_state=42, shuffle=True)

# 确保在填充特征后将 padded_features 转为 NumPy 数组
padded_features = np.array([
    np.pad(f, ((0, max_length - f.shape[0]),
(0, max_width - f.shape[1])), mode='constant') for f in features
])

# 在分割数据集后，将 X_train 转为 NumPy 数组
X_train, X_test, y_train, y_test = train_test_split(padded_features, labels, test_size=0.3, random_state=42, shuffle=True)
X_train = np.array(X_train)  # 确保 X_train 是 NumPy 数组
y_train = np.ravel(y_train)   # 确保 y_train 是一维数组

# 打印形状以检查
print(X_train.shape)
print(y_train.shape)

# 创建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=0)

# 使用 tqdm 包裹 fit 方法
mse_list = []
with tqdm(total=1, desc="训练进度") as pbar:
     model.fit(X_train, y_train)


# 进行预测
y_pred_train = model.predict(X_train)  # 可以预测训练集
y_pred_test = model.predict(X_test)    # 预测测试集

# 计算均方误差
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# 输出均方误差
print(f"训练集均方误差: {mse_train:.4f}")
print(f"测试集均方误差: {mse_test:.4f}")

# 绘制拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='yellow', label='预测值')  # 添加预测值的散点图
plt.scatter(y_test, y_test, color='blue', label='实际值')  # 添加实际值的散点图
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='理想预测')
plt.xlabel('实际 Tc_AD')
plt.ylabel('预测 Tc_AD')
plt.title('测试集 Tc_AD 预测值与实际值比较')
plt.legend()
plt.grid()
plt.show()


