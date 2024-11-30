import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from sklearn.externals import joblib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import lightgbm as lgb # 引入LightGBM
import pickle #序列化和反序列化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def Sigma_3(data,selected_features):
    import numpy as np
    new_data=data[selected_features] #过滤正常数据
    features = list(new_data.columns)[:-1]
    # 计算每个特征的均值和标准差
    means = new_data[features].mean()
    stds = new_data[features].std()

    # 遍历每个特征
    for feature in features:
        # 计算±3σ的范围
        lower_bound = means[feature] - 3 * stds[feature]
        upper_bound = means[feature] + 3 * stds[feature]
        
        # 将不在范围内的数据填充为均值
        new_data[feature] = np.where((new_data[feature] < lower_bound) | (new_data[feature] > upper_bound), means[feature], new_data[feature])
    return new_data
def select_columns(data):#线性回顾选择特征
    #空值处理
    mean_value = data['出院诊断LENTH_MAX'].mean()
    data['出院诊断LENTH_MAX'].fillna(mean_value, inplace=True)
    #线性回归选择相关性特征
    from sklearn.linear_model import LinearRegression
    all_columns = data.columns.tolist()#所有列
    X = data[all_columns[:-1]]
    y = data["RES"]
    oversampler=SMOTE(random_state=42,sampling_strategy=0.8) #过采样是必须的，不然选出的选特征效果不好
    smote_x,smote_y=oversampler.fit_resample(X,y)
    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(smote_x, smote_y, test_size=0.2, random_state=42)

    # 模型训练
    model = LinearRegression()
    model.fit(X_train, y_train)

    correlation_scores = pd.Series(model.coef_, index=all_columns[:-1])
    # 筛选出相关性评分大于0.2或小于-0.2的特征列名
    selected_features = correlation_scores[(correlation_scores > 0.2) | (correlation_scores < -0.2)].index.tolist()
    return selected_features

def train(selected_features,pca_number,data,model_name='随机森林',select_somte=True,select_pca=False,smote_number=0.8,test_sz=0.2):#模型训练
    
    #空值处理
    mean_value = data['出院诊断LENTH_MAX'].mean()
    data['出院诊断LENTH_MAX'].fillna(mean_value, inplace=True)
    X=Sigma_3(data,selected_features) # 异常值处理（数据，特征因子）
    #X=data[selected_features].copy()
    Y = data["RES"]
    if select_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_number) 
        X = pca.fit_transform(X)
    if select_somte:
        oversampler = SMOTE(random_state=42, sampling_strategy=smote_number) # 过采样
        X, Y = oversampler.fit_resample(X, Y)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_sz, random_state=42)

    # 可视化
    def plot_confusion_matrix(cm, title):
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(title)
        plt.savefig('static/'+title+".png")

    if model_name=="随机森林":
        # 随机森林
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train.values.ravel())
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        print("随机森林准确率: ", rf_acc)
        print("随机森林分类报告:\n", classification_report(y_test, rf_pred)) 
        # 绘制混淆矩阵   
        rf_cm = confusion_matrix(y_test, rf_pred)#随机森林
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plot_confusion_matrix(rf_cm, '随机森林混淆矩阵')
        # 保存模型
        joblib.dump(rf, 'models/random_forest_model.pkl')
        return rf_acc,classification_report(y_test, rf_pred)
    else:
        # 使用LightGBM
        lgbm = lgb.LGBMClassifier()
        lgbm.fit(X_train, y_train.values.ravel())
        lgbm_pred = lgbm.predict(X_test)
        lgbm_acc = accuracy_score(y_test, lgbm_pred)
        print("LightGBM准确率: ", lgbm_acc)
        print("LightGBM分类报告:\n", classification_report(y_test, lgbm_pred))
        # 绘制混淆矩阵
        lgbm_cm = confusion_matrix(y_test, lgbm_pred)#light GBM
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plot_confusion_matrix(lgbm_cm, 'LightGBM混淆矩阵')
        # 保存模型
        lgbm.booster_.save_model('models/lightgbm_model.txt')
        return lgbm_acc,classification_report(y_test, lgbm_pred)
    
def load_and_predict(data, model_name):#模型的应用
    # 将字典转换为DataFrame
    df = pd.DataFrame([data])
    print("df打印结果:")
    print(df)
    # 根据模型名称加载模型
    if model_name == "随机森林":
        model_path = "models/random_forest_model.pkl"
        with open(model_path, 'rb') as file:
            model = joblib.load(file)#序列化和反序列化
    elif model_name == "lightGBM":
        model_path = "models/lightgbm_model.txt"
        model = lgb.Booster(model_file=model_path)
    else:
        return "未知的模型名称"
   
    # 预测
    prediction = model.predict(df.values.reshape(1, -1))
    
    print("预测结果：",prediction)
    #返回0或者1
    return prediction[0]
if __name__=='__main__':
    import argparse
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Machine Learning Classification Task')

    # 添加命令行参数
    parser.add_argument('--file_path', type=str, default='【东软集团A08】医保特征数据16000（修订版）.csv', help='File path for the dataset')
    parser.add_argument('--model_name', type=str, default='随机森林', help='Name of the model (随机森林 or other)')
    parser.add_argument('--select_smote', type=bool, default=True, help='Whether to use SMOTE for oversampling')
    parser.add_argument('--select_pca', type=bool, default=False, help='Whether to use PCA for dimensionality reduction')
    parser.add_argument('--smote_number', type=float, default=0.8, help='Oversampling ratio for SMOTE')
    parser.add_argument('--test_sz', type=float, default=0.2, help='test size ')
    # 解析命令行参数
    args = parser.parse_args()
    '''
    try:
        from app import data
    except ImportError:
        print("Error: 'data' not imported from app.py. Make sure the variable is defined in app.py.")
    except NameError:
        print("Error: 'data' is not defined. Please make sure data is defined in app.py.")
    except Exception as e:
        print("An error occurred while importing data from app.py:", e)
    '''
    #读取文件
    data2 = pd.read_csv(args.file_path, encoding='GB2312')
    #特征筛选
    selected_features=select_columns(data2)
    #3σ异常值检测
    print('特征值:',selected_features)
    train(selected_features=selected_features,#打分特征列表
          pca_number=len(selected_features)-1,
          data=data2,#数据
          model_name=args.model_name, #模型名字
          select_somte=args.select_smote, #是否选择过采样
          select_pca=args.select_pca, #是否选择PCAPCA主成分
          smote_number=args.smote_number, #过采样比例
          test_sz=args.test_sz)#测试集比例

#虚拟环境中运行：python ..\..\rantree.py --file_path ..\..\【东软集团A08】医保特征数据16000（修订版）.csv

