import pandas as pd
import joblib
from sklearn.metrics import classification_report

# 加载模型
random_forest_model = joblib.load('models/random_forest_model.pkl')

# 加载测试数据
data2 = pd.read_csv('【东软集团A08】医保特征数据16000（修订版）.csv', encoding='GB2312')
# 提取后3000行作为测试数据
features = data2.iloc[-3000:, :][['ALL_SUM', '药品费发生金额_SUM', '检查费发生金额_SUM', '治疗费发生金额_SUM', '手术费发生金额_SUM', 
                                  '床位费发生金额_SUM', '医用材料发生金额_SUM', '其它发生金额_SUM', '药品在总金额中的占比', 
                                  '个人支付的药品占比', '检查总费用在总金额占比', '个人支付检查费用占比', '治疗费用在总金额占比', 
                                  '个人支付治疗费用占比', 'BZ_民政救助']]

labels = data2.iloc[-3000:, :]['RES']
# 使用模型进行预测
predictions = random_forest_model.predict(features)

# 生成分类报告
report = classification_report(labels, predictions)
print(report)
