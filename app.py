from flask import Flask, jsonify, request,render_template 
import subprocess
#render_template 渲染模板并显示，模板要求放在template目录
from rantree import train,select_columns,load_and_predict
import pandas as pd

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('show.html')

ALLOWED_EXTENSIONS = {'csv'}  # 允许上传的文件类型

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    global data

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'})

    if file and allowed_file(file.filename):
        if file.filename.lower().endswith('.csv'):
            data = pd.read_csv(file, encoding='gbk')
        elif file.filename.lower().endswith('.xls') or file.filename.lower().endswith('.xlsx'):
            data = pd.read_excel(file, encoding='gbk')
        print(data)
        table_5_row = data.head().to_json(orient='records')
        # 返回文件上传成功的信息以及预览数据
        return jsonify({
            'message': '文件上传成功',
            'table_5_row':table_5_row
            })
    else:
        return jsonify({'message': '上传的文件不是 CSV 文件'})
    

@app.route('/submit', methods=['POST'])#模型训练与输出
def submit_form():
    print('返回的参数:',request)
    data3 = request.json
    model_name = data3.get('post_model_name')
    select_smote = data3.get('post_smote_check')
    select_pca = data3.get('post_pca_check')
    smote_number = float(data3.get('post_somte_size'))
    test_sz = float(data3.get('post_test_size', 0.2))
    try:
        data2=data
    except NameError:
        return jsonify({'acc': '错误', 'report': '未选择训练文件'})
    selected_features=select_columns(data2)
    acc,report=train(selected_features=selected_features,#打分特征列表
          pca_number=len(selected_features)-1,
          data=data2,#数据
          model_name=model_name, #模型名字
          select_somte=select_smote, #是否选择过采样
          select_pca=select_pca, #是否选择PCA主成分
          smote_number=smote_number, #过采样比例
          test_sz=test_sz)#测试集比例
    return jsonify({'acc': acc, 'report': report})

@app.route('/load_model', methods=['POST'])#模型应用和输出
def load_model():
    # 获取前端发送的数据
    data = request.get_json()#返回模型名称+训练参数的字典
    print(data)  # 仅用于调试，显示接收到的数据
    # 假设前端发送的数据中包含模型名称
    model_name = data.pop('model_name2', None)  # 从数据中提取模型名称，并从字典中移除
    if not model_name:
        print("not分支:",model_name)
        return jsonify({'error': '模型名称未提供'}), 400

    # 调用load_and_predict函数进行预测
    result = load_and_predict(data, model_name)
    print("result:",result)
    if result>=0.6:
        result="预测为骗保的概率很大，请谨慎报销"
    else:
        result="预测为正常，但仍需仔细核对信息"
    # 将结果以JSON格式返回给前端
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)