from flask import Flask, jsonify, request,render_template 
#render_template 渲染模板并显示，模板要求放在template目录
import pandas as pd
from io import StringIO

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('show.html')

@app.route('/get_file_handle')
def get_file_handle(): #文件处理-导航栏 加载
    html_1='''
        <div class="form-group">
            <label for="fileInput">选择文件</label>
            <input type="file" id="fileInput" class="form-control">
            <button id="load">加载</button>
        </div>
    
        <div class="table_info" id="ajaxtable_info">
            <!-- AJAX请求的内容将显示在这里 -->
        </div>

        '''
    return html_1

@app.route('/get_table_info', methods=['GET'])
def get_table_info():
    data = pd.read_csv("uploads\A0816000.csv", encoding='GB2312')
    n, m = data.shape
    table_html = "<p>表格行数：{}</p><p>表格列数：{}</p>".format(n, m)
    table_html += "<table>"
    for index, row in data.iterrows():
        table_html += "<tr>"
        for column in row:
            table_html += "<td>{}</td>".format(column)
        table_html += "</tr>"
    table_html += "</table>"
    #return table_html
    t_html='''
            <table>
  <tr>
    <td>&nbsp;</td>
    <td>Knocky</td>
    <td>Flor</td>
    <td>Ella</td>
    <td>Juan</td>
  </tr>
  <tr>
    <td>Breed</td>
    <td>Jack Russell</td>
    <td>Poodle</td>
    <td>Streetdog</td>
    <td>Cocker Spaniel</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>16</td>
    <td>9</td>
    <td>10</td>
    <td>5</td>
  </tr>
  <tr>
    <td>Owner</td>
    <td>Mother-in-law</td>
    <td>Me</td>
    <td>Me</td>
    <td>Sister-in-law</td>
  </tr>
  <tr>
    <td>Eating Habits</td>
    <td>Eats everyone's leftovers</td>
    <td>Nibbles at food</td>
    <td>Hearty eater</td>
    <td>Will eat till he explodes</td>
  </tr>
</table>
        '''
    return t_html
@app.route('/model_html', methods=['GET'])#模型训练和测试-导航栏 加载
def model_html():
    html_2='''
            <h2>选择数据处理方式</h2>
    <form>
        <div class="option-section">
            <label><input type="checkbox" name="dataProcessing" value="PCA"> PCA</label>
            <label><input type="checkbox" name="dataProcessing" value="SMOTE"> SMOTE过采样</label>
            <label id="smote-range-label" style="display:none;">过采样比例: <input type="range" min="0.1" max="1" step="0.1" value="0.1" class="slider" id="smote-range"></label>
        </div>
        <div class="option-section">
            <label for="test-ratio">测试集比例:</label>
            <input type="range" id="test-ratio" name="testRatio" min="0.1" max="1" step="0.1" value="0.3" class="slider">
        </div>
        <div class="option-section">
            <label>选择模型:</label>
            <label><input type="radio" name="model" value="RandomForest"> 随机森林</label>
            <label><input type="radio" name="model" value="LightGBM"> LightGBM</label>
        </div>
        <div class="option-section">
            <label><input type="checkbox" name="crossValidation"> 是否使用交叉验证</label>
        </div>
        <button type="submit">提交</button>
    </form>
    
    <script>
        // 确保DOM加载完成后执行
        document.addEventListener('DOMContentLoaded', function() {
            // 为复选框添加事件监听器，处理SMOTE复选框显示和隐藏效果
            document.querySelector('input[name="dataProcessing"][value="SMOTE"]').addEventListener('change', function() {
                document.getElementById('smote-range-label').style.display = this.checked ? 'block' : 'none';
            });

            // 初始化状态，手动触发一次事件处理函数
            document.querySelector('input[name="dataProcessing"][value="SMOTE"]').dispatchEvent(new Event('change'));
        });
    </script>
    '''

    return html_2
if __name__ == '__main__':
    app.run(debug=True)
    