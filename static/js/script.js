/*HTML文档加载完成并且DOM树构建完成后触发 */
document.addEventListener('DOMContentLoaded', function() {
    
/*滚轮滚动效果 */
        let lastScrollTop = 0; // 用于存储上一次滚动位置

        window.addEventListener('wheel', function(event) {
        let currentScroll = window.scrollY;
        if (event.deltaY > 0) {
            // 向下滚动
            document.querySelector('.header').style.height = '50px';
        } else if (event.deltaY < 0) {
            // 向上滚动
            document.querySelector('.header').style.height = '500px';
            //console.log('向上滚动时显示的警告信息'); 调试
        }
        lastScrollTop = currentScroll;
    });
/*模拟手机滚轮效果 */
let lastTouchY = 0; // 用于存储上一次触摸位置

window.addEventListener('touchstart', function(event) {
    lastTouchY = event.touches[0].clientY;
});

window.addEventListener('touchmove', function(event) {
    let currentTouchY = event.touches[0].clientY;
    const header = document.querySelector('.header');

    if (currentTouchY > lastTouchY) {
        // 向下滑动
        header.style.height = '500px';
    } else if (currentTouchY < lastTouchY) {
        // 向上滑动
        header.style.height = '50px';
    }
    lastTouchY = currentTouchY;
    });
    /*为复选框添加事件监听器 ——SMOTE 被选中 显示和隐藏的效果。*/
        /*在DOM加载时渲染复选框,避免因为复选框不存在而出错*/
    /*document.querySelector('input[name="dataProcessing"][value="SMOTE"]').addEventListener('change', function() {
        document.getElementById('smote-range-label').style.display = this.checked ? 'block' : 'none';
    });*/

});
/*--------------------------------------------------导航栏-切换------------------------ */
const navItems = document.querySelectorAll('.nav-item');// 获取所有具有类名为 'nav-item' 的元素

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const contentToShow = item.getAttribute('data-content') + '-content';
        const contents = document.querySelectorAll('#content > div');
        contents.forEach(content => {
            if (content.id === contentToShow) {
                content.classList.remove('hidden');
            } else {
                content.classList.add('hidden');
            }
        });
    });
});
/*---------------------------------选择文件-----------------------------------------*/
//处理用户将文件拖拽到指定区域的操作
function dropHandler(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    handleFile(file);
}
//处理拖拽文件到指定区域时的 dragover 事件
function dragOverHandler(event) {
    event.preventDefault();
}
//处理用户点击指定区域的操作
function clickHandler() {
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.style.display = 'none';
fileInput.addEventListener('change', function() {
    const file = fileInput.files[0];
    handleFile(file);
});
document.body.appendChild(fileInput);
fileInput.click();
}

function generatePreviewTable(jsonData) {
    // 检查是否提供了 JSON 数据
    if (!jsonData || !jsonData.length) {
        console.error('没有提供 JSON 数据。');
        return;
    }

    // 获取目标 div 元素
    const tableContainer = document.getElementById('table-content');

    // 创建表格元素
    const table = document.createElement('table');
    table.setAttribute('border', '1'); // 设置表格边框样式

    // 创建表头
    const headerRow = document.createElement('tr');
    const headers = Object.keys(jsonData[0]); // 获取第一行对象的键作为表头
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // 创建表格主体
    const tbody = document.createElement('tbody');
    jsonData.forEach(row => {
        const dataRow = document.createElement('tr');
        Object.values(row).forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            dataRow.appendChild(td);
        });
        tbody.appendChild(dataRow);
    });
    table.appendChild(tbody);

    // 将表格添加到目标 div 元素中
    tableContainer.innerHTML = ''; // 清空 div 内容
    tableContainer.appendChild(table);
}

function handleFile(file) { //路由：选择文件
const formData = new FormData();
formData.append('file', file);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(data); // 在控制台中查看后端返回的数据

    // 获取要更新的 div 元素
    const fileContentDiv = document.getElementById('region-content');

    // 根据后端返回的数据更新 div 内容和样式
    if (data.message === '文件上传成功') {
        fileContentDiv.innerText = '文件上传成功';
        fileContentDiv.classList.add('success'); // 添加成功样式类
        console.log("文件上传成功");
        /*----------------------------------文件上传成功后操作——显示预览表格------------------------------------ */

        /*
        <json乱码 数据转中文>
        JSON.parse()方法将其转换为JavaScript对象
        JavaScript对象转换回包含Unicode转义序列的JSON字符串 JSON.parse
         */
        const data_row_5=JSON.parse(data.table_5_row); //后端返回前五行示例数据 JSON.stringify
       
        //console.log("json数据："+JSON.stringify(data_row_5))
        // 生成预览表格
        generatePreviewTable(data_row_5);
    } else {
        fileContentDiv.innerText = data.message; // 显示其他返回结果
        console.log(data.message);
    }
})
.catch(error => {
    console.error('Error:', error);
});
}
/*-------------------------------模型训练--页面效果----------------------------*/
function showSmoteSlider() { //隐藏显示-SMOTE比例滑块
        var smoteCheckbox = document.getElementById("smote");
        var smoteSlider = document.getElementById("smote_slider");
        if (smoteCheckbox.checked) {
            smoteSlider.style.display = "block";
        } else {
            smoteSlider.style.display = "none";
        }
    }
    //更新滑块值的显示函数
    document.getElementById("smote_number").oninput = function() {
        document.getElementById("smote_value").innerHTML = this.value;
    };

    document.getElementById("test_sz").oninput = function() {
        document.getElementById("test_sz_value").innerHTML = this.value;
    };

    // 混淆矩阵（图片）路径根据选择的模型名字变更
    function updateImage() {
        var modelSelect = document.getElementById("model_name");
        var selectedModel = modelSelect.options[modelSelect.selectedIndex].value;
        var img = document.getElementById("confusion_matrix");
        img.src = "../static/" + selectedModel + "混淆矩阵.png";
    }
    /*-----------------------------------接收输出--------------------------------*/
    //接收路由的准确率和分类报告
    document.querySelector('#form1').addEventListener('submit', function(event) {
        event.preventDefault(); // 阻止表单的默认提交行为刷新全部页面

        var selectedModel = document.getElementById('model_name').value;
        var smoteCheckbox = document.getElementById("smote");
        var smoteSlider = document.getElementById("smote_value").textContent;
        console.log("过采样比例"+smoteSlider)
        if (smoteCheckbox.checked) {
            var smote_check=true;
            var somte_size=smoteSlider;
        } else {
            var smote_check=false
        }
        var pcaCheckbox = document.getElementById("pca");
        if(pcaCheckbox.checked){
            var pca_check=true;
        }
        else{
            var pca_check=false;
        }
        var test_size=document.getElementById("test_sz_value").value;
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                post_model_name: selectedModel,//模型名称
                post_smote_check:smote_check,//是否过采样
                post_somte_size:somte_size,//过采样比例
                post_pca_check:pca_check,//是否PCA
                post_test_size:test_size//测试集比例
            })
        })//成功后解析出准确率和模型参数
        .then(response => response.json())
        .then(data => {
            3
            var acc = data.acc;
            var report = data.report;
            updateImage() //更新图片
            updateOutput(acc, report);//输出
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    //document.getElementById("output")来获取<textarea>元素，并将输出结果设置为<textarea>的值。

        function updateOutput(acc,report) {
        document.getElementById('output').value = acc + '\n' + report;
        }


/*-----------------------------------模型测试--------------------------------------------*/
function loadModel() { //表单AJAX向Flask后端发送数据
    var xhr = new XMLHttpRequest();
    var url = "/load_model"; // Flask路由地址
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            document.getElementById("resultBox").innerHTML = "预测结果：" + response.result;
        }
        else{
            var response = JSON.parse(xhr.responseText);
            document.getElementById("resultBox").innerHTML ="出错了："+response.result
        }
        // 
    };
    var data = new FormData(document.getElementById("modelInputForm"));
    var object = {};
    data.forEach(function(value, key){
        object[key] = value;
    });
    var jsonData = JSON.stringify(object);
    xhr.send(jsonData);
        }