document.addEventListener('DOMContentLoaded', function() {
    // 示例：更改背景颜色的函数
    function changeBackgroundColor(color) {
        document.getElementById('ajaxContent').style.backgroundColor = color;
    }

    // 绑定导航栏的点击事件，你可以在这里添加Ajax请求
    document.getElementById('fileProcessing').addEventListener('click', function() {
        // 示例：改变背景颜色
        changeBackgroundColor('#ffdddd');
    });

    document.getElementById('featureSelection').addEventListener('click', function() {
        changeBackgroundColor('#ddffdd');
    });

    document.getElementById('modelTraining').addEventListener('click', function() {
        changeBackgroundColor('#ddddff');
    });
});