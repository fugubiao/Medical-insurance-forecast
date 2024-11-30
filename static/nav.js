#触发 文件加载事件
        document.getElementById('load').addEventListener('click', function() {
            fetch('/get_table_info')
            .then(response => response.json())
            .then(data => {
            const info = data.table_info.replace(/\n/g, "<br>");  // 将换行符转换为<br>以在HTML中正确显示
            document.getElementById('ajaxtable_info').innerHTML = info;
        
        })
        .catch(error => console.error('Error:', error));
  });