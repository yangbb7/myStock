import React from 'react';

// 最简单的测试组件，不依赖任何外部库
const MinimalTest: React.FC = () => {
  const handleClick = () => {
    console.log('🔥 MINIMAL TEST BUTTON CLICKED!');
    alert('最简单的按钮点击成功！');
    
    // 直接发送网络请求
    fetch('http://localhost:8000/health')
      .then(response => {
        console.log('🔥 Health check response:', response.status);
        return response.json();
      })
      .then(data => {
        console.log('🔥 Health check data:', data);
        alert('健康检查成功: ' + JSON.stringify(data));
      })
      .catch(error => {
        console.error('🔥 Health check error:', error);
        alert('健康检查失败: ' + error);
      });
  };

  const handleSystemStart = () => {
    console.log('🔥 SYSTEM START BUTTON CLICKED!');
    
    fetch('http://localhost:8000/system/start', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({})
    })
      .then(response => {
        console.log('🔥 System start response:', response.status);
        return response.json();
      })
      .then(data => {
        console.log('🔥 System start data:', data);
        alert('系统启动成功: ' + JSON.stringify(data));
      })
      .catch(error => {
        console.error('🔥 System start error:', error);
        alert('系统启动失败: ' + error);
      });
  };

  return (
    <div style={{ padding: '50px', textAlign: 'center' }}>
      <h1>🧪 最简单的测试页面</h1>
      
      <div style={{ margin: '20px 0' }}>
        <button 
          onClick={handleClick}
          style={{
            padding: '15px 30px',
            fontSize: '18px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            margin: '10px'
          }}
        >
          🧪 测试健康检查
        </button>
        
        <button 
          onClick={handleSystemStart}
          style={{
            padding: '15px 30px',
            fontSize: '18px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            margin: '10px'
          }}
        >
          🚀 测试系统启动
        </button>
      </div>
      
      <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f5f5f5' }}>
        <h3>调试信息</h3>
        <p>当前URL: {window.location.href}</p>
        <p>用户代理: {navigator.userAgent.substring(0, 50)}...</p>
        <p>在线状态: {navigator.onLine ? '在线' : '离线'}</p>
      </div>
    </div>
  );
};

export default MinimalTest;