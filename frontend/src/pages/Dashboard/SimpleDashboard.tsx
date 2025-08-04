import React from 'react';
import { Card, Button, Space, message } from 'antd';

// 最简单的直接API测试组件
const SimpleApiTest: React.FC = () => {
  const testApi = async () => {
    console.log('🔥 [DEBUG] Button clicked - testApi function called');
    
    try {
      console.log('🔥 [DEBUG] About to make fetch request...');
      console.log('🔥 [DEBUG] URL: http://localhost:8000/system/start');
      console.log('🔥 [DEBUG] Method: POST');
      
      const response = await fetch('http://localhost:8000/system/start', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({}),
      });
      
      console.log('🔥 [DEBUG] Fetch completed');
      console.log('🔥 [DEBUG] Response status:', response.status);
      console.log('🔥 [DEBUG] Response headers:', response.headers);
      
      const data = await response.json();
      console.log('🔥 [DEBUG] Response data:', data);
      
      if (response.ok) {
        console.log('🔥 [DEBUG] Success - showing success message');
        message.success('API调用成功！');
      } else {
        console.log('🔥 [DEBUG] Error - showing error message');
        message.error(`API调用失败: ${response.status}`);
      }
    } catch (error) {
      console.error('🔥 [DEBUG] Exception caught:', error);
      console.error('🔥 [DEBUG] Error type:', typeof error);
      console.error('🔥 [DEBUG] Error message:', error?.message);
      console.error('🔥 [DEBUG] Error stack:', error?.stack);
      message.error(`网络错误: ${error}`);
    }
  };

  return (
    <Card title="简单API测试" style={{ margin: '20px' }}>
      <Space direction="vertical" size="middle">
        <Space>
          <Button 
            type="default"
            onClick={() => {
              console.log('🔥 [DEBUG] Basic test button clicked!');
              alert('按钮点击事件正常工作！');
              message.info('基础点击测试成功');
            }}
          >
            🧪 基础点击测试
          </Button>
          
          <Button 
            type="primary" 
            onClick={() => {
              console.log('🔥 [DEBUG] Console test button clicked');
              console.log('🔥 [DEBUG] Current time:', new Date().toISOString());
              console.log('🔥 [DEBUG] Window location:', window.location.href);
              console.log('🔥 [DEBUG] User agent:', navigator.userAgent);
              message.success('控制台测试完成，请查看控制台输出');
            }}
          >
            📝 控制台测试
          </Button>
        </Space>
        
        <Space>
          <Button 
            type="primary" 
            onClick={(e) => {
              console.log('🔥 [DEBUG] START BUTTON CLICKED!');
              console.log('🔥 [DEBUG] Event object:', e);
              
              try {
                // 最简单的同步测试
                console.log('🔥 [DEBUG] About to call testApi...');
                testApi().catch(error => {
                  console.error('🔥 [DEBUG] testApi failed:', error);
                });
                console.log('🔥 [DEBUG] testApi called successfully');
              } catch (error) {
                console.error('🔥 [DEBUG] Exception in onClick:', error);
              }
            }}
          >
            🚀 测试 /system/start
          </Button>
          
          <Button 
            type="primary"
            style={{ backgroundColor: '#ff4d4f', borderColor: '#ff4d4f' }}
            onClick={() => {
              console.log('🔥 [DEBUG] DIRECT START TEST CLICKED!');
              
              // 最直接的fetch调用
              fetch('http://localhost:8000/system/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}'
              })
              .then(response => {
                console.log('🔥 [DEBUG] Direct fetch response:', response.status);
                return response.json();
              })
              .then(data => {
                console.log('🔥 [DEBUG] Direct fetch data:', data);
                alert('直接调用成功！');
              })
              .catch(error => {
                console.error('🔥 [DEBUG] Direct fetch error:', error);
                alert('直接调用失败：' + error);
              });
            }}
          >
            🔴 直接启动测试
          </Button>
          <Button 
            onClick={() => {
              console.log('🔥 [DEBUG] Health check button clicked');
              console.log('🔥 [DEBUG] About to fetch health endpoint...');
              
              fetch('http://localhost:8000/health')
                .then(r => {
                  console.log('🔥 [DEBUG] Health check response received');
                  console.log('🔥 [DEBUG] Status:', r.status);
                  return r.json();
                })
                .then(data => {
                  console.log('🔥 [DEBUG] Health check data:', data);
                  message.success('健康检查成功');
                })
                .catch(err => {
                  console.error('🔥 [DEBUG] Health check error:', err);
                  message.error('健康检查失败');
                });
            }}
          >
            ❤️ 测试健康检查
          </Button>
        </Space>
      </Space>
      <div style={{ marginTop: '16px', padding: '16px', background: '#f5f5f5' }}>
        <h4>🔍 调试信息</h4>
        <p><strong>当前页面:</strong> {window.location.href}</p>
        <p><strong>用户代理:</strong> {navigator.userAgent.substring(0, 100)}...</p>
        <p><strong>在线状态:</strong> {navigator.onLine ? '✅ 在线' : '❌ 离线'}</p>
        
        <h4>📋 测试步骤</h4>
        <ol>
          <li>点击 "🧪 基础点击测试" - 验证按钮事件是否工作</li>
          <li>点击 "📝 控制台测试" - 验证控制台日志是否正常</li>
          <li>点击 "❤️ 测试健康检查" - 验证网络请求是否发送</li>
          <li>点击 "🚀 测试 /system/start" - 测试系统启动API</li>
        </ol>
        
        <h4>🔧 调试提示</h4>
        <p>• 按 F12 打开开发者工具</p>
        <p>• 查看 Console 标签的日志输出</p>
        <p>• 查看 Network 标签的网络请求</p>
        <p>• 所有调试日志都以 🔥 [DEBUG] 开头</p>
      </div>
    </Card>
  );
};

// 网络状态检测组件
const NetworkStatus: React.FC = () => {
  const [isOnline, setIsOnline] = React.useState(navigator.onLine);
  
  React.useEffect(() => {
    const handleOnline = () => {
      console.log('🔥 [DEBUG] Network status: ONLINE');
      setIsOnline(true);
    };
    
    const handleOffline = () => {
      console.log('🔥 [DEBUG] Network status: OFFLINE');
      setIsOnline(false);
    };
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);
  
  return (
    <Card size="small" style={{ marginBottom: '16px' }}>
      <Space>
        <span>网络状态:</span>
        <span style={{ color: isOnline ? '#52c41a' : '#ff4d4f' }}>
          {isOnline ? '✅ 在线' : '❌ 离线'}
        </span>
        <Button 
          size="small"
          onClick={() => {
            console.log('🔥 [DEBUG] Testing basic connectivity...');
            fetch('https://httpbin.org/get')
              .then(r => {
                console.log('🔥 [DEBUG] External connectivity test successful');
                message.success('外部网络连接正常');
              })
              .catch(e => {
                console.error('🔥 [DEBUG] External connectivity test failed:', e);
                message.error('外部网络连接失败');
              });
          }}
        >
          测试外部连接
        </Button>
      </Space>
    </Card>
  );
};

// 重要说明组件
const ImportantNotice: React.FC = () => {
  return (
    <Card 
      title="🚨 重要说明" 
      style={{ margin: '20px', border: '2px solid #ff4d4f' }}
      headStyle={{ backgroundColor: '#fff2f0' }}
    >
      <div style={{ color: '#ff4d4f', fontWeight: 'bold', marginBottom: '16px' }}>
        如果你看到了"启动系统"、"停止系统"、"重启系统"这些按钮，它们可能来自：
      </div>
      <ul>
        <li><strong>浏览器扩展</strong> - 某些浏览器扩展可能注入了这些按钮</li>
        <li><strong>其他页面组件</strong> - 可能有其他组件仍在渲染</li>
        <li><strong>缓存问题</strong> - 浏览器缓存了旧版本的页面</li>
        <li><strong>多个服务</strong> - 可能有多个前端服务在运行</li>
      </ul>
      
      <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#f0f2f5' }}>
        <strong>请只测试下面这些调试按钮：</strong>
        <br />
        🧪 基础点击测试 | 📝 控制台测试 | 🚀 测试 /system/start | 🔴 直接启动测试 | ❤️ 测试健康检查
      </div>
    </Card>
  );
};

const SimpleDashboard: React.FC = () => {
  React.useEffect(() => {
    console.log('🔥 [DEBUG] SimpleDashboard component mounted');
    console.log('🔥 [DEBUG] Current URL:', window.location.href);
    console.log('🔥 [DEBUG] Document ready state:', document.readyState);
    
    // 检查页面上是否有其他按钮
    setTimeout(() => {
      const allButtons = document.querySelectorAll('button');
      console.log('🔥 [DEBUG] Total buttons on page:', allButtons.length);
      allButtons.forEach((btn, index) => {
        console.log(`🔥 [DEBUG] Button ${index}:`, btn.textContent?.trim());
      });
    }, 1000);
  }, []);
  
  return (
    <div style={{ padding: '24px' }}>
      <h1>🔧 系统调试仪表板</h1>
      <ImportantNotice />
      <NetworkStatus />
      <SimpleApiTest />
    </div>
  );
};

export default SimpleDashboard;