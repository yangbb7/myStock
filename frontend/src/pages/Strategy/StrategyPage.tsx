import React, { useState } from 'react';
import { Tabs, Card } from 'antd';
import { CodeOutlined, BlockOutlined, AppstoreOutlined, BarChartOutlined, RobotOutlined } from '@ant-design/icons';
import StrategyConfigForm from './StrategyConfigForm';
import StrategyOperations from './StrategyOperations';
import StrategyPerformanceMonitor from './StrategyPerformanceMonitor';
import VisualStrategyBuilder from './VisualStrategyBuilder';
import StrategyTemplateLibrary from './StrategyTemplateLibrary';
import AIStrategyAssistant from './AIStrategyAssistant';

const { TabPane } = Tabs;

const StrategyPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('visual');

  return (
    <div className="strategy-page">
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane
            tab={
              <span>
                <BlockOutlined />
                可视化构建
              </span>
            }
            key="visual"
          >
            <VisualStrategyBuilder />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <AppstoreOutlined />
                策略模板
              </span>
            }
            key="templates"
          >
            <StrategyTemplateLibrary />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <RobotOutlined />
                AI助手
              </span>
            }
            key="ai"
          >
            <AIStrategyAssistant />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <CodeOutlined />
                代码编辑
              </span>
            }
            key="code"
          >
            <StrategyConfigForm />
          </TabPane>
          
          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                策略监控
              </span>
            }
            key="monitor"
          >
            <StrategyPerformanceMonitor />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default StrategyPage;