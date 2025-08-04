import React, { useState, useMemo } from 'react';
import {
  Card,
  Alert,
  List,
  Badge,
  Typography,
  Space,
  Button,
  Tag,
  Row,
  Col,
  Statistic,
  Empty,
  Modal,
  Table,
  DatePicker,
  Select,
  Input,
  Divider,
  Tooltip,
  Progress,
  notification
} from 'antd';
import {
  ExclamationCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  ClearOutlined,
  BellOutlined,
  HistoryOutlined,
  FilterOutlined,
  ExportOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import { useRiskAlerts, RiskAlert } from '../../hooks/useRealTime';

const { Text, Title } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { Search } = Input;

interface RiskAlertsSystemProps {
  maxDisplayAlerts?: number;
  showMetrics?: boolean;
  autoRefresh?: boolean;
}

interface AlertFilter {
  level?: string[];
  dateRange?: [dayjs.Dayjs, dayjs.Dayjs];
  keyword?: string;
}

const RiskAlertsSystem: React.FC<RiskAlertsSystemProps> = ({
  maxDisplayAlerts = 20,
  showMetrics = true,
  autoRefresh = true
}) => {
  const [isHistoryModalVisible, setIsHistoryModalVisible] = useState(false);
  const [filter, setFilter] = useState<AlertFilter>({});
  const [acknowledgedAlerts, setAcknowledgedAlerts] = useState<Set<string>>(new Set());

  // Real-time risk alerts
  const {
    alerts,
    latestAlert,
    isConnected,
    error,
    clearAlerts,
    dismissAlert
  } = useRiskAlerts({
    autoSubscribe: autoRefresh,
    maxAlertsSize: 100
  });

  // Fetch historical alerts
  const {
    data: historicalAlerts,
    isLoading: isLoadingHistory,
    refetch: refetchHistory
  } = useQuery({
    queryKey: ['riskAlerts', filter],
    queryFn: () => api.risk.getAlerts({
      startDate: filter.dateRange?.[0]?.toISOString(),
      endDate: filter.dateRange?.[1]?.toISOString(),
    }),
    enabled: isHistoryModalVisible,
  });

  // Alert statistics
  const alertStats = useMemo(() => {
    const stats = {
      total: alerts.length,
      critical: 0,
      error: 0,
      warning: 0,
      acknowledged: acknowledgedAlerts.size,
      unacknowledged: 0
    };

    alerts.forEach(alert => {
      switch (alert.level) {
        case 'critical':
          stats.critical++;
          break;
        case 'error':
          stats.error++;
          break;
        case 'warning':
          stats.warning++;
          break;
      }
    });

    stats.unacknowledged = stats.total - stats.acknowledged;
    return stats;
  }, [alerts, acknowledgedAlerts]);

  // Filtered alerts for display
  const filteredAlerts = useMemo(() => {
    let filtered = [...alerts];

    if (filter.level && filter.level.length > 0) {
      filtered = filtered.filter(alert => filter.level!.includes(alert.level));
    }

    if (filter.keyword) {
      const keyword = filter.keyword.toLowerCase();
      filtered = filtered.filter(alert =>
        alert.message.toLowerCase().includes(keyword)
      );
    }

    if (filter.dateRange) {
      const [start, end] = filter.dateRange;
      filtered = filtered.filter(alert => {
        const alertTime = dayjs(alert.timestamp);
        return alertTime.isAfter(start) && alertTime.isBefore(end);
      });
    }

    return filtered.slice(0, maxDisplayAlerts);
  }, [alerts, filter, maxDisplayAlerts]);

  const getAlertIcon = (level: RiskAlert['level']) => {
    switch (level) {
      case 'critical':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff7a45' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      default:
        return <ExclamationCircleOutlined />;
    }
  };

  const getAlertColor = (level: RiskAlert['level']): string => {
    switch (level) {
      case 'critical':
        return '#ff4d4f';
      case 'error':
        return '#ff7a45';
      case 'warning':
        return '#faad14';
      default:
        return '#1890ff';
    }
  };

  const getAlertPriority = (level: RiskAlert['level']): number => {
    switch (level) {
      case 'critical':
        return 3;
      case 'error':
        return 2;
      case 'warning':
        return 1;
      default:
        return 0;
    }
  };

  const acknowledgeAlert = (alertIndex: number) => {
    const alert = alerts[alertIndex];
    if (alert) {
      const alertId = `${alert.timestamp}-${alertIndex}`;
      setAcknowledgedAlerts(prev => new Set([...prev, alertId]));
      
      notification.success({
        message: '告警已确认',
        description: `已确认${alert.level === 'critical' ? '严重' : alert.level === 'error' ? '错误' : '警告'}告警`,
        duration: 2,
      });
    }
  };

  const isAlertAcknowledged = (alertIndex: number): boolean => {
    const alert = alerts[alertIndex];
    if (!alert) return false;
    const alertId = `${alert.timestamp}-${alertIndex}`;
    return acknowledgedAlerts.has(alertId);
  };

  const formatRiskMetrics = (metrics: RiskAlert['metrics']) => {
    return (
      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title="日盈亏"
            value={metrics.dailyPnl}
            precision={2}
            prefix="¥"
            valueStyle={{
              color: metrics.dailyPnl >= 0 ? '#3f8600' : '#cf1322',
              fontSize: '12px'
            }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="当前回撤"
            value={metrics.currentDrawdown}
            precision={2}
            suffix="%"
            valueStyle={{
              color: '#cf1322',
              fontSize: '12px'
            }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="风险利用率"
            value={metrics.riskUtilization.dailyLossRatio * 100}
            precision={1}
            suffix="%"
            valueStyle={{
              color: metrics.riskUtilization.dailyLossRatio > 0.8 ? '#cf1322' : '#3f8600',
              fontSize: '12px'
            }}
          />
        </Col>
      </Row>
    );
  };

  const exportAlerts = () => {
    const csvContent = [
      ['时间', '级别', '消息', '日盈亏', '当前回撤', '风险利用率'].join(','),
      ...filteredAlerts.map(alert => [
        new Date(alert.timestamp).toLocaleString(),
        alert.level === 'critical' ? '严重' : alert.level === 'error' ? '错误' : '警告',
        `"${alert.message}"`,
        alert.metrics.dailyPnl.toFixed(2),
        alert.metrics.currentDrawdown.toFixed(2),
        (alert.metrics.riskUtilization.dailyLossRatio * 100).toFixed(1)
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `risk_alerts_${dayjs().format('YYYY-MM-DD_HH-mm-ss')}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const historyColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a: any, b: any) => dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix(),
    },
    {
      title: '级别',
      dataIndex: 'level',
      key: 'level',
      render: (level: string) => (
        <Tag color={getAlertColor(level as RiskAlert['level'])}>
          {level === 'critical' ? '严重' : level === 'error' ? '错误' : '警告'}
        </Tag>
      ),
      filters: [
        { text: '严重', value: 'critical' },
        { text: '错误', value: 'error' },
        { text: '警告', value: 'warning' },
      ],
      onFilter: (value: any, record: any) => record.level === value,
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
    },
    {
      title: '日盈亏',
      dataIndex: ['metrics', 'dailyPnl'],
      key: 'dailyPnl',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          ¥{value.toFixed(2)}
        </span>
      ),
      sorter: (a: any, b: any) => a.metrics.dailyPnl - b.metrics.dailyPnl,
    },
    {
      title: '当前回撤',
      dataIndex: ['metrics', 'currentDrawdown'],
      key: 'currentDrawdown',
      render: (value: number) => `${value.toFixed(2)}%`,
      sorter: (a: any, b: any) => a.metrics.currentDrawdown - b.metrics.currentDrawdown,
    },
  ];

  if (!isConnected) {
    return (
      <Card title="风险告警系统">
        <Alert
          message="连接断开"
          description="WebSocket连接已断开，无法接收实时风险告警"
          type="warning"
          showIcon
        />
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="风险告警系统">
        <Alert
          message="连接错误"
          description={error.message}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  return (
    <>
      <Card
        title={
          <Space>
            <BellOutlined />
            <span>风险告警系统</span>
            {alertStats.unacknowledged > 0 && (
              <Badge count={alertStats.unacknowledged} />
            )}
          </Space>
        }
        extra={
          <Space>
            <Button
              size="small"
              icon={<FilterOutlined />}
              onClick={() => setIsHistoryModalVisible(true)}
            >
              历史告警
            </Button>
            <Button
              size="small"
              icon={<ExportOutlined />}
              onClick={exportAlerts}
              disabled={filteredAlerts.length === 0}
            >
              导出
            </Button>
            <Button
              size="small"
              icon={<ClearOutlined />}
              onClick={clearAlerts}
              disabled={alerts.length === 0}
            >
              清空
            </Button>
          </Space>
        }
      >
        {/* Latest Critical Alert */}
        {latestAlert && latestAlert.level === 'critical' && (
          <Alert
            message={`严重告警: ${latestAlert.message}`}
            description={
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  时间: {new Date(latestAlert.timestamp).toLocaleString()}
                </Text>
                {showMetrics && (
                  <div style={{ marginTop: '8px' }}>
                    {formatRiskMetrics(latestAlert.metrics)}
                  </div>
                )}
              </Space>
            }
            type="error"
            showIcon
            closable
            style={{ marginBottom: '16px' }}
          />
        )}

        {/* Alert Statistics */}
        <Row gutter={16} style={{ marginBottom: '16px' }}>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic
                title="总告警数"
                value={alertStats.total}
                valueStyle={{ fontSize: '16px' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic
                title="严重告警"
                value={alertStats.critical}
                valueStyle={{
                  color: alertStats.critical > 0 ? '#ff4d4f' : '#666',
                  fontSize: '16px'
                }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic
                title="错误告警"
                value={alertStats.error}
                valueStyle={{
                  color: alertStats.error > 0 ? '#ff7a45' : '#666',
                  fontSize: '16px'
                }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <Statistic
                title="未确认"
                value={alertStats.unacknowledged}
                valueStyle={{
                  color: alertStats.unacknowledged > 0 ? '#faad14' : '#666',
                  fontSize: '16px'
                }}
              />
            </Card>
          </Col>
        </Row>

        {/* Alert Filters */}
        <Row gutter={16} style={{ marginBottom: '16px' }}>
          <Col xs={24} sm={8}>
            <Select
              mode="multiple"
              placeholder="筛选告警级别"
              style={{ width: '100%' }}
              value={filter.level}
              onChange={(value) => setFilter(prev => ({ ...prev, level: value }))}
            >
              <Option value="critical">严重</Option>
              <Option value="error">错误</Option>
              <Option value="warning">警告</Option>
            </Select>
          </Col>
          <Col xs={24} sm={8}>
            <RangePicker
              style={{ width: '100%' }}
              value={filter.dateRange}
              onChange={(dates) => setFilter(prev => ({ ...prev, dateRange: dates as [dayjs.Dayjs, dayjs.Dayjs] }))}
              showTime
            />
          </Col>
          <Col xs={24} sm={8}>
            <Search
              placeholder="搜索告警消息"
              value={filter.keyword}
              onChange={(e) => setFilter(prev => ({ ...prev, keyword: e.target.value }))}
              allowClear
            />
          </Col>
        </Row>

        {/* Alert List */}
        {filteredAlerts.length === 0 ? (
          <Empty
            description="暂无告警信息"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <List
            itemLayout="vertical"
            dataSource={filteredAlerts}
            renderItem={(alert, index) => {
              const isAcknowledged = isAlertAcknowledged(index);
              const priority = getAlertPriority(alert.level);
              
              return (
                <List.Item
                  key={`${alert.timestamp}-${index}`}
                  style={{
                    opacity: isAcknowledged ? 0.6 : 1,
                    border: priority >= 2 ? `2px solid ${getAlertColor(alert.level)}` : undefined,
                    borderRadius: '4px',
                    padding: '12px',
                    marginBottom: '8px'
                  }}
                  actions={[
                    <Tooltip title={isAcknowledged ? '已确认' : '确认告警'}>
                      <Button
                        key="acknowledge"
                        type={isAcknowledged ? "default" : "primary"}
                        size="small"
                        icon={isAcknowledged ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                        onClick={() => acknowledgeAlert(index)}
                        disabled={isAcknowledged}
                      >
                        {isAcknowledged ? '已确认' : '确认'}
                      </Button>
                    </Tooltip>,
                    <Button
                      key="dismiss"
                      type="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => dismissAlert(index)}
                    >
                      忽略
                    </Button>
                  ]}
                  extra={
                    <Space direction="vertical" align="end">
                      <Tag color={getAlertColor(alert.level)}>
                        {alert.level === 'critical' ? '严重' : 
                         alert.level === 'error' ? '错误' : '警告'}
                      </Tag>
                      {priority >= 2 && (
                        <Tag color="red">高优先级</Tag>
                      )}
                      {isAcknowledged && (
                        <Tag color="green">已确认</Tag>
                      )}
                    </Space>
                  }
                >
                  <List.Item.Meta
                    avatar={getAlertIcon(alert.level)}
                    title={
                      <Space>
                        <Text strong style={{ color: isAcknowledged ? '#999' : undefined }}>
                          {alert.message}
                        </Text>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(alert.timestamp).toLocaleString()}
                        </Text>
                      </Space>
                    }
                    description={
                      showMetrics && (
                        <div style={{ marginTop: '8px' }}>
                          {formatRiskMetrics(alert.metrics)}
                        </div>
                      )
                    }
                  />
                </List.Item>
              );
            }}
          />
        )}

        {alerts.length > maxDisplayAlerts && (
          <div style={{ textAlign: 'center', marginTop: '16px' }}>
            <Text type="secondary">
              显示最近 {maxDisplayAlerts} 条告警，共 {alerts.length} 条
            </Text>
          </div>
        )}
      </Card>

      {/* Historical Alerts Modal */}
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            <span>历史告警</span>
          </Space>
        }
        open={isHistoryModalVisible}
        onCancel={() => setIsHistoryModalVisible(false)}
        width={1000}
        footer={null}
      >
        <Table
          columns={historyColumns}
          dataSource={historicalAlerts || []}
          loading={isLoadingHistory}
          rowKey={(record) => `${record.timestamp}-${record.message}`}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: 800 }}
        />
      </Modal>
    </>
  );
};

export default RiskAlertsSystem;