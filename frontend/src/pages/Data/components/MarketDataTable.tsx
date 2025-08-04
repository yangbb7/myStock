import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  Table,
  Input,
  Select,
  Button,
  Space,
  Tag,
  Tooltip,
  Typography,
  Row,
  Col,
  Statistic,
  Badge,
  message,
  Modal,
  Form,
  AutoComplete,
} from 'antd';
import {
  SearchOutlined,
  ReloadOutlined,
  DownloadOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  SortDescendingOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import type { ColumnsType, TableProps } from 'antd/es/table';
import { useMarketData } from '../../../hooks/useRealTime';
import { MarketData } from '../../../services/types';
import { formatCurrency, formatPercent, formatNumber, formatTime } from '../../../utils/format';
import { api } from '../../../services/api';

const { Search } = Input;
const { Option } = Select;
const { Text } = Typography;

interface MarketDataTableProps {
  height?: number;
  showControls?: boolean;
  symbols?: string[];
}

interface MarketDataRow extends MarketData {
  key: string;
  priceChange?: number;
  priceChangePercent?: number;
  isRising?: boolean;
  isFalling?: boolean;
}

const MarketDataTable: React.FC<MarketDataTableProps> = ({
  height = 600,
  showControls = true,
  symbols: initialSymbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH'],
}) => {
  const [searchText, setSearchText] = useState('');
  const [sortField, setSortField] = useState<string>('symbol');
  const [sortOrder, setSortOrder] = useState<'ascend' | 'descend'>('ascend');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(initialSymbols);
  const [filteredData, setFilteredData] = useState<MarketDataRow[]>([]);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [addForm] = Form.useForm();
  const [searchOptions, setSearchOptions] = useState<{value: string, label: string}[]>([]);
  const [stockInfoCache, setStockInfoCache] = useState<Record<string, {name: string, exchange: string}>>({});

  // Use REST API to fetch real-time data instead of WebSocket
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch real-time data from REST API using batch endpoint
  const fetchMarketData = useCallback(async () => {
    if (selectedSymbols.length === 0) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('🔄 [MarketDataTable] Batch fetching data for symbols:', selectedSymbols);
      const startTime = Date.now();
      
      // Use batch API for better performance
      const response = await api.data.getRealTimePricesBatch(selectedSymbols);
      
      if (response && response.success && response.data && response.data.results) {
        const newMarketData: Record<string, MarketData> = {};
        
        response.data.results.forEach((result: any) => {
          if (result.success && result.current_price > 0) {
            newMarketData[result.symbol] = {
              symbol: result.symbol,
              name: result.name || result.symbol,
              price: result.current_price,
              previousClose: 0, // 批量API暂时不提供这些数据
              open: 0,
              high: 0,
              low: 0,
              volume: 0,
              timestamp: result.timestamp,
            };
          }
        });

        setMarketData(newMarketData);
        setIsConnected(true);
        
        const endTime = Date.now();
        const duration = endTime - startTime;
        console.log(`✅ [MarketDataTable] Batch data fetched successfully in ${duration}ms:`, newMarketData);
        console.log(`📊 [MarketDataTable] Success rate: ${response.data.successful}/${response.data.total}`);
        
      } else {
        throw new Error('Invalid batch response format');
      }
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(new Error(errorMessage));
      setIsConnected(false);
      console.error('❌ [MarketDataTable] Batch fetch failed:', err);
      
      // Fallback to individual API calls if batch fails
      console.log('🔄 [MarketDataTable] Falling back to individual API calls...');
      try {
        const dataPromises = selectedSymbols.map(async (symbol) => {
          try {
            const response = await api.data.getRealTimePrice(symbol);
            if (response && response.success && response.data) {
              return {
                symbol,
                data: {
                  symbol: response.data.symbol,
                  name: response.data.name || symbol,
                  price: response.data.current_price || 0,
                  previousClose: response.data.previous_close || 0,
                  open: response.data.open || 0,
                  high: response.data.high || 0,
                  low: response.data.low || 0,
                  volume: response.data.volume || 0,
                  timestamp: response.data.timestamp || new Date().toISOString(),
                }
              };
            }
            return null;
          } catch (err) {
            console.error(`❌ [MarketDataTable] Failed to fetch ${symbol}:`, err);
            return null;
          }
        });

        const results = await Promise.all(dataPromises);
        const newMarketData: Record<string, MarketData> = {};
        
        results.forEach(result => {
          if (result) {
            newMarketData[result.symbol] = result.data;
          }
        });

        setMarketData(newMarketData);
        setIsConnected(true);
        setError(null);
        console.log('✅ [MarketDataTable] Fallback data fetched successfully:', newMarketData);
        
      } catch (fallbackErr) {
        console.error('❌ [MarketDataTable] Fallback also failed:', fallbackErr);
      }
    } finally {
      setIsLoading(false);
    }
  }, [selectedSymbols]);

  // Auto-fetch data on mount and when symbols change
  useEffect(() => {
    fetchMarketData();
    
    // Set up interval to refresh data every 5 seconds
    const interval = setInterval(fetchMarketData, 5000);
    
    return () => clearInterval(interval);
  }, [fetchMarketData]);

  // 使用从 REST API 获取的真实市场数据
  const realMarketData = useMemo(() => {
    return marketData;
  }, [marketData]);

  // Transform market data to table rows
  const tableData = useMemo(() => {
    const rows: MarketDataRow[] = Object.entries(realMarketData).map(([symbol, data]) => {
      const previousPrice = data.previousClose || data.price;
      const priceChange = data.price - previousPrice;
      const priceChangePercent = previousPrice > 0 ? (priceChange / previousPrice) * 100 : 0;

      // 优先使用WebSocket数据中的股票名称，其次使用缓存的名称
      const stockName = data.name || stockInfoCache[symbol]?.name || symbol;
      
      return {
        ...data,
        key: symbol,
        name: stockName,  // 确保 name 字段存在
        priceChange,
        priceChangePercent,
        isRising: priceChange > 0,
        isFalling: priceChange < 0,
      };
    });

    return rows;
  }, [realMarketData, stockInfoCache]);

  // Filter and sort data
  useEffect(() => {
    let filtered = tableData;

    // Apply search filter
    if (searchText) {
      filtered = filtered.filter(item =>
        item.symbol.toLowerCase().includes(searchText.toLowerCase())
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aValue = a[sortField as keyof MarketDataRow];
      const bValue = b[sortField as keyof MarketDataRow];
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'ascend' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'ascend' ? aValue - bValue : bValue - aValue;
      }
      
      return 0;
    });

    setFilteredData(filtered);
    setLastUpdateTime(new Date());
  }, [tableData, searchText, sortField, sortOrder]);

  // Handle symbol selection change
  const handleSymbolChange = useCallback((symbols: string[]) => {
    setSelectedSymbols(symbols);
  }, []);

  // Initialize search options from API
  useEffect(() => {
    const loadInitialStocks = async () => {
      try {
        const searchResult = await api.stock.search('', 20);
        if (searchResult?.data?.stocks) {
          const options = searchResult.data.stocks.map((stock: any) => ({
            value: stock.symbol,
            label: `${stock.name}(${stock.symbol})`
          }));
          setSearchOptions(options);
          
          // Cache stock info
          const cache: Record<string, {name: string, exchange: string}> = {};
          searchResult.data.stocks.forEach((stock: any) => {
            cache[stock.symbol] = {
              name: stock.name,
              exchange: stock.exchange
            };
          });
          setStockInfoCache(cache);
        }
      } catch (error) {
        console.warn('Failed to load initial stocks from API:', error);
      }
    };
    
    loadInitialStocks();
  }, []);

  // Handle add custom stock
  const handleAddStock = useCallback(() => {
    setIsAddModalVisible(true);
  }, []);

  // Handle modal cancel
  const handleModalCancel = useCallback(() => {
    setIsAddModalVisible(false);
    addForm.resetFields();
  }, [addForm]);

  // Validate stock symbol format
  const validateStockSymbol = (symbol: string): boolean => {
    // 中国股市代码格式：6位数字.SH 或 6位数字.SZ 或 3位数字.SZ
    const pattern = /^(\d{6}\.(SH|SZ)|\d{3}\.(SZ))$/;
    return pattern.test(symbol.toUpperCase());
  };
  
  // Get stock name from cache or symbol
  const getStockName = (symbol: string): string => {
    return stockInfoCache[symbol]?.name || symbol;
  };
  
  // Get stock display name
  const getStockDisplayName = (symbol: string): string => {
    const name = getStockName(symbol);
    return name !== symbol ? `${name}(${symbol})` : symbol;
  };

  // Handle add stock form submit
  const handleAddStockSubmit = useCallback(async () => {
    try {
      const values = await addForm.validateFields();
      const { stockInput } = values;
      
      let symbolToAdd = '';
      
      // Check if input is a symbol or search by name
      if (validateStockSymbol(stockInput)) {
        symbolToAdd = stockInput.toUpperCase();
      } else {
        // Search in cache first
        const foundSymbol = Object.entries(stockInfoCache).find(
          ([symbol, info]) => info.name.includes(stockInput) || symbol.includes(stockInput.toUpperCase())
        );
        
        if (foundSymbol) {
          symbolToAdd = foundSymbol[0];
        } else {
          // Try to construct symbol if it's a 6-digit number
          const numericInput = stockInput.replace(/\D/g, '');
          if (numericInput.length === 6) {
            // Default to .SH for 6-digit codes starting with 6, .SZ for others
            const exchange = numericInput.startsWith('6') ? 'SH' : 'SZ';
            symbolToAdd = `${numericInput}.${exchange}`;
          } else if (numericInput.length === 3) {
            symbolToAdd = `${numericInput}.SZ`;
          } else {
            message.error('请输入有效的股票代码或名称');
            return;
          }
        }
      }
      
      if (!selectedSymbols.includes(symbolToAdd)) {
        // Try to get stock info from API to get the real name
        try {
          const stockInfo = await api.stock.getInfo(symbolToAdd);
          const stockName = stockInfo?.data?.name || symbolToAdd;
          
          // Update cache
          setStockInfoCache(prev => ({
            ...prev,
            [symbolToAdd]: {
              name: stockName,
              exchange: stockInfo?.data?.exchange || (symbolToAdd.endsWith('.SH') ? 'SH' : 'SZ')
            }
          }));
          
          const newSymbols = [...selectedSymbols, symbolToAdd];
          setSelectedSymbols(newSymbols);
          message.success(`已添加股票: ${stockName}(${symbolToAdd})`);
        } catch (error) {
          // Still add the symbol even if API fails
          const newSymbols = [...selectedSymbols, symbolToAdd];
          setSelectedSymbols(newSymbols);
          message.success(`已添加股票: ${symbolToAdd}`);
        }
      } else {
        message.warning('该股票已在监控列表中');
      }
      
      handleModalCancel();
    } catch (error) {
      console.error('添加股票失败:', error);
    }
  }, [addForm, selectedSymbols, handleModalCancel]);

  // Handle search in add modal
  const handleStockSearch = useCallback(async (searchText: string) => {
    if (!searchText) {
      // Load default stocks when no search text
      try {
        const searchResult = await api.stock.search('', 20);
        if (searchResult?.data?.stocks) {
          const options = searchResult.data.stocks.map((stock: any) => ({
            value: stock.symbol,
            label: `${stock.name}(${stock.symbol})`
          }));
          setSearchOptions(options);
        }
      } catch (error) {
        console.warn('Failed to load default stocks:', error);
        setSearchOptions([]);
      }
      return;
    }

    try {
      // Search from API
      const searchResult = await api.stock.search(searchText, 15);
      if (searchResult?.data?.stocks && searchResult.data.stocks.length > 0) {
        const apiOptions = searchResult.data.stocks.map((stock: any) => ({
          value: stock.symbol,
          label: `${stock.name}(${stock.symbol})`
        }));
        setSearchOptions(apiOptions);
        
        // Update cache with search results
        const newCache: Record<string, {name: string, exchange: string}> = {};
        searchResult.data.stocks.forEach((stock: any) => {
          newCache[stock.symbol] = {
            name: stock.name,
            exchange: stock.exchange
          };
        });
        setStockInfoCache(prev => ({ ...prev, ...newCache }));
      } else {
        setSearchOptions([]);
      }
    } catch (error) {
      console.warn('API search failed:', error);
      setSearchOptions([]);
    }
  }, []);

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    await fetchMarketData();
    message.success('数据刷新成功');
  }, [fetchMarketData]);

  // Handle export
  const handleExport = useCallback(() => {
    const csvContent = [
      ['股票代码', '最新价', '涨跌额', '涨跌幅', '成交量', '更新时间'].join(','),
      ...filteredData.map(row => [
        row.symbol,
        row.price.toFixed(2),
        (row.priceChange || 0).toFixed(2),
        ((row.priceChangePercent || 0)).toFixed(2) + '%',
        row.volume.toLocaleString(),
        formatTime(row.timestamp),
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `market_data_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    message.success('数据导出成功');
  }, [filteredData]);

  // Price change cell renderer
  const renderPriceChange = (value: number, record: MarketDataRow) => {
    const color = value > 0 ? '#52c41a' : value < 0 ? '#ff4d4f' : '#666';
    const prefix = value > 0 ? '+' : '';
    
    return (
      <Text style={{ color, fontWeight: 'bold' }}>
        {prefix}{formatCurrency(value)}
      </Text>
    );
  };

  // Price change percent cell renderer
  const renderPriceChangePercent = (value: number, record: MarketDataRow) => {
    const color = value > 0 ? '#52c41a' : value < 0 ? '#ff4d4f' : '#666';
    const prefix = value > 0 ? '+' : '';
    
    return (
      <Text style={{ color, fontWeight: 'bold' }}>
        {prefix}{value.toFixed(2)}%
      </Text>
    );
  };

  // Price cell renderer with animation
  const renderPrice = (value: number, record: MarketDataRow) => {
    const color = record.isRising ? '#52c41a' : record.isFalling ? '#ff4d4f' : '#666';
    
    return (
      <Text 
        style={{ 
          color, 
          fontWeight: 'bold',
          fontSize: '16px',
          transition: 'all 0.3s ease',
        }}
      >
        {formatCurrency(value)}
      </Text>
    );
  };

  // Volume cell renderer
  const renderVolume = (value: number) => {
    return (
      <Text>{formatNumber(value)}</Text>
    );
  };

  // Timestamp cell renderer
  const renderTimestamp = (value: string) => {
    return (
      <Tooltip title={new Date(value).toLocaleString()}>
        <Text type="secondary">{formatTime(value)}</Text>
      </Tooltip>
    );
  };

  // Table columns configuration
  const columns: ColumnsType<MarketDataRow> = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 150,
      fixed: 'left',
      sorter: true,
      render: (value: string, record: MarketDataRow) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Text strong style={{ fontSize: '12px', color: '#666' }}>{record.name || value}</Text>
            {record.isRising && <Badge status="success" />}
            {record.isFalling && <Badge status="error" />}
          </Space>
          <Text type="secondary" style={{ fontSize: '11px' }}>{value}</Text>
        </Space>
      ),
    },
    {
      title: '最新价',
      dataIndex: 'price',
      key: 'price',
      width: 100,
      align: 'right',
      sorter: true,
      render: renderPrice,
    },
    {
      title: '涨跌额',
      dataIndex: 'priceChange',
      key: 'priceChange',
      width: 100,
      align: 'right',
      sorter: true,
      render: renderPriceChange,
    },
    {
      title: '涨跌幅',
      dataIndex: 'priceChangePercent',
      key: 'priceChangePercent',
      width: 100,
      align: 'right',
      sorter: true,
      render: renderPriceChangePercent,
    },
    {
      title: '成交量',
      dataIndex: 'volume',
      key: 'volume',
      width: 120,
      align: 'right',
      sorter: true,
      render: renderVolume,
    },
    {
      title: '最高价',
      dataIndex: 'high',
      key: 'high',
      width: 100,
      align: 'right',
      sorter: true,
      render: (value?: number) => value ? formatCurrency(value) : '-',
    },
    {
      title: '最低价',
      dataIndex: 'low',
      key: 'low',
      width: 100,
      align: 'right',
      sorter: true,
      render: (value?: number) => value ? formatCurrency(value) : '-',
    },
    {
      title: '开盘价',
      dataIndex: 'open',
      key: 'open',
      width: 100,
      align: 'right',
      sorter: true,
      render: (value?: number) => value ? formatCurrency(value) : '-',
    },
    {
      title: '更新时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      align: 'center',
      sorter: true,
      render: renderTimestamp,
    },
  ];

  // Table change handler
  const handleTableChange: TableProps<MarketDataRow>['onChange'] = (pagination, filters, sorter) => {
    if (Array.isArray(sorter)) return;
    
    if (sorter.field && sorter.order) {
      setSortField(sorter.field as string);
      setSortOrder(sorter.order);
    }
  };

  // Connection status indicator
  const connectionStatus = (
    <Space>
      <Badge 
        status={isConnected ? 'success' : 'error'} 
        text={isConnected ? '已连接' : '未连接'} 
      />
      {lastUpdateTime && (
        <Text type="secondary">
          最后更新: {formatTime(lastUpdateTime.toISOString())}
        </Text>
      )}
    </Space>
  );

  return (
    <Card
      title={
        <Space>
          <Text strong>实时行情数据</Text>
          {connectionStatus}
        </Space>
      }
      extra={
        showControls && (
          <Space>
            <Button
              icon={<PlusOutlined />}
              onClick={handleAddStock}
              type="primary"
            >
              添加股票
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
              loading={!isConnected}
            >
              刷新
            </Button>
            <Button
              icon={<DownloadOutlined />}
              onClick={handleExport}
              disabled={filteredData.length === 0}
            >
              导出
            </Button>
          </Space>
        )
      }
    >
      {showControls && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Search
              placeholder="搜索股票代码"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              onSearch={setSearchText}
              allowClear
            />
          </Col>
          <Col span={8}>
            <Select
              mode="multiple"
              placeholder="选择股票代码"
              value={selectedSymbols}
              onChange={handleSymbolChange}
              style={{ width: '100%' }}
              maxTagCount={3}
              optionLabelProp="label"
              loading={searchOptions.length === 0}
            >
              {searchOptions.map(option => (
                <Option key={option.value} value={option.value} label={option.label}>
                  {option.label}
                </Option>
              ))}
            </Select>
          </Col>
          <Col span={8}>
            <Space>
              <Select
                value={sortField}
                onChange={setSortField}
                style={{ width: 120 }}
              >
                <Option value="symbol">股票代码</Option>
                <Option value="price">最新价</Option>
                <Option value="priceChange">涨跌额</Option>
                <Option value="priceChangePercent">涨跌幅</Option>
                <Option value="volume">成交量</Option>
              </Select>
              <Button
                icon={sortOrder === 'ascend' ? <SortAscendingOutlined /> : <SortDescendingOutlined />}
                onClick={() => setSortOrder(sortOrder === 'ascend' ? 'descend' : 'ascend')}
              />
            </Space>
          </Col>
        </Row>
      )}

      {error && (
        <div style={{ marginBottom: 16 }}>
          <Tag color="error">连接错误: {error.message}</Tag>
        </div>
      )}

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic
            title="监控股票数"
            value={selectedSymbols.length}
            suffix="只"
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="上涨股票"
            value={filteredData.filter(item => item.isRising).length}
            suffix="只"
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="下跌股票"
            value={filteredData.filter(item => item.isFalling).length}
            suffix="只"
            valueStyle={{ color: '#ff4d4f' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="平盘股票"
            value={filteredData.filter(item => !item.isRising && !item.isFalling).length}
            suffix="只"
            valueStyle={{ color: '#666' }}
          />
        </Col>
      </Row>

      <Table<MarketDataRow>
        columns={columns}
        dataSource={filteredData}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showQuickJumper: true,
          showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
        }}
        scroll={{ y: height - 200, x: 1000 }}
        size="small"
        onChange={handleTableChange}
        rowClassName={(record) => {
          if (record.isRising) return 'market-data-row-rising';
          if (record.isFalling) return 'market-data-row-falling';
          return '';
        }}
      />

      <style jsx>{`
        .market-data-row-rising {
          background-color: rgba(82, 196, 26, 0.05);
        }
        .market-data-row-falling {
          background-color: rgba(255, 77, 79, 0.05);
        }
        .market-data-row-rising:hover,
        .market-data-row-falling:hover {
          background-color: rgba(0, 0, 0, 0.05) !important;
        }
      `}</style>

      {/* Add Stock Modal */}
      <Modal
        title="添加自定义股票"
        open={isAddModalVisible}
        onOk={handleAddStockSubmit}
        onCancel={handleModalCancel}
        okText="添加"
        cancelText="取消"
        width={500}
      >
        <Form
          form={addForm}
          layout="vertical"
          initialValues={{
            stockInput: ''
          }}
        >
          <Form.Item
            label="股票代码或名称"
            name="stockInput"
            rules={[
              { required: true, message: '请输入股票代码或名称' },
              { min: 3, message: '请输入至少3个字符' }
            ]}
          >
            <AutoComplete
              options={searchOptions}
              onSearch={handleStockSearch}
              placeholder="输入股票代码(如: 600519.SH)或名称(如: 贵州茅台)"
              style={{ width: '100%' }}
              filterOption={(inputValue, option) =>
                option!.label.toLowerCase().indexOf(inputValue.toLowerCase()) !== -1
              }
            />
          </Form.Item>
          <div style={{ color: '#666', fontSize: '12px', marginTop: '8px' }}>
            <div>支持的格式：</div>
            <div>• 完整代码：600519.SH, 000001.SZ</div>
            <div>• 6位数字：600519 (自动判断交易所)</div>
            <div>• 股票名称：贵州茅台, 平安银行</div>
          </div>
        </Form>
      </Modal>
    </Card>
  );
};

export { MarketDataTable };