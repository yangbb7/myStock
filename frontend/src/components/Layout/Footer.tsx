import React from 'react';
import { Layout, Space, Divider } from 'antd';
import { GithubOutlined, HeartOutlined } from '@ant-design/icons';

const { Footer: AntFooter } = Layout;

interface FooterProps {
  showLinks?: boolean;
  showCopyright?: boolean;
  customContent?: React.ReactNode;
}

export const Footer: React.FC<FooterProps> = ({
  showLinks = true,
  showCopyright = true,
  customContent,
}) => {
  const currentYear = new Date().getFullYear();

  return (
    <AntFooter
      className="footer-container"
      style={{
        textAlign: 'center',
        padding: '24px 50px',
      }}
    >
      {customContent || (
        <div>
          {showLinks && (
            <Space split={<Divider type="vertical" />} size="middle">
              <a
                href="https://github.com/myquant"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#666' }}
              >
                <GithubOutlined /> GitHub
              </a>
              <a
                href="/docs"
                style={{ color: '#666' }}
              >
                文档
              </a>
              <a
                href="/api"
                style={{ color: '#666' }}
              >
                API
              </a>
              <a
                href="/support"
                style={{ color: '#666' }}
              >
                支持
              </a>
            </Space>
          )}
          
          {showLinks && showCopyright && <Divider />}
          
          {showCopyright && (
            <div style={{ color: '#666', fontSize: '12px' }}>
              <div>
                myQuant 量化交易系统 ©{currentYear} Created with{' '}
                <HeartOutlined style={{ color: '#ff4d4f' }} /> by myQuant Team
              </div>
              <div style={{ marginTop: '4px' }}>
                版本 1.0.0 | 构建时间: {new Date().toLocaleString()}
              </div>
            </div>
          )}
        </div>
      )}
    </AntFooter>
  );
};

export default Footer;