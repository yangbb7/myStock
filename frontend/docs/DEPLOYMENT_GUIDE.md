# myQuant Frontend 部署指南

## 目录

1. [部署概述](#部署概述)
2. [环境准备](#环境准备)
3. [本地部署](#本地部署)
4. [Docker部署](#docker部署)
5. [生产环境部署](#生产环境部署)
6. [监控和维护](#监控和维护)
7. [故障排除](#故障排除)

## 部署概述

myQuant Frontend 是一个基于React的单页应用程序(SPA)，支持多种部署方式：

- **开发环境**: 使用Vite开发服务器
- **测试环境**: 使用Docker容器部署
- **生产环境**: 使用Nginx + Docker部署

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │      Nginx      │    │   myQuant API   │
│    (Optional)   │────│   (Frontend)    │────│   (Backend)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │   Static Files  │
                       │   (dist/)       │
                       └─────────────────┘
```

## 环境准备

### 系统要求

#### 开发环境
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Node.js**: 18.0.0+
- **npm**: 9.0.0+
- **内存**: 最少4GB，推荐8GB+
- **磁盘空间**: 最少2GB可用空间

#### 生产环境
- **操作系统**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **Docker**: 20.10.0+
- **Docker Compose**: 2.0.0+
- **内存**: 最少2GB，推荐4GB+
- **磁盘空间**: 最少5GB可用空间
- **网络**: 稳定的互联网连接

### 依赖服务

- **myQuant Backend API**: 后端API服务必须先部署并运行
- **数据库**: 后端依赖的数据库服务
- **Redis**: 缓存服务（可选）

## 本地部署

### 1. 克隆项目

```bash
# 克隆项目仓库
git clone <repository-url>
cd myquant-frontend

# 切换到frontend目录
cd frontend
```

### 2. 安装依赖

```bash
# 安装项目依赖
npm install

# 验证安装
npm list --depth=0
```

### 3. 环境配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

环境变量配置示例：

```bash
# .env
# API配置
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# 应用配置
VITE_APP_TITLE=myQuant Frontend
VITE_APP_VERSION=1.0.0

# 开发配置
VITE_DEV_MODE=true
VITE_ENABLE_MOCK=false

# 监控配置
VITE_ENABLE_ANALYTICS=false
VITE_SENTRY_DSN=
```

### 4. 启动开发服务器

```bash
# 启动开发服务器
npm run dev

# 服务器将在 http://localhost:3000 启动
```

### 5. 验证部署

访问 `http://localhost:3000` 验证应用是否正常运行：

- 检查页面是否正常加载
- 验证API连接是否正常
- 测试主要功能模块

## Docker部署

### 1. 构建Docker镜像

创建Dockerfile：

```dockerfile
# Dockerfile
FROM node:18-alpine as builder

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production && npm cache clean --force

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 生产阶段
FROM nginx:alpine

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 创建日志目录
RUN mkdir -p /var/log/nginx

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
```

### 2. Nginx配置

创建nginx.conf：

```nginx
# nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # 基本配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 16M;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;

        # 安全头
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

        # 前端路由支持
        location / {
            try_files $uri $uri/ /index.html;
            
            # 缓存策略
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }

        # API代理
        location /api/ {
            proxy_pass http://backend:8000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # WebSocket代理
        location /ws/ {
            proxy_pass http://backend:8000/ws/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # 健康检查
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # 错误页面
        error_page 404 /index.html;
        error_page 500 502 503 504 /50x.html;
        location = /50x.html {
            root /usr/share/nginx/html;
        }
    }
}
```

### 3. 构建和运行

```bash
# 构建Docker镜像
docker build -t myquant-frontend:latest .

# 运行容器
docker run -d \
  --name myquant-frontend \
  -p 3000:80 \
  --restart unless-stopped \
  myquant-frontend:latest

# 查看容器状态
docker ps
docker logs myquant-frontend
```

### 4. Docker Compose部署

创建docker-compose.yml：

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: myquant-frontend
    ports:
      - "3000:80"
    environment:
      - NODE_ENV=production
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - myquant-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  backend:
    image: myquant-backend:latest
    container_name: myquant-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/myquant
    restart: unless-stopped
    networks:
      - myquant-network

networks:
  myquant-network:
    driver: bridge
```

启动服务：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f frontend
```

## 生产环境部署

### 1. 服务器准备

#### 系统更新

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

#### 安装Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 防火墙配置

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 2. SSL证书配置

#### 使用Let's Encrypt

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取SSL证书
sudo certbot --nginx -d yourdomain.com

# 自动续期
sudo crontab -e
# 添加以下行
0 12 * * * /usr/bin/certbot renew --quiet
```

#### Nginx SSL配置

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # SSL配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    # 其他配置...
}

# HTTP重定向到HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. 生产环境配置

#### 环境变量

```bash
# .env.production
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WS_BASE_URL=wss://api.yourdomain.com
VITE_APP_TITLE=myQuant Trading System
VITE_APP_VERSION=1.0.0
VITE_DEV_MODE=false
VITE_ENABLE_ANALYTICS=true
VITE_SENTRY_DSN=https://your-sentry-dsn
```

#### 构建优化

```bash
# 生产构建
npm run build

# 分析构建产物
npm install -g webpack-bundle-analyzer
npx webpack-bundle-analyzer dist/static/js/*.js
```

### 4. 部署脚本

创建部署脚本 `deploy.sh`：

```bash
#!/bin/bash

set -e

echo "开始部署 myQuant Frontend..."

# 配置变量
PROJECT_DIR="/opt/myquant-frontend"
BACKUP_DIR="/opt/backups/myquant-frontend"
DOCKER_IMAGE="myquant-frontend:latest"

# 创建备份
echo "创建备份..."
mkdir -p $BACKUP_DIR
docker save $DOCKER_IMAGE > $BACKUP_DIR/myquant-frontend-$(date +%Y%m%d_%H%M%S).tar

# 停止现有服务
echo "停止现有服务..."
docker-compose -f $PROJECT_DIR/docker-compose.yml down

# 拉取最新代码
echo "更新代码..."
cd $PROJECT_DIR
git pull origin main

# 构建新镜像
echo "构建新镜像..."
docker build -t $DOCKER_IMAGE .

# 启动服务
echo "启动服务..."
docker-compose up -d

# 健康检查
echo "等待服务启动..."
sleep 30

# 验证部署
if curl -f http://localhost:3000/health; then
    echo "部署成功！"
else
    echo "部署失败，回滚..."
    docker-compose down
    docker load < $BACKUP_DIR/myquant-frontend-$(ls -t $BACKUP_DIR | head -1)
    docker-compose up -d
    exit 1
fi

# 清理旧备份（保留最近5个）
find $BACKUP_DIR -name "*.tar" -type f -mtime +5 -delete

echo "部署完成！"
```

使用部署脚本：

```bash
# 赋予执行权限
chmod +x deploy.sh

# 执行部署
./deploy.sh
```

### 5. 负载均衡配置

#### Nginx负载均衡

```nginx
upstream frontend_servers {
    server frontend1:80 weight=1 max_fails=3 fail_timeout=30s;
    server frontend2:80 weight=1 max_fails=3 fail_timeout=30s;
    server frontend3:80 backup;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://frontend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 健康检查
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
    }
}
```

## 监控和维护

### 1. 日志管理

#### 配置日志轮转

```bash
# 创建logrotate配置
sudo nano /etc/logrotate.d/myquant-frontend

# 内容如下
/var/log/nginx/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 nginx nginx
    postrotate
        if [ -f /var/run/nginx.pid ]; then
            kill -USR1 `cat /var/run/nginx.pid`
        fi
    endscript
}
```

#### 日志监控脚本

```bash
#!/bin/bash
# monitor-logs.sh

LOG_FILE="/var/log/nginx/error.log"
ALERT_EMAIL="admin@yourdomain.com"

# 检查错误日志
ERROR_COUNT=$(tail -n 100 $LOG_FILE | grep -c "error")

if [ $ERROR_COUNT -gt 10 ]; then
    echo "检测到大量错误: $ERROR_COUNT" | mail -s "Frontend Error Alert" $ALERT_EMAIL
fi
```

### 2. 性能监控

#### 系统资源监控

```bash
#!/bin/bash
# monitor-resources.sh

# CPU使用率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

# 内存使用率
MEM_USAGE=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')

# 磁盘使用率
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

echo "CPU: ${CPU_USAGE}%, Memory: ${MEM_USAGE}%, Disk: ${DISK_USAGE}%"

# 告警阈值
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "CPU使用率过高: ${CPU_USAGE}%" | mail -s "High CPU Alert" admin@yourdomain.com
fi
```

#### 应用监控

```bash
#!/bin/bash
# monitor-app.sh

# 检查容器状态
if ! docker ps | grep -q myquant-frontend; then
    echo "Frontend容器未运行" | mail -s "Container Down Alert" admin@yourdomain.com
    docker-compose up -d
fi

# 检查应用响应
if ! curl -f http://localhost:3000/health; then
    echo "Frontend应用无响应" | mail -s "App Down Alert" admin@yourdomain.com
fi
```

### 3. 自动化维护

#### 定时任务配置

```bash
# 编辑crontab
crontab -e

# 添加以下任务
# 每5分钟检查应用状态
*/5 * * * * /opt/scripts/monitor-app.sh

# 每小时检查系统资源
0 * * * * /opt/scripts/monitor-resources.sh

# 每天凌晨2点清理Docker镜像
0 2 * * * docker system prune -f

# 每周日凌晨3点重启服务
0 3 * * 0 /opt/myquant-frontend/restart.sh
```

#### 自动更新脚本

```bash
#!/bin/bash
# auto-update.sh

cd /opt/myquant-frontend

# 检查是否有新版本
git fetch origin
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ $LOCAL != $REMOTE ]; then
    echo "发现新版本，开始自动更新..."
    ./deploy.sh
    echo "自动更新完成"
else
    echo "已是最新版本"
fi
```

## 故障排除

### 1. 常见问题

#### 应用无法启动

```bash
# 检查Docker容器状态
docker ps -a

# 查看容器日志
docker logs myquant-frontend

# 检查端口占用
netstat -tulpn | grep :3000

# 检查磁盘空间
df -h
```

#### API连接失败

```bash
# 检查后端服务状态
curl -f http://backend:8000/health

# 检查网络连接
docker network ls
docker network inspect myquant-network

# 检查DNS解析
nslookup backend
```

#### 静态资源加载失败

```bash
# 检查Nginx配置
nginx -t

# 重新加载Nginx配置
nginx -s reload

# 检查文件权限
ls -la /usr/share/nginx/html/
```

### 2. 性能问题

#### 页面加载缓慢

```bash
# 检查网络延迟
ping yourdomain.com

# 检查DNS解析时间
dig yourdomain.com

# 分析网络请求
curl -w "@curl-format.txt" -o /dev/null -s http://yourdomain.com
```

#### 内存使用过高

```bash
# 检查内存使用
free -h
docker stats

# 分析内存泄漏
docker exec -it myquant-frontend top
```

### 3. 安全问题

#### SSL证书问题

```bash
# 检查证书状态
openssl x509 -in /etc/letsencrypt/live/yourdomain.com/cert.pem -text -noout

# 测试SSL配置
openssl s_client -connect yourdomain.com:443

# 续期证书
certbot renew --dry-run
```

#### 安全扫描

```bash
# 使用nmap扫描端口
nmap -sS -O yourdomain.com

# 检查SSL安全性
sslscan yourdomain.com

# 检查HTTP头安全性
curl -I https://yourdomain.com
```

### 4. 恢复操作

#### 从备份恢复

```bash
# 停止当前服务
docker-compose down

# 恢复Docker镜像
docker load < /opt/backups/myquant-frontend/backup.tar

# 重启服务
docker-compose up -d
```

#### 数据库恢复

```bash
# 如果有数据库依赖，恢复数据库
docker exec -i postgres psql -U user -d myquant < backup.sql
```

## 联系支持

如果遇到无法解决的问题，请联系技术支持：

- **邮箱**: devops@yourdomain.com
- **电话**: +86-400-123-4567
- **文档**: https://docs.yourdomain.com
- **问题跟踪**: https://github.com/yourorg/myquant-frontend/issues

---

*本指南最后更新时间: 2025年1月*