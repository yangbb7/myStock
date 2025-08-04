# 持续集成API测试指南

## GitHub Actions配置

### 完整的CI/CD流程配置

```yaml
# .github/workflows/api-integration-test.yml
name: API Integration Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # 每日自动测试
    - cron: '0 2 * * *'

jobs:
  api-integration-test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: myquant_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Python dependencies
      run: |
        cd myQuant
        pip install -r requirements.txt

    - name: Install Node.js dependencies
      run: |
        cd frontend-web-interface
        npm ci

    - name: Start backend service
      run: |
        cd myQuant
        python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
        sleep 10

    - name: Wait for backend to be ready
      run: |
        timeout 30 bash -c 'until curl -f http://localhost:8000/health; do sleep 1; done'

    - name: Build frontend
      run: |
        cd frontend-web-interface
        npm run build

    - name: Start frontend service
      run: |
        cd frontend-web-interface
        npm run preview -- --port 3000 &
        sleep 5

    - name: Run API integration tests
      run: |
        cd frontend-web-interface
        npm run test:integration

    - name: Run E2E tests with Playwright
      run: |
        cd frontend-web-interface
        npx playwright test

    - name: Generate test report
      run: |
        cd frontend-web-interface
        npm run test:report

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          frontend-web-interface/test-results/
          frontend-web-interface/playwright-report/

    - name: Comment PR with test results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const path = './frontend-web-interface/test-results/summary.json';
          if (fs.existsSync(path)) {
            const results = JSON.parse(fs.readFileSync(path, 'utf8'));
            const comment = `## API Integration Test Results
            
            - **Total Tests**: ${results.total}
            - **Passed**: ${results.passed}
            - **Failed**: ${results.failed}
            - **Success Rate**: ${results.successRate}%
            
            ${results.failed > 0 ? '❌ Some tests failed. Please check the details.' : '✅ All tests passed!'}`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          }
```

## 测试脚本配置

### package.json测试脚本

```json
{
  "scripts": {
    "test:integration": "jest --config jest.integration.config.js",
    "test:e2e": "playwright test",
    "test:api": "jest --config jest.api.config.js",
    "test:report": "node scripts/generate-test-report.js",
    "test:all": "npm run test:integration && npm run test:e2e && npm run test:report"
  }
}
```

### Jest集成测试配置

```javascript
// jest.integration.config.js
module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/integration/**/*.test.js'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup/integration.js'],
  testTimeout: 30000,
  collectCoverage: true,
  coverageDirectory: 'coverage/integration',
  coverageReporters: ['text', 'lcov', 'html'],
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'test-results',
      outputName: 'integration-results.xml'
    }]
  ]
};
```

### Playwright E2E测试配置

```javascript
// playwright.config.js
module.exports = {
  testDir: './tests/e2e',
  timeout: 30000,
  expect: {
    timeout: 5000
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['junit', { outputFile: 'test-results/e2e-results.xml' }]
  ],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    }
  ],
  webServer: [
    {
      command: 'cd ../myQuant && python -m uvicorn main:app --port 8000',
      port: 8000,
      reuseExistingServer: !process.env.CI
    },
    {
      command: 'npm run preview -- --port 3000',
      port: 3000,
      reuseExistingServer: !process.env.CI
    }
  ]
};
```

## 测试报告生成

### 自动化测试报告脚本

```javascript
// scripts/generate-test-report.js
const fs = require('fs');
const path = require('path');

async function generateTestReport() {
  const results = {
    timestamp: new Date().toISOString(),
    integration: await parseJestResults('test-results/integration-results.xml'),
    e2e: await parsePlaywrightResults('test-results/e2e-results.xml'),
    api: await runApiHealthCheck()
  };

  const summary = {
    total: results.integration.total + results.e2e.total,
    passed: results.integration.passed + results.e2e.passed,
    failed: results.integration.failed + results.e2e.failed,
    successRate: Math.round(((results.integration.passed + results.e2e.passed) / (results.integration.total + results.e2e.total)) * 100)
  };

  // 生成HTML报告
  const htmlReport = generateHtmlReport(results, summary);
  fs.writeFileSync('test-results/report.html', htmlReport);

  // 生成JSON摘要
  fs.writeFileSync('test-results/summary.json', JSON.stringify(summary, null, 2));

  console.log(`Test Report Generated: ${summary.passed}/${summary.total} tests passed (${summary.successRate}%)`);
}

async function runApiHealthCheck() {
  const endpoints = [
    'http://localhost:8000/health',
    'http://localhost:8000/metrics',
    'http://localhost:3000'
  ];

  const results = [];
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint);
      results.push({
        endpoint,
        status: response.status,
        healthy: response.ok
      });
    } catch (error) {
      results.push({
        endpoint,
        status: 0,
        healthy: false,
        error: error.message
      });
    }
  }

  return results;
}

generateTestReport().catch(console.error);
```

## 监控和告警

### 测试失败通知配置

```yaml
# .github/workflows/notify-on-failure.yml
name: Test Failure Notification

on:
  workflow_run:
    workflows: ["API Integration Tests"]
    types:
      - completed

jobs:
  notify:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    
    steps:
    - name: Send Slack notification
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#dev-alerts'
        text: |
          🚨 API Integration Tests Failed!
          
          Repository: ${{ github.repository }}
          Branch: ${{ github.ref }}
          Commit: ${{ github.sha }}
          
          Please check the test results and fix the issues.
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

    - name: Create GitHub Issue
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: `API Integration Tests Failed - ${new Date().toISOString().split('T')[0]}`,
            body: `## Test Failure Report
            
            The API integration tests have failed on branch \`${context.ref}\`.
            
            **Workflow Run**: ${context.payload.workflow_run.html_url}
            **Commit**: ${context.sha}
            
            Please investigate and fix the failing tests.
            
            ### Next Steps
            1. Check the workflow logs for detailed error information
            2. Run tests locally to reproduce the issue
            3. Fix the failing tests or API compatibility issues
            4. Create a pull request with the fixes
            
            This issue will be automatically closed when the tests pass again.`,
            labels: ['bug', 'ci-failure', 'high-priority']
          });
```

## 本地开发测试

### 开发环境测试脚本

```bash
#!/bin/bash
# scripts/run-local-tests.sh

echo "🚀 Starting local API integration tests..."

# 启动后端服务
echo "📡 Starting backend service..."
cd myQuant
python -m uvicorn main:app --port 8000 &
BACKEND_PID=$!

# 等待后端启动
echo "⏳ Waiting for backend to be ready..."
timeout 30 bash -c 'until curl -f http://localhost:8000/health; do sleep 1; done'

if [ $? -ne 0 ]; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID
    exit 1
fi

# 启动前端服务
echo "🌐 Starting frontend service..."
cd ../frontend-web-interface
npm run build
npm run preview -- --port 3000 &
FRONTEND_PID=$!

# 等待前端启动
echo "⏳ Waiting for frontend to be ready..."
sleep 5

# 运行测试
echo "🧪 Running integration tests..."
npm run test:integration

echo "🎭 Running E2E tests..."
npm run test:e2e

echo "📊 Generating test report..."
npm run test:report

# 清理进程
echo "🧹 Cleaning up..."
kill $BACKEND_PID $FRONTEND_PID

echo "✅ Local tests completed!"
```

## 性能监控集成

### API性能监控配置

```javascript
// tests/performance/api-performance.test.js
const { performance } = require('perf_hooks');

describe('API Performance Tests', () => {
  const performanceThresholds = {
    '/health': 100,
    '/metrics': 150,
    '/portfolio/summary': 200,
    '/strategy/performance': 250
  };

  Object.entries(performanceThresholds).forEach(([endpoint, threshold]) => {
    test(`${endpoint} should respond within ${threshold}ms`, async () => {
      const start = performance.now();
      
      const response = await fetch(`http://localhost:8000${endpoint}`);
      
      const end = performance.now();
      const responseTime = end - start;

      expect(response.ok).toBe(true);
      expect(responseTime).toBeLessThan(threshold);
      
      console.log(`${endpoint}: ${responseTime.toFixed(2)}ms`);
    });
  });
});
```

这个持续集成测试指南提供了完整的自动化测试流程，确保API集成的持续稳定性和兼容性。