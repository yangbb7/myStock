# myQuant Frontend

Modern web interface for the myQuant modular monolithic quantitative trading system.

## Features

- **Real-time Dashboard**: System status, portfolio overview, and risk monitoring
- **Strategy Management**: Create, configure, and monitor trading strategies
- **Data Monitoring**: Real-time market data and technical analysis
- **Order Management**: Track and manage trading orders
- **Portfolio Analysis**: Comprehensive investment portfolio analysis
- **Risk Management**: Real-time risk monitoring and alerts
- **Backtesting**: Strategy backtesting and simulation
- **System Administration**: System configuration and monitoring

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **UI Library**: Ant Design 5.0+
- **State Management**: Zustand + React Query
- **Charts**: Apache ECharts
- **Real-time**: Socket.IO
- **HTTP Client**: Axios
- **Utilities**: Day.js, Lodash

## Development Setup

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting
- `npm run type-check` - Run TypeScript type checking
- `npm run preview` - Preview production build

### Development Tools

- **ESLint**: Code linting with TypeScript support
- **Prettier**: Code formatting
- **Husky**: Git hooks for pre-commit checks
- **lint-staged**: Run linters on staged files

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout/         # Layout components
│   ├── Charts/         # Chart components
│   ├── Forms/          # Form components
│   └── Common/         # Common components
├── pages/              # Page components
│   ├── Dashboard/      # System dashboard
│   ├── Strategy/       # Strategy management
│   ├── Data/           # Data monitoring
│   ├── Orders/         # Order management
│   ├── Portfolio/      # Portfolio analysis
│   ├── Risk/           # Risk management
│   ├── Backtest/       # Backtesting
│   └── System/         # System administration
├── services/           # API and external services
│   ├── api.ts          # HTTP API client
│   ├── websocket.ts    # WebSocket service
│   └── types.ts        # Type definitions
├── stores/             # State management
│   ├── systemStore.ts  # System state
│   ├── dataStore.ts    # Data state
│   └── userStore.ts    # User state
├── hooks/              # Custom React hooks
│   ├── useApi.ts       # API hooks
│   ├── useWebSocket.ts # WebSocket hooks
│   └── useRealTime.ts  # Real-time data hooks
├── utils/              # Utility functions
│   ├── constants.ts    # Application constants
│   ├── format.ts       # Formatting utilities
│   └── helpers.ts      # Helper functions
└── styles/             # Global styles and themes
    ├── globals.css     # Global CSS
    └── themes/         # Theme configurations
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# Application Configuration
VITE_APP_TITLE=myQuant Frontend
VITE_APP_VERSION=1.0.0

# Development Configuration
VITE_DEV_MODE=true
VITE_ENABLE_MOCK=false
```

## API Integration

The frontend integrates with the myQuant backend API:

- **Health Check**: `/health` - System health status
- **Metrics**: `/metrics` - System metrics and performance
- **Strategy Management**: `/strategy/*` - Strategy operations
- **Data Access**: `/data/*` - Market data and tick data
- **Order Management**: `/order/*` - Order operations
- **Portfolio**: `/portfolio/*` - Portfolio information
- **Risk Management**: `/risk/*` - Risk metrics and alerts
- **Analytics**: `/analytics/*` - Performance analysis

## Real-time Features

- WebSocket connections for real-time data updates
- Live market data streaming
- Real-time system status monitoring
- Order status updates
- Risk alert notifications

## Build and Deployment

### Production Build

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

### Docker Deployment

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Contributing

1. Follow the established code style (ESLint + Prettier)
2. Write TypeScript with proper type definitions
3. Add unit tests for new components
4. Update documentation as needed
5. Use conventional commit messages

## License

This project is part of the myQuant trading system.