#!/bin/bash

# Production build script for myQuant Frontend
set -e

# Configuration
PROJECT_NAME="myquant-frontend"
BUILD_DIR="dist"
DOCKER_IMAGE="myquant-frontend"
DOCKER_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed"
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        error "npm is not installed"
        exit 1
    fi
    
    # Check Node.js version
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    REQUIRED_VERSION="18.0.0"
    
    if ! node -e "process.exit(require('semver').gte('$NODE_VERSION', '$REQUIRED_VERSION') ? 0 : 1)" 2>/dev/null; then
        error "Node.js version $NODE_VERSION is not supported. Required: $REQUIRED_VERSION or higher"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Clean previous builds
clean_build() {
    log "Cleaning previous builds..."
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log "Removed existing build directory"
    fi
    
    # Clean npm cache
    npm cache clean --force
    
    success "Build cleanup completed"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    # Install production dependencies
    npm ci --only=production --no-audit --no-fund
    
    success "Dependencies installed"
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Type checking
    log "Running type check..."
    npm run type-check
    
    # Linting
    log "Running linter..."
    npm run lint
    
    # Unit tests
    log "Running unit tests..."
    npm run test:run
    
    success "All tests passed"
}

# Build application
build_application() {
    log "Building application for production..."
    
    # Set production environment
    export NODE_ENV=production
    
    # Build the application
    npm run build
    
    # Verify build output
    if [ ! -d "$BUILD_DIR" ]; then
        error "Build failed - output directory not found"
        exit 1
    fi
    
    if [ ! -f "$BUILD_DIR/index.html" ]; then
        error "Build failed - index.html not found"
        exit 1
    fi
    
    # Display build statistics
    log "Build statistics:"
    du -sh "$BUILD_DIR"
    find "$BUILD_DIR" -name "*.js" -o -name "*.css" | wc -l | xargs echo "Total assets:"
    
    success "Application built successfully"
}

# Optimize build
optimize_build() {
    log "Optimizing build..."
    
    # Compress static assets if tools are available
    if command -v gzip &> /dev/null; then
        log "Creating gzip compressed assets..."
        find "$BUILD_DIR" -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" \) -exec gzip -k {} \;
    fi
    
    if command -v brotli &> /dev/null; then
        log "Creating brotli compressed assets..."
        find "$BUILD_DIR" -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" \) -exec brotli -k {} \;
    fi
    
    success "Build optimization completed"
}

# Build Docker image
build_docker_image() {
    log "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        warning "Docker not found, skipping image build"
        return 0
    fi
    
    # Build production Docker image
    docker build -f Dockerfile.prod -t "$DOCKER_IMAGE:$DOCKER_TAG" .
    
    # Tag with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    docker tag "$DOCKER_IMAGE:$DOCKER_TAG" "$DOCKER_IMAGE:$TIMESTAMP"
    
    success "Docker image built: $DOCKER_IMAGE:$DOCKER_TAG"
}

# Generate build report
generate_report() {
    log "Generating build report..."
    
    REPORT_FILE="build-report-$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "myQuant Frontend Build Report"
        echo "============================="
        echo "Build Date: $(date)"
        echo "Node.js Version: $(node --version)"
        echo "npm Version: $(npm --version)"
        echo ""
        echo "Build Statistics:"
        echo "Build Size: $(du -sh $BUILD_DIR | cut -f1)"
        echo "Total Files: $(find $BUILD_DIR -type f | wc -l)"
        echo "JS Files: $(find $BUILD_DIR -name "*.js" | wc -l)"
        echo "CSS Files: $(find $BUILD_DIR -name "*.css" | wc -l)"
        echo "Image Files: $(find $BUILD_DIR -name "*.png" -o -name "*.jpg" -o -name "*.svg" | wc -l)"
        echo ""
        echo "File Listing:"
        find "$BUILD_DIR" -type f -exec ls -lh {} \; | sort -k5 -hr
    } > "$REPORT_FILE"
    
    success "Build report generated: $REPORT_FILE"
}

# Main build function
main() {
    log "Starting production build for $PROJECT_NAME..."
    
    check_prerequisites
    clean_build
    install_dependencies
    run_tests
    build_application
    optimize_build
    build_docker_image
    generate_report
    
    success "Production build completed successfully!"
    log "Build output available in: $BUILD_DIR"
    log "Docker image: $DOCKER_IMAGE:$DOCKER_TAG"
}

# Parse command line arguments
case "${1:-build}" in
    build)
        main
        ;;
    clean)
        clean_build
        ;;
    test)
        run_tests
        ;;
    docker)
        build_docker_image
        ;;
    *)
        echo "Usage: $0 {build|clean|test|docker}"
        echo ""
        echo "Commands:"
        echo "  build  - Full production build (default)"
        echo "  clean  - Clean build artifacts"
        echo "  test   - Run tests only"
        echo "  docker - Build Docker image only"
        exit 1
        ;;
esac