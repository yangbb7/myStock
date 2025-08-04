#!/bin/bash

# myQuant Frontend Production Deployment Script
# This script handles comprehensive deployment with environment support

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_NAME="myquant-frontend"
PROJECT_DIR="/opt/myquant-frontend"
BACKUP_DIR="/opt/backups/myquant-frontend"
DOCKER_IMAGE="myquant-frontend"
HEALTH_CHECK_URL="http://localhost/health"
MAX_RETRIES=30
RETRY_INTERVAL=10

# Environment-specific settings
ENVIRONMENT="${1:-production}"
case "$ENVIRONMENT" in
    "development"|"dev")
        DOCKER_COMPOSE_FILE="docker-compose.yml"
        DOCKER_TAG="dev"
        PORT="3000"
        ;;
    "staging")
        DOCKER_COMPOSE_FILE="docker-compose.yml"
        DOCKER_TAG="staging"
        PORT="3000"
        ;;
    "production"|"prod")
        DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
        DOCKER_TAG="latest"
        PORT="80"
        ;;
    *)
        echo "Invalid environment: $ENVIRONMENT"
        echo "Valid environments: development, staging, production"
        exit 1
        ;;
esac

FULL_IMAGE_NAME="${DOCKER_IMAGE}:${DOCKER_TAG}"
HEALTH_CHECK_URL="http://localhost:${PORT}/health"

# =============================================================================
# Colors and Logging
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check available memory
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$AVAILABLE_MEMORY" -lt 1024 ]; then
        warning "Low available memory: ${AVAILABLE_MEMORY}MB (recommended: 1GB+)"
    fi
    
    # Check available disk space
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_DISK" -lt 5 ]; then
        warning "Low available disk space: ${AVAILABLE_DISK}GB (recommended: 5GB+)"
    fi
    
    success "System requirements check completed"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for $ENVIRONMENT environment..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check Docker version
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    debug "Docker version: $DOCKER_VERSION"
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Docker Compose version
    COMPOSE_VERSION=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    debug "Docker Compose version: $COMPOSE_VERSION"
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        error "Git is not installed"
        exit 1
    fi
    
    # Check if curl is installed (for health checks)
    if ! command -v curl &> /dev/null; then
        error "curl is not installed"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Validate environment configuration
validate_environment() {
    log "Validating environment configuration..."
    
    # Check if project directory exists
    if [ ! -d "$PROJECT_DIR" ]; then
        error "Project directory does not exist: $PROJECT_DIR"
        exit 1
    fi
    
    # Check if Docker Compose file exists
    if [ ! -f "$PROJECT_DIR/$DOCKER_COMPOSE_FILE" ]; then
        error "Docker Compose file not found: $PROJECT_DIR/$DOCKER_COMPOSE_FILE"
        exit 1
    fi
    
    # Check if environment file exists
    ENV_FILE="$PROJECT_DIR/.env.$ENVIRONMENT"
    if [ ! -f "$ENV_FILE" ] && [ "$ENVIRONMENT" != "development" ]; then
        warning "Environment file not found: $ENV_FILE"
        warning "Using default configuration"
    fi
    
    success "Environment validation completed"
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup() {
    log "Creating backup for $ENVIRONMENT environment..."
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Create backup filename with timestamp and environment
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/${PROJECT_NAME}-${ENVIRONMENT}-${TIMESTAMP}.tar"
    
    # Backup Docker image if it exists
    if docker image inspect "$FULL_IMAGE_NAME" &> /dev/null; then
        log "Backing up Docker image: $FULL_IMAGE_NAME"
        docker save "$FULL_IMAGE_NAME" > "$BACKUP_FILE"
        success "Docker image backup created: $BACKUP_FILE"
    else
        warning "No existing Docker image found to backup: $FULL_IMAGE_NAME"
    fi
    
    # Backup configuration files
    CONFIG_BACKUP="$BACKUP_DIR/config-${ENVIRONMENT}-${TIMESTAMP}.tar.gz"
    cd "$PROJECT_DIR"
    tar -czf "$CONFIG_BACKUP" \
        .env* \
        docker-compose*.yml \
        nginx*.conf \
        ssl.conf \
        monitoring/ \
        2>/dev/null || true
    
    if [ -f "$CONFIG_BACKUP" ]; then
        success "Configuration backup created: $CONFIG_BACKUP"
    fi
    
    # Keep only the last 10 backups per environment
    log "Cleaning up old backups..."
    find "$BACKUP_DIR" -name "${PROJECT_NAME}-${ENVIRONMENT}-*.tar" -type f | \
        sort -r | tail -n +11 | xargs -r rm -f
    find "$BACKUP_DIR" -name "config-${ENVIRONMENT}-*.tar.gz" -type f | \
        sort -r | tail -n +11 | xargs -r rm -f
    
    success "Backup process completed"
}

# =============================================================================
# Service Management Functions
# =============================================================================

stop_services() {
    log "Stopping existing services..."
    
    cd "$PROJECT_DIR"
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        # Graceful shutdown with timeout
        timeout 60 docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans || {
            warning "Graceful shutdown timed out, forcing stop..."
            docker-compose -f "$DOCKER_COMPOSE_FILE" kill
            docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
        }
        success "Services stopped"
    else
        warning "Docker Compose file not found, skipping service stop"
    fi
}

start_services() {
    log "Starting services for $ENVIRONMENT environment..."
    
    cd "$PROJECT_DIR"
    
    # Set environment file if it exists
    ENV_FILE=".env.$ENVIRONMENT"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        debug "Loaded environment variables from $ENV_FILE"
    fi
    
    # Start services in detached mode
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    success "Services started"
}

# =============================================================================
# Source Code Management
# =============================================================================

update_source() {
    log "Updating source code..."
    
    cd "$PROJECT_DIR"
    
    # Stash any local changes
    if ! git diff --quiet; then
        warning "Local changes detected, stashing..."
        git stash push -m "Auto-stash before deployment $(date)"
    fi
    
    # Fetch latest changes
    git fetch origin
    
    # Get current and remote commit hashes
    LOCAL_COMMIT=$(git rev-parse HEAD)
    REMOTE_COMMIT=$(git rev-parse origin/main)
    
    if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
        info "Already up to date (commit: ${LOCAL_COMMIT:0:8})"
        return 0
    fi
    
    log "Updating from ${LOCAL_COMMIT:0:8} to ${REMOTE_COMMIT:0:8}..."
    git pull origin main
    
    success "Source code updated"
}

# =============================================================================
# Build Functions
# =============================================================================

build_image() {
    log "Building Docker image for $ENVIRONMENT..."
    
    cd "$PROJECT_DIR"
    
    # Choose appropriate Dockerfile
    DOCKERFILE="Dockerfile"
    if [ "$ENVIRONMENT" = "production" ]; then
        DOCKERFILE="Dockerfile.prod"
    fi
    
    if [ ! -f "$DOCKERFILE" ]; then
        error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi
    
    # Build arguments
    BUILD_ARGS=""
    if [ "$ENVIRONMENT" = "production" ]; then
        BUILD_ARGS="--build-arg NODE_ENV=production"
    elif [ "$ENVIRONMENT" = "staging" ]; then
        BUILD_ARGS="--build-arg NODE_ENV=staging"
    fi
    
    # Build the image with progress output
    log "Building image: $FULL_IMAGE_NAME"
    docker build \
        -f "$DOCKERFILE" \
        -t "$FULL_IMAGE_NAME" \
        $BUILD_ARGS \
        --progress=plain \
        .
    
    # Tag with timestamp for versioning
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    docker tag "$FULL_IMAGE_NAME" "${DOCKER_IMAGE}:${ENVIRONMENT}-${TIMESTAMP}"
    
    success "Docker image built successfully: $FULL_IMAGE_NAME"
}

# =============================================================================
# Health Check Functions
# =============================================================================

health_check() {
    log "Performing health check..."
    
    local retries=0
    local start_time=$(date +%s)
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -f -s --max-time 5 "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            success "Health check passed after ${duration}s"
            return 0
        fi
        
        retries=$((retries + 1))
        log "Health check attempt $retries/$MAX_RETRIES failed, retrying in ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done
    
    error "Health check failed after $MAX_RETRIES attempts"
    return 1
}

# Extended health checks
extended_health_check() {
    log "Performing extended health checks..."
    
    # Check container status
    if ! docker-compose -f "$PROJECT_DIR/$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        error "Some containers are not running"
        return 1
    fi
    
    # Check API connectivity (if backend is available)
    if curl -f -s --max-time 5 "http://localhost:8000/health" > /dev/null 2>&1; then
        success "Backend API is accessible"
    else
        warning "Backend API is not accessible"
    fi
    
    # Check WebSocket connectivity
    if command -v wscat &> /dev/null; then
        if timeout 5 wscat -c "ws://localhost:8000/ws" -x "ping" > /dev/null 2>&1; then
            success "WebSocket connection is working"
        else
            warning "WebSocket connection failed"
        fi
    fi
    
    success "Extended health checks completed"
}

# =============================================================================
# Rollback Functions
# =============================================================================

rollback() {
    error "Deployment failed, initiating rollback..."
    
    # Stop current services
    cd "$PROJECT_DIR"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    # Find the latest backup for this environment
    LATEST_BACKUP=$(find "$BACKUP_DIR" -name "${PROJECT_NAME}-${ENVIRONMENT}-*.tar" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_BACKUP" ] && [ -f "$LATEST_BACKUP" ]; then
        log "Restoring from backup: $(basename "$LATEST_BACKUP")"
        docker load < "$LATEST_BACKUP"
        
        # Start services
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        
        # Wait and check health
        sleep 30
        if health_check; then
            success "Rollback completed successfully"
            send_notification "ROLLBACK_SUCCESS" "Rollback completed successfully for $ENVIRONMENT"
        else
            error "Rollback failed, manual intervention required"
            send_notification "ROLLBACK_FAILED" "Rollback failed for $ENVIRONMENT, manual intervention required"
            exit 1
        fi
    else
        error "No backup found for rollback"
        send_notification "ROLLBACK_FAILED" "No backup found for rollback in $ENVIRONMENT"
        exit 1
    fi
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup() {
    log "Cleaning up..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused volumes (be careful in production)
    if [ "$ENVIRONMENT" != "production" ]; then
        docker volume prune -f
    fi
    
    # Remove unused networks
    docker network prune -f
    
    success "Cleanup completed"
}

# =============================================================================
# Notification Functions
# =============================================================================

send_notification() {
    local status=$1
    local message=$2
    
    log "Sending notification: $status - $message"
    
    # Slack notification
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 myQuant Frontend Deployment\\n**Environment:** $ENVIRONMENT\\n**Status:** $status\\n**Message:** $message\\n**Time:** $(date)\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    # Email notification
    if [ -n "${ALERT_EMAIL:-}" ] && command -v mail &> /dev/null; then
        echo "Deployment $status for $ENVIRONMENT environment: $message" | \
            mail -s "myQuant Frontend Deployment - $status" "$ALERT_EMAIL" || true
    fi
    
    # Log to syslog
    logger -t "myquant-deployment" "Environment: $ENVIRONMENT, Status: $status, Message: $message"
}

# =============================================================================
# Monitoring Functions
# =============================================================================

setup_monitoring() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log "Setting up monitoring for production..."
        
        # Ensure monitoring directories exist
        mkdir -p "$PROJECT_DIR/monitoring/"{prometheus,grafana,loki,promtail}
        
        # Start monitoring stack if not already running
        if ! docker ps | grep -q prometheus; then
            log "Starting monitoring stack..."
            docker-compose -f "$PROJECT_DIR/$DOCKER_COMPOSE_FILE" up -d prometheus grafana loki promtail
        fi
        
        success "Monitoring setup completed"
    fi
}

# =============================================================================
# Security Functions
# =============================================================================

security_check() {
    log "Performing security checks..."
    
    # Check for exposed ports
    EXPOSED_PORTS=$(docker ps --format "table {{.Ports}}" | grep -v PORTS | grep "0.0.0.0" || true)
    if [ -n "$EXPOSED_PORTS" ]; then
        warning "Exposed ports detected:"
        echo "$EXPOSED_PORTS"
    fi
    
    # Check for running containers as root
    ROOT_CONTAINERS=$(docker ps --format "{{.Names}}" | xargs -I {} docker exec {} whoami 2>/dev/null | grep -c "^root$" || true)
    if [ "$ROOT_CONTAINERS" -gt 0 ]; then
        warning "$ROOT_CONTAINERS containers running as root"
    fi
    
    success "Security checks completed"
}

# =============================================================================
# Main Deployment Function
# =============================================================================

deploy() {
    local start_time=$(date +%s)
    
    log "Starting deployment of $PROJECT_NAME to $ENVIRONMENT environment..."
    info "Docker image: $FULL_IMAGE_NAME"
    info "Compose file: $DOCKER_COMPOSE_FILE"
    info "Health check URL: $HEALTH_CHECK_URL"
    
    # Pre-deployment checks
    check_system_requirements
    validate_environment
    
    # Create backup
    create_backup
    
    # Stop existing services
    stop_services
    
    # Update source code
    update_source
    
    # Build new image
    build_image
    
    # Start services
    start_services
    
    # Setup monitoring (production only)
    setup_monitoring
    
    # Perform health checks
    if health_check && extended_health_check; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        success "Deployment completed successfully in ${duration}s!"
        send_notification "SUCCESS" "Frontend deployment to $ENVIRONMENT completed successfully in ${duration}s"
        
        # Security check
        security_check
        
        # Cleanup
        cleanup
        
        # Display deployment info
        info "Deployment Summary:"
        info "- Environment: $ENVIRONMENT"
        info "- Image: $FULL_IMAGE_NAME"
        info "- Duration: ${duration}s"
        info "- Health check: ✓ Passed"
        
    else
        rollback
        send_notification "FAILED" "Frontend deployment to $ENVIRONMENT failed and was rolled back"
        exit 1
    fi
}

# =============================================================================
# Command Line Interface
# =============================================================================

show_help() {
    cat << EOF
myQuant Frontend Deployment Script

Usage: $0 [ENVIRONMENT] [COMMAND] [OPTIONS]

Environments:
  development, dev    - Development environment
  staging            - Staging environment  
  production, prod   - Production environment (default)

Commands:
  deploy             - Deploy the application (default)
  rollback           - Rollback to the previous version
  health-check       - Check application health
  cleanup            - Clean up unused Docker resources
  backup             - Create backup only
  build              - Build Docker image only
  start              - Start services only
  stop               - Stop services only
  status             - Show service status
  logs               - Show service logs

Options:
  --debug            - Enable debug output
  --force            - Force deployment without confirmations
  --no-backup        - Skip backup creation
  --no-cleanup       - Skip cleanup after deployment

Examples:
  $0 production deploy
  $0 staging rollback
  $0 development health-check
  $0 production deploy --debug
  $0 staging build --no-backup

EOF
}

# Parse command line arguments
COMMAND="deploy"
DEBUG=false
FORCE=false
NO_BACKUP=false
NO_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        development|dev|staging|production|prod)
            ENVIRONMENT=$1
            ;;
        deploy|rollback|health-check|cleanup|backup|build|start|stop|status|logs)
            COMMAND=$1
            ;;
        --debug)
            DEBUG=true
            ;;
        --force)
            FORCE=true
            ;;
        --no-backup)
            NO_BACKUP=true
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Update configuration based on parsed environment
case "$ENVIRONMENT" in
    "development"|"dev")
        DOCKER_COMPOSE_FILE="docker-compose.yml"
        DOCKER_TAG="dev"
        PORT="3000"
        ;;
    "staging")
        DOCKER_COMPOSE_FILE="docker-compose.yml"
        DOCKER_TAG="staging"
        PORT="3000"
        ;;
    "production"|"prod")
        DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
        DOCKER_TAG="latest"
        PORT="80"
        ;;
esac

FULL_IMAGE_NAME="${DOCKER_IMAGE}:${DOCKER_TAG}"
HEALTH_CHECK_URL="http://localhost:${PORT}/health"

# Confirmation for production
if [ "$ENVIRONMENT" = "production" ] && [ "$FORCE" = "false" ]; then
    echo -e "${YELLOW}WARNING: You are about to deploy to PRODUCTION environment.${NC}"
    echo -e "${YELLOW}This action will affect the live system.${NC}"
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
fi

# Execute command
case "$COMMAND" in
    deploy)
        check_root
        check_prerequisites
        deploy
        ;;
    rollback)
        check_root
        check_prerequisites
        rollback
        ;;
    health-check)
        health_check && extended_health_check
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        create_backup
        ;;
    build)
        check_prerequisites
        build_image
        ;;
    start)
        check_prerequisites
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        ;;
    logs)
        cd "$PROJECT_DIR"
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f --tail=100
        ;;
    *)
        error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac