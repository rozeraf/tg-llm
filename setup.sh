#!/usr/bin/env bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="telegram-llm-bot"
REQUIRED_TOOLS=("docker" "docker-compose" "make")

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_blue() { echo -e "${BLUE}[INFO]${NC} $1"; }

print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    TELEGRAM LLM BOT SETUP                   ║
║              Production-Ready Multi-User Bot                ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed!"
            case "$tool" in
                "docker")
                    log_warn "Install Docker: https://docs.docker.com/engine/install/"
                    ;;
                "docker-compose")
                    log_warn "Install Docker Compose: https://docs.docker.com/compose/install/"
                    ;;
                "make")
                    log_warn "Install make: sudo pacman -S make (Arch Linux)"
                    ;;
            esac
            return 1
        else
            log_info "✓ $tool is installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running!"
        log_warn "Start Docker service: sudo systemctl start docker"
        return 1
    fi
    
    log_info "All requirements satisfied!"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env if it doesn't exist
    if [[ ! -f .env ]]; then
        log_info "Creating .env file from template..."
        cp .env.example .env 2>/dev/null || {
            log_warn ".env.example not found, creating basic .env"
            cat > .env << 'EOF'
# Telegram Bot Configuration
TELEGRAM_TOKEN=""
API_KEY=""

# PostgreSQL Database Configuration
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=tgllm
POSTGRES_USER=tgllm
POSTGRES_PASSWORD=""

# Security Configuration
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60

# Feature Flags
ENABLE_MONITORING=false
ENABLE_WEB_SEARCH=true
LOG_LEVEL=INFO
EOF
        }
        log_warn "Please edit .env file with your tokens and passwords!"
        log_warn "Required: TELEGRAM_TOKEN, API_KEY, POSTGRES_PASSWORD"
    else
        log_info "✓ .env file already exists"
    fi
    
    # Create directories
    local dirs=("uploads" "logs" "backups" "monitoring/grafana/dashboards" "monitoring/grafana/datasources")
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    # Set proper permissions
    chmod 700 uploads logs backups
    log_info "Set secure permissions for data directories"
}

validate_env() {
    log_info "Validating environment configuration..."
    
    if [[ ! -f .env ]]; then
        log_error ".env file not found!"
        return 1
    fi
    
    # Check required variables
    local required_vars=("TELEGRAM_TOKEN" "POSTGRES_PASSWORD")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=.." .env; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        printf '%s\n' "${missing_vars[@]}" | sed 's/^/  - /'
        log_warn "Please edit .env file and set these variables"
        return 1
    fi
    
    log_info "✓ Environment configuration is valid"
}

create_monitoring_config() {
    log_info "Creating monitoring configuration..."
    
    # Grafana datasource
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Simple dashboard
    mkdir -p monitoring/grafana/dashboards
    cat > monitoring/grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log_info "✓ Monitoring configuration created"
}

build_and_start() {
    log_info "Building and starting services..."
    
    # Build images
    log_info "Building Docker images..."
    if ! make build; then
        log_error "Failed to build Docker images!"
        return 1
    fi
    
    # Start services
    log_info "Starting services..."
    if ! make start; then
        log_error "Failed to start services!"
        return 1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be healthy..."
    make health-wait
    
    log_info "✓ All services are running!"
}

print_usage() {
    log_blue "Setup completed successfully!"
    echo
    log_info "Available commands:"
    echo "  make help              - Show all available commands"
    echo "  make logs              - View application logs"
    echo "  make health            - Check service health"
    echo "  make backup            - Create database backup"
    echo "  make start-monitoring  - Start with Grafana/Prometheus"
    echo
    log_info "Access URLs:"
    echo "  Bot: Running in background (check logs)"
    echo "  Grafana: http://localhost:3000 (admin/admin) - if monitoring enabled"
    echo "  Prometheus: http://localhost:9090 - if monitoring enabled"
    echo
    log_warn "Next steps:"
    echo "1. Check logs: make logs-bot"
    echo "2. Test bot by sending /start to your Telegram bot"
    echo "3. Monitor health: make health"
    echo
    log_info "Configuration files:"
    echo "  .env - Environment variables"
    echo "  docker-compose.yml - Service configuration"
    echo "  Makefile - Management commands"
}

cleanup_on_error() {
    log_error "Setup failed! Cleaning up..."
    make clean-soft 2>/dev/null || true
    exit 1
}

interactive_setup() {
    echo
    read -p "Do you want to configure Telegram token now? (y/N): " configure_token
    if [[ "$configure_token" =~ ^[Yy]$ ]]; then
        read -p "Enter your Telegram bot token: " telegram_token
        sed -i "s/TELEGRAM_TOKEN=\"\"/TELEGRAM_TOKEN=\"$telegram_token\"/" .env
        log_info "✓ Telegram token configured"
    fi
    
    read -p "Do you want to configure API key now? (y/N): " configure_api
    if [[ "$configure_api" =~ ^[Yy]$ ]]; then
        read -p "Enter your API key: " api_key
        sed -i "s/API_KEY=/API_KEY=$api_key/" .env
        log_info "✓ API key configured"
    fi
    
    read -p "Enter PostgreSQL password (or press Enter for auto-generated): " postgres_password
    if [[ -z "$postgres_password" ]]; then
        postgres_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-16)
        log_info "Generated PostgreSQL password: $postgres_password"
    fi
    sed -i "s/POSTGRES_PASSWORD=\"\"/POSTGRES_PASSWORD=\"$postgres_password\"/" .env
    log_info "✓ PostgreSQL password configured"
    
    read -p "Enable monitoring (Grafana/Prometheus)? (y/N): " enable_monitoring
    if [[ "$enable_monitoring" =~ ^[Yy]$ ]]; then
        sed -i "s/ENABLE_MONITORING=false/ENABLE_MONITORING=true/" .env
        log_info "✓ Monitoring enabled"
    fi
}

main() {
    # Trap errors
    trap cleanup_on_error ERR
    
    print_banner
    
    # Check if running with --help
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --skip-build    Skip Docker build step"
        echo "  --interactive   Interactive configuration"
        echo "  --monitoring    Enable monitoring stack"
        echo "  --help         Show this help"
        exit 0
    fi
    
    log_info "Starting Telegram LLM Bot setup..."
    
    # Check requirements
    check_requirements
    
    # Setup environment
    setup_environment
    
    # Interactive configuration if requested
    if [[ "${1:-}" == "--interactive" ]]; then
        interactive_setup
    fi
    
    # Validate configuration
    validate_env
    
    # Create monitoring config
    create_monitoring_config
    
    # Build and start services (unless skipped)
    if [[ "${1:-}" != "--skip-build" ]]; then
        build_and_start
    else
        log_info "Skipping build step as requested"
    fi
    
    # Enable monitoring if requested
    if [[ "${1:-}" == "--monitoring" ]]; then
        log_info "Starting monitoring stack..."
        make start-monitoring
    fi
    
    # Print usage information
    print_usage
    
    log_info "Setup completed successfully! 🚀"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi