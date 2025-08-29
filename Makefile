# Telegram Bot Management Makefile
.DEFAULT_GOAL := help
.PHONY: help build start stop restart logs clean backup restore test lint format security-check health

# Environment variables
COMPOSE_FILE ?= docker-compose.yml
PROJECT_NAME ?= tg-llm-bot
BACKUP_FILE ?= latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

help: ## Show this help message
	@echo "$(GREEN)Telegram Bot Management Commands$(NC)"
	@echo "================================"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(BLUE)<target>$(NC)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: ## Build Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build --no-cache
	@echo "$(GREEN)Build completed!$(NC)"

start: ## Start all services
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@make health-wait

stop: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) stop
	@echo "$(YELLOW)Services stopped!$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) restart
	@echo "$(GREEN)Services restarted!$(NC)"

start-monitoring: ## Start services with monitoring
	@echo "$(GREEN)Starting services with monitoring...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) --profile monitoring up -d
	@echo "$(GREEN)Services with monitoring started!$(NC)"
	@echo "$(BLUE)Grafana:$(NC) http://localhost:3000 (admin/admin)"
	@echo "$(BLUE)Prometheus:$(NC) http://localhost:9090"

logs: ## Show logs for all services
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f

logs-bot: ## Show bot logs only
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f bot

logs-db: ## Show database logs only
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f db

clean: ## Stop and remove all containers, networks, volumes
	@echo "$(RED)Cleaning up all resources...$(NC)"
	@read -p "Are you sure? This will delete all data! (y/N) " confirm && [ "$$confirm" = "y" ]
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down -v --remove-orphans
	docker system prune -f
	@echo "$(RED)Cleanup completed!$(NC)"

clean-soft: ## Stop and remove containers but keep volumes
	@echo "$(YELLOW)Soft cleanup (keeping data)...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down --remove-orphans
	@echo "$(YELLOW)Soft cleanup completed!$(NC)"

backup: ## Create database backup
	@echo "$(GREEN)Creating database backup...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) --profile backup up backup
	@echo "$(GREEN)Backup completed! Check ./backups/ directory$(NC)"

restore: ## Restore database from backup
	@echo "$(RED)Restoring database from backup: $(BACKUP_FILE)$(NC)"
	@read -p "This will overwrite the current database! Continue? (y/N) " confirm && [ "$$confirm" = "y" ]
	@if [ ! -f "./backups/$(BACKUP_FILE)" ]; then \
		echo "$(RED)Backup file not found: ./backups/$(BACKUP_FILE)$(NC)"; \
		exit 1; \
	fi
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec -T db psql -U $(POSTGRES_USER) -d postgres -c "DROP DATABASE IF EXISTS $(POSTGRES_DB);"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec -T db psql -U $(POSTGRES_USER) -d postgres -c "CREATE DATABASE $(POSTGRES_DB);"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec -T db psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < ./backups/$(BACKUP_FILE)
	@echo "$(GREEN)Database restored from $(BACKUP_FILE)!$(NC)"

health: ## Check health status of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps
	@echo "\n$(BLUE)Database connection test:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec db pg_isready -U $(POSTGRES_USER) -d $(POSTGRES_DB) || echo "$(RED)Database not ready$(NC)"
	@echo "$(BLUE)Redis connection test:$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec redis redis-cli ping || echo "$(RED)Redis not ready$(NC)"

health-wait: ## Wait for services to be healthy
	@echo "$(BLUE)Waiting for services to be healthy...$(NC)"
	@timeout=60; \
	while [ $$timeout -gt 0 ]; do \
		if docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec db pg_isready -U $(POSTGRES_USER) -d $(POSTGRES_DB) >/dev/null 2>&1; then \
			echo "$(GREEN)Database is ready!$(NC)"; \
			break; \
		fi; \
		echo "Waiting for database... ($$timeout seconds left)"; \
		sleep 2; \
		timeout=$$((timeout-2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "$(RED)Timeout waiting for database!$(NC)"; \
		exit 1; \
	fi

shell-bot: ## Open shell in bot container
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot bash

shell-db: ## Open PostgreSQL shell
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec db psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

shell-redis: ## Open Redis shell
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec redis redis-cli

update: ## Update bot code and restart
	@echo "$(BLUE)Updating bot...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build bot --no-cache
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) restart bot
	@echo "$(GREEN)Bot updated and restarted!$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -m pytest tests/ -v
	@echo "$(GREEN)Tests completed!$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -m flake8 main.py
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -m black --check main.py
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -m black main.py
	@echo "$(GREEN)Code formatted!$(NC)"

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@# Check for common security issues
	@grep -r "password.*=" . --exclude-dir=.git || echo "No hardcoded passwords found"
	@grep -r "api_key.*=" . --exclude-dir=.git || echo "No hardcoded API keys found"
	@grep -r "secret.*=" . --exclude-dir=.git || echo "No hardcoded secrets found"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -m pip check
	@echo "$(GREEN)Security check completed!$(NC)"

stats: ## Show container stats
	@echo "$(BLUE)Container statistics:$(NC)"
	docker stats --no-stream $(shell docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps -q)

disk-usage: ## Show disk usage
	@echo "$(BLUE)Docker disk usage:$(NC)"
	docker system df
	@echo "\n$(BLUE)Volume usage:$(NC)"
	docker volume ls -q | xargs docker volume inspect | grep -E "(Name|Mountpoint)" | paste - -

env-check: ## Check environment configuration
	@echo "$(BLUE)Environment Configuration Check:$(NC)"
	@echo "================================"
	@if [ -f .env ]; then \
		echo "$(GREEN)✓ .env file found$(NC)"; \
		echo "Required variables:"; \
		for var in TELEGRAM_TOKEN POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD; do \
			if grep -q "^$$var=" .env; then \
				echo "$(GREEN)✓ $$var is set$(NC)"; \
			else \
				echo "$(RED)✗ $$var is missing$(NC)"; \
			fi; \
		done; \
	else \
		echo "$(RED)✗ .env file not found!$(NC)"; \
		echo "$(YELLOW)Copy .env.example to .env and fill in your values$(NC)"; \
	fi

monitor: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring services...$(NC)"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 || echo "Open http://localhost:3000 in your browser"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:9090 || echo "Open http://localhost:9090 in your browser"

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -f docker-compose.dev.yml -p $(PROJECT_NAME) up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)Hot reload enabled$(NC)"

prod: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	@make env-check
	@make security-check
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d --scale bot=2
	@echo "$(GREEN)Production environment started with load balancing!$(NC)"

debug: ## Start with debug logs
	@echo "$(BLUE)Starting with debug logging...$(NC)"
	LOG_LEVEL=DEBUG docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up bot

benchmark: ## Run performance benchmark
	@echo "$(BLUE)Running performance benchmark...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot python -c "
	import time
	import asyncio
	import psycopg2
	
	# Database connection test
	start = time.time()
	for i in range(100):
		conn = psycopg2.connect(
			host='db', database='$(POSTGRES_DB)', 
			user='$(POSTGRES_USER)', password='$(POSTGRES_PASSWORD)'
		)
		conn.close()
	print(f'DB connections: {100/(time.time()-start):.2f} conn/sec')
	"

# Development helpers
install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

clean-logs: ## Clean up old log files
	@echo "$(YELLOW)Cleaning up log files...$(NC)"
	find ./logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec bot find /app/logs -name "*.log" -mtime +7 -delete || true
	@echo "$(GREEN)Log cleanup completed!$(NC)"