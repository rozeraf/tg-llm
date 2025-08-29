#!/bin/bash

# PostgreSQL Backup Script with encryption and rotation
# Usage: ./backup.sh [database_name]

set -euo pipefail

# Configuration
BACKUP_DIR="/backups"
DB_HOST="${POSTGRES_HOST:-db}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${1:-${POSTGRES_DB:-tgllm}}"
DB_USER="${POSTGRES_USER:-tgllm}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
COMPRESS_BACKUPS="${COMPRESS_BACKUPS:-true}"
ENCRYPT_BACKUPS="${ENCRYPT_BACKUPS:-false}"
ENCRYPTION_PASSWORD="${BACKUP_ENCRYPTION_PASSWORD:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Generate backup filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql"

log_info "Starting backup of database: ${DB_NAME}"
log_info "Backup file: ${BACKUP_FILE}"

# Perform database backup
if ! pg_dump \
    --host="${DB_HOST}" \
    --port="${DB_PORT}" \
    --username="${DB_USER}" \
    --dbname="${DB_NAME}" \
    --verbose \
    --clean \
    --if-exists \
    --create \
    --format=plain \
    --no-password \
    > "${BACKUP_FILE}"; then
    log_error "Database backup failed!"
    exit 1
fi

# Check if backup file was created and has content
if [[ ! -s "${BACKUP_FILE}" ]]; then
    log_error "Backup file is empty or was not created!"
    exit 1
fi

BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
log_info "Backup completed successfully. Size: ${BACKUP_SIZE}"

# Compress backup if enabled
if [[ "${COMPRESS_BACKUPS}" == "true" ]]; then
    log_info "Compressing backup..."
    if gzip "${BACKUP_FILE}"; then
        BACKUP_FILE="${BACKUP_FILE}.gz"
        COMPRESSED_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
        log_info "Compression completed. New size: ${COMPRESSED_SIZE}"
    else
        log_warn "Compression failed, keeping uncompressed backup"
    fi
fi

# Encrypt backup if enabled
if [[ "${ENCRYPT_BACKUPS}" == "true" && -n "${ENCRYPTION_PASSWORD}" ]]; then
    log_info "Encrypting backup..."
    if command -v gpg >/dev/null 2>&1; then
        if gpg --batch --yes --passphrase "${ENCRYPTION_PASSWORD}" --cipher-algo AES256 --compress-algo 1 --symmetric --output "${BACKUP_FILE}.gpg" "${BACKUP_FILE}"; then
            rm "${BACKUP_FILE}"
            BACKUP_FILE="${BACKUP_FILE}.gpg"
            log_info "Encryption completed"
        else
            log_warn "Encryption failed, keeping unencrypted backup"
        fi
    else
        log_warn "GPG not available, skipping encryption"
    fi
fi

# Create backup metadata
METADATA_FILE="${BACKUP_FILE}.meta"
cat > "${METADATA_FILE}" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "database": "${DB_NAME}",
    "host": "${DB_HOST}",
    "port": "${DB_PORT}",
    "user": "${DB_USER}",
    "backup_file": "$(basename "${BACKUP_FILE}")",
    "original_size": "${BACKUP_SIZE}",
    "compressed": ${COMPRESS_BACKUPS},
    "encrypted": ${ENCRYPT_BACKUPS},
    "created_at": "$(date -Iseconds)"
}
EOF

# Create symbolic link to latest backup
LATEST_LINK="${BACKUP_DIR}/latest_${DB_NAME}.sql"
[[ "${COMPRESS_BACKUPS}" == "true" ]] && LATEST_LINK="${LATEST_LINK}.gz"
[[ "${ENCRYPT_BACKUPS}" == "true" && -n "${ENCRYPTION_PASSWORD}" ]] && LATEST_LINK="${LATEST_LINK}.gpg"

rm -f "${LATEST_LINK}"
ln -s "$(basename "${BACKUP_FILE}")" "${LATEST_LINK}"

# Clean up old backups
log_info "Cleaning up backups older than ${BACKUP_RETENTION_DAYS} days..."
DELETED_COUNT=0

find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql*" -type f -mtime +${BACKUP_RETENTION_DAYS} | while read -r old_backup; do
    log_info "Removing old backup: $(basename "${old_backup}")"
    rm -f "${old_backup}" "${old_backup}.meta"
    ((DELETED_COUNT++)) || true
done

# Verify backup integrity
log_info "Verifying backup integrity..."
BACKUP_TO_VERIFY="${BACKUP_FILE}"

# Handle compressed/encrypted files for verification
if [[ "${BACKUP_FILE}" == *.gz ]]; then
    BACKUP_TO_VERIFY="${BACKUP_FILE%.gz}"
    gunzip -c "${BACKUP_FILE}" > "${BACKUP_TO_VERIFY}.tmp"
    BACKUP_TO_VERIFY="${BACKUP_TO_VERIFY}.tmp"
elif [[ "${BACKUP_FILE}" == *.gpg ]]; then
    BACKUP_TO_VERIFY="${BACKUP_FILE%.gpg}"
    if gpg --batch --yes --passphrase "${ENCRYPTION_PASSWORD}" --decrypt "${BACKUP_FILE}" > "${BACKUP_TO_VERIFY}.tmp" 2>/dev/null; then
        BACKUP_TO_VERIFY="${BACKUP_TO_VERIFY}.tmp"
    else
        log_warn "Cannot verify encrypted backup without decryption"
        BACKUP_TO_VERIFY=""
    fi
fi

if [[ -n "${BACKUP_TO_VERIFY}" && -f "${BACKUP_TO_VERIFY}" ]]; then
    if grep -q "PostgreSQL database dump" "${BACKUP_TO_VERIFY}" 2>/dev/null; then
        log_info "Backup verification passed"
    else
        log_warn "Backup verification failed - file may be corrupted"
    fi
    
    # Clean up temporary verification file
    [[ "${BACKUP_TO_VERIFY}" == *.tmp ]] && rm -f "${BACKUP_TO_VERIFY}"
fi

# Generate backup report
REPORT_FILE="${BACKUP_DIR}/backup_report_$(date +%Y%m).log"
cat >> "${REPORT_FILE}" << EOF
$(date -Iseconds): Backup completed
  Database: ${DB_NAME}
  File: $(basename "${BACKUP_FILE}")
  Size: ${BACKUP_SIZE}
  Status: SUCCESS
  Duration: $((SECONDS))s

EOF

log_info "Backup process completed successfully!"
log_info "Backup saved as: $(basename "${BACKUP_FILE}")"
log_info "Total time: $((SECONDS)) seconds"

# Optional: Send notification (if notification service is configured)
if command -v curl >/dev/null 2>&1 && [[ -n "${BACKUP_WEBHOOK_URL:-}" ]]; then
    curl -X POST "${BACKUP_WEBHOOK_URL}" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"Database backup completed successfully\",
            \"database\": \"${DB_NAME}\",
            \"file\": \"$(basename "${BACKUP_FILE}")\",
            \"size\": \"${BACKUP_SIZE}\",
            \"timestamp\": \"$(date -Iseconds)\"
        }" \
        2>/dev/null || log_warn "Failed to send backup notification"
fi

exit 0