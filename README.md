# Telegram AI Bot с PostgreSQL

Многопользовательский Telegram-бот с поддержкой множественных LLM, thinking режима, веб-поиска и обработки файлов.

## Основные возможности

- **Многопользовательская архитектура** с изоляцией данных
- **AES-256 шифрование** API-ключей с ротацией ключей
- **Thinking режим** для сложных задач с автоматическим определением
- **Веб-поиск** с интеграцией DuckDuckGo
- **Обработка файлов**: PDF, изображения, текстовые документы
- **Rate limiting** и защита от злоупотреблений
- **PostgreSQL** с connection pooling и оптимизацией
- **Redis** для кэширования и сессий
- **Мониторинг** через Prometheus и Grafana
- **Автоматические бэкапы** с шифрованием

## Быстрый старт

### 1. Клонирование и настройка

```fish
# Клонируем репозиторий
gh repo clone your-username/telegram-llm-bot
cd telegram-llm-bot

# Копируем и настраиваем окружение
cp .env.example .env
# Отредактируйте .env файл с вашими токенами
```

### 2. Настройка переменных окружения

Отредактируйте `.env`:

```bash
TELEGRAM_TOKEN="ваш_токен_бота"
API_KEY=ваш_api_ключ_openrouter
POSTGRES_PASSWORD=безопасный_пароль
```

### 3. Запуск

```fish
# Сборка и запуск всех сервисов
make build
make start

# Или одной командой
make build start

# Проверка статуса
make health
```

## Управление через Makefile

### Основные команды

```fish
make start          # Запустить все сервисы
make stop           # Остановить сервисы
make restart        # Перезапустить
make logs           # Посмотреть логи
make logs-bot       # Логи только бота
make health         # Проверить статус
```

### Мониторинг

```fish
make start-monitoring  # Запуск с Grafana/Prometheus
make monitor          # Открыть дашборды
```

### Базы данных

```fish
make backup          # Создать бэкап
make restore         # Восстановить из бэкапа
make shell-db        # PostgreSQL shell
make shell-redis     # Redis shell
```

### Разработка

```fish
make dev            # Режим разработки
make test           # Запустить тесты  
make lint           # Проверить код
make format         # Форматировать код
make security-check # Проверка безопасности
```

## Архитектура

### Компоненты

1. **Bot Container**: Основное приложение на Python
2. **PostgreSQL**: База данных с партицированием
3. **Redis**: Кэш и rate limiting
4. **Prometheus**: Сбор метрик
5. **Grafana**: Визуализация метрик

### База данных

- Партицированная таблица сообщений
- Индексы для быстрых запросов
- Автоматическая очистка старых данных
- Triggers для обновления статистики

### Безопасность

- AES-256 шифрование с ротацией ключей каждые 24 часа
- Rate limiting: 20 запросов в минуту
- Валидация загружаемых файлов
- Изоляция контейнеров через отдельную сеть
- Non-root пользователь в контейнерах

## Конфигурация

### Настройки LLM

```python
# В .env
DEFAULT_MODEL=mistralai/mistral-7b-instruct:free
DEFAULT_REASONING_MODEL=deepseek/deepseek-r1-distill-llama-70b
DEFAULT_ROUTER_MODEL=anthropic/claude-3.5-sonnet:beta
```

### Лимиты и производительность

```python
# База данных
DB_POOL_MIN=5
DB_POOL_MAX=20

# Rate limiting
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60

# Файлы
MAX_FILE_SIZE=52428800  # 50MB
```

## Мониторинг

### Grafana Dashboards

- **Bot Metrics**: Количество пользователей, сообщений, ошибок
- **Database Performance**: Запросы, соединения, размер БД
- **System Resources**: CPU, память, диск

### Prometheus Metrics

- `bot_users_total` - общее количество пользователей
- `bot_messages_total` - счетчик сообщений
- `bot_requests_duration` - время обработки запросов
- `bot_errors_total` - счетчик ошибок по типам

### Доступ к мониторингу

```fish
# Запуск с мониторингом
make start-monitoring

# URLs:
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Резервное копирование

### Автоматические бэкапы

Настроено через cron в контейнере:
- Ежедневно в 2:00 UTC
- Сжатие gzip
- Опциональное шифрование GPG
- Ротация: хранение 30 дней

### Ручные бэкапы

```fish
# Создать бэкап
make backup

# Восстановить последний бэкап
make restore

# Восстановить конкретный бэкап
BACKUP_FILE=backup_tgllm_20241201_020000.sql.gz make restore
```

## Обработка файлов

### Поддерживаемые форматы

- **PDF**: Извлечение текста через pdfplumber и PyPDF2
- **Изображения**: JPG, PNG, WebP (готовность к Vision API)
- **Текст**: TXT, Markdown

### Безопасность файлов

- Проверка размера (максимум 50MB)
- Валидация расширений
- Блокировка опасных типов файлов
- Автоматическая очистка через 24 часа

## Команды бота

### Пользовательские команды

- `/start` - Регистрация и настройка
- `/settings` - Управление настройками
- `/help` - Справка

### Функции

- **Thinking режим**: Автоматическое определение сложных задач
- **Веб-поиск**: Поиск актуальной информации
- **Контекст файлов**: Обработка загруженных документов
- **History управление**: Настраиваемый размер контекста

## Логирование

### Уровни логирования

```python
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Файлы логов

- `bot.log` - основные логи бота
- `error.log` - ошибки и исключения
- Ротация: 10MB файлы, 5 архивов

### Просмотр логов

```fish
make logs           # Все сервисы
make logs-bot       # Только бот
make logs-db        # База данных

# Логи в реальном времени
docker-compose -f docker-compose.yml logs -f --tail=100 bot
```

## Разработка

### Структура проекта

```
├── main.py                 # Основной код бота
├── requirements.txt        # Python зависимости
├── Dockerfile             # Контейнер бота
├── docker-compose.yml     # Оркестрация сервисов
├── init.sql              # Инициализация БД
├── Makefile              # Автоматизация
├── monitoring/           # Конфигурация мониторинга
├── scripts/             # Скрипты администрирования
└── backups/            # Резервные копии
```

### Добавление новых функций

1. Создайте новый handler в `main.py`
2. Добавьте соответствующие SQL-запросы в `init.sql`
3. Обновите тесты в `tests/`
4. Добавьте метрики в Prometheus

### Тестирование

```fish
# Запуск тестов
make test

# С покрытием кода
docker-compose exec bot python -m pytest --cov=main tests/

# Только unit тесты
docker-compose exec bot python -m pytest tests/unit/
```

## Производительность

### Оптимизация PostgreSQL

- Настроенные параметры для быстродействия
- Партицирование таблицы сообщений
- Индексы для частых запросов
- Connection pooling

### Кэширование

- Redis для сессий пользователей
- Кэш результатов веб-поиска
- Rate limiting в памяти

### Scaling

```fish
# Запуск нескольких экземпляров бота
docker-compose up --scale bot=3
```

## Безопасность

### Проверки безопасности

```fish
make security-check  # Поиск уязвимостей
make env-check      # Проверка конфигурации
```

### Рекомендации

1. Используйте сильные пароли в `.env`
2. Регулярно обновляйте зависимости
3. Мониторьте логи на предмет атак
4. Настройте файрволл на сервере
5. Используйте HTTPS для webhook (если нужен)

## Troubleshooting

### Частые проблемы

**База данных не запускается**
```fish
make clean-soft  # Мягкая очистка
make start       # Перезапуск
```

**Бот не отвечает**
```fish
make logs-bot           # Проверить логи
make health            # Проверить статус
docker-compose restart bot  # Перезапуск бота
```

**Проблемы с памятью**
```fish
make stats             # Статистика контейнеров
make disk-usage        # Использование диска
```

### Отладка

```fish
# Режим отладки
make debug

# Shell в контейнере
make shell-bot

# Проверка подключения к БД
make shell-db
```

## Лицензия

MIT License - см. файл LICENSE

## Поддержка

Для вопросов и отчетов об ошибках создавайте issues в GitHub репозитории.