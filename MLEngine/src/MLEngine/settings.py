# -*- coding: utf-8 -*-

import os
import json

# =========================================================================
# CONFIGURAÇÕES BÁSICAS DO PROJETO
# =========================================================================

BOT_NAME = "MLEngine"
SPIDER_MODULES = ["MLEngine.spiders"]
NEWSPIDER_MODULE = "MLEngine.spiders"
ROBOTSTXT_OBEY = False

# =========================================================================
# CONFIGURAÇÃO DE AMBIENTE DOCKER/ENV
# =========================================================================

START_URLS_RAW = os.environ.get('START_URLS_LIST')
URLS_LIST = []

if START_URLS_RAW:
    try:
        URLS_LIST = json.loads(START_URLS_RAW)
    except json.JSONDecodeError:
        URLS_LIST = []

CUSTOM_SETTINGS = {
    'URLS_LIST': URLS_LIST,
}

# Credenciais PostgreSQL
PGDATABASE = os.environ.get('PGDATABASE')
PGUSER = os.environ.get('PGUSER')
PGPASSWORD = os.environ.get('PGPASSWORD')
PGHOST = os.environ.get('PGHOST')
PGPORT = os.environ.get('PGPORT')

# =========================================================================
# CONFIGURAÇÕES DE CRAWLING
# =========================================================================

DOWNLOAD_DELAY = 2.5            # 2.5s entre requests
CONCURRENT_REQUESTS = 3          # Até 3 requests simultâneas
COOKIES_ENABLED = False 

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' 

DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-PT,pt;q=0.8,en-US;q=0.5,en;q=0.3', 
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.google.com/', 
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
}

HTTPERROR_ALLOWED_CODES = [403, 404]  # Mantém 404 para log

LOG_LEVEL = 'INFO'
# LOG_LEVEL = 'WARNING'  # Apenas warnings e erros serão mostrados em produção
# LOG_LEVEL = 'DEBUG'  # Para debug detalhado

# =========================================================================
# PLAYWRIGHT CONFIGURATION
# =========================================================================

DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': 500,
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': None,
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 15000  # Timeout reduzido para 15s

ITEM_PIPELINES = {
    'MLEngine.pipelines.PostgresPipeline': 300,
}

# =========================================================================
# LOG & MONITORAMENTO DE ITEMS
# =========================================================================

# Ativa contagem de items processados
# Pode ser usado no spider com signals
# from scrapy import signals
# self.crawler.stats.inc_value('items_scraped_count')
