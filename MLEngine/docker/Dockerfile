FROM python:3.11-slim

# Set non-interactive mode for Debian package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Instalar dependências de sistema (Camada Pesada - Fica em Cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libegl1 \
    libgl1 \
    libdbus-1-3 \
    ca-certificates \
    fonts-liberation \
    libappindicator3-1 \
    libasound2-plugins \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 🚨 AQUI ESTÁ O TRUQUE DE OTIMIZAÇÃO
# -----------------------------------------------------------

# 2. Copia APENAS o ficheiro de requisitos primeiro
COPY ../src/requirements.txt /app/requirements.txt

# 3. Instala dependências Python e Browser (Ficam em Cache se o requirements.txt não mudar)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scrapy-playwright
RUN playwright install chromium

# 4. SÓ AGORA copia o resto do código. 
# Se mudares o código do spider, o Docker SÓ executa daqui para baixo (super rápido e leve).
COPY ../src /app

# Env & Cmd
ENV PYTHONPATH=/app
CMD ["scrapy", "crawl", "remax_imovel"]