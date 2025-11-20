import scrapy
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from scrapy.utils.project import get_project_settings
from scrapy_playwright.page import PageMethod
from MLEngine.items import ImovelItem
from scrapy import signals
import time
import psycopg2 
import sys 

SETTINGS = get_project_settings()

class RemaxSpider(scrapy.Spider):
    name = "remax_imovel"
    allowed_domains = ["remax.pt"]
    start_urls = SETTINGS.getlist('URLS_LIST')

    # --- CONFIGURAÇÕES DE RESILIÊNCIA E ACELERAÇÃO ---
    custom_settings = {
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000, # 60 segundos
        'DOWNLOAD_TIMEOUT': 60,
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 3, 
        'CONCURRENT_REQUESTS': 3,
        'DOWNLOAD_DELAY': 2.5,
    }

    # --- VARIÁVEIS DE CONTROLO E CACHE ---
    existing_listings = {} 

    # ------------------------------
    # 1. CONSTRUTOR (__init__)
    # ------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_processed = 0
        self.pages_processed = 0
        self.start_time = None
        # O self.load_existing_data() é chamado aqui, o que causava o erro.
        # Agora a função está definida abaixo, mas garantimos que está ligada.
        self.load_existing_data() 

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_closed(self, spider):
        self.logger.info(f"🏁 Spider encerrado. Total de imóveis processados: {self.items_processed}")

    # ------------------------------
    # 2. LÓGICA DE CACHE INCREMENTAL (TTL)
    # ------------------------------
    def load_existing_data(self):
        """Carrega ID, Preço e Data da última recolha para o cache (Lógica TTL)."""
        try:
            conn = psycopg2.connect(
                host=SETTINGS.get('PGHOST'), user=SETTINGS.get('PGUSER'), 
                password=SETTINGS.get('PGPASSWORD'), dbname=SETTINGS.get('PGDATABASE'), 
                port=SETTINGS.get('PGPORT')
            )
            cur = conn.cursor()
            cur.execute("SELECT url_id, preco_atual, last_crawled FROM imoveis")
            rows = cur.fetchall()
            for row in rows:
                self.existing_listings[row[0]] = {'price': float(row[1]) if row[1] else 0.0, 'date': row[2]}
            
            self.logger.info(f"♻️ CACHE: Carregados {len(self.existing_listings)} imóveis da BD.")
            cur.close()
            conn.close()
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao carregar cache da BD: {e}")
            
    # ------------------------------
    # FUNÇÕES AUXILIARES DE EXTRAÇÃO
    # ------------------------------
    def clean_num_str(self, raw_str):
        """Função genérica para limpar números e converter para float."""
        if not raw_str: return 0.0
        clean = re.sub(r'[^\d]', '', raw_str)
        return float(clean) if clean else 0.0
        
    def extract_area(self, card):
        area_raw = card.xpath('.//b[contains(text(), "m²")]/text()').get()
        return int(self.clean_num_str(area_raw))

    def extract_price(self, card):
        price_raw = card.xpath('.//span[contains(text(), "€")]/text()').get()
        return self.clean_num_str(price_raw)

    def extract_freguesia(self, card):
        loc = card.css('p.text-ellipsis::text').get()
        return loc.split(',')[0].strip() if loc else "Desconhecido"

    def get_next_page_url(self, current_url):
        """Cálculo do URL da próxima página."""
        parsed_url = urlparse(current_url)
        query_params = parse_qs(parsed_url.query)
        current_page = int(query_params.get('p', [1])[0])
        query_params['p'] = [str(current_page + 1)]
        new_query = urlencode(query_params, doseq=True)
        return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, new_query, parsed_url.fragment))

    # ------------------------------
    # START REQUESTS
    # ------------------------------
    def start_requests(self):
        self.start_time = time.time()
        for url in self.start_urls:
            yield self.make_listing_request(url, page_number=1)

    def make_listing_request(self, url, page_number):
        """Cria o pedido com 'rede de segurança' (errback)."""
        return scrapy.Request(
            url=url,
            callback=self.parse,
            errback=self.errback_pagination, # Se falhar, chama isto
            meta={
                "playwright": True,
                "playwright_context_kwargs": {
                    "ignore_https_errors": True,
                    "viewport": {"width": 1920, "height": 1080}
                },
                "playwright_page_methods": [
                    PageMethod("wait_for_load_state", "networkidle"),
                    PageMethod("wait_for_selector", 'div.grid div[id^="listing-list-card-"]', timeout=60000)
                ],
                "page_number": page_number
            }
        )
        
    def errback_pagination(self, failure):
        """O PLANO B: Se a página falhar (TimeoutError), salta para a próxima!"""
        page_num = failure.request.meta.get('page_number', 1)
        self.logger.error(f"❌ ERRO CRÍTICO na Página {page_num}: {failure.value}")
        
        next_page = page_num + 1
        self.logger.warning(f"⚠️ Recuperação: A saltar para a página {next_page}...")
        
        current_url = failure.request.url
        parsed = urlparse(current_url)
        qs = parse_qs(parsed.query)
        qs['p'] = [str(next_page)]
        new_query = urlencode(qs, doseq=True)
        next_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        
        yield self.make_listing_request(next_url, next_page)

    # ------------------------------
    # PARSE DA LISTA (Com Lógica TTL)
    # ------------------------------
    def parse(self, response):
        page_num = response.meta.get('page_number', 1)
        if response.status == 404:
            self.logger.error(f"404 Not Found: {response.url}")
            return

        cards = response.css('a[data-id="listing-card-link"]')
        num_cards = len(cards)
        self.pages_processed += 1

        # Estimativa de progresso (Log)
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.pages_processed
        pages_left = max(0, 450 - self.pages_processed) 
        eta_min = int((avg_time * pages_left) // 60)
        self.logger.info(f"✅ Pág {page_num} | Items: {num_cards} | ETA: ~{eta_min} min")

        TTL_DAYS = 7 
        now = datetime.now()

        for card in cards:
            link_relativo = card.attrib.get('href')
            if not link_relativo: continue

            full_link = link_relativo if link_relativo.startswith('http') else f"https://remax.pt{link_relativo}"
            id_match = re.search(r'/(\d+-\d+)$', full_link)
            current_id = id_match.group(1) if id_match else None
            
            price_val = self.extract_price(card)
            area_val = self.extract_area(card)
            freguesia_val = self.extract_freguesia(card)

            # --- LÓGICA DE DECISÃO TTL (PULA SE NÃO HOUVE ALTERAÇÃO) ---
            should_scrape = True
            if current_id in self.existing_listings:
                db_data = self.existing_listings[current_id]
                price_changed = db_data['price'] != price_val
                last_date = db_data['date']
                days_since = 999
                
                if last_date:
                    try:
                        if isinstance(last_date, str):
                            last_date = datetime.strptime(last_date.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        days_since = (now.date() - last_date.date()).days
                    except ValueError:
                        pass
                
                is_stale = days_since >= TTL_DAYS

                if not price_changed and not is_stale:
                    should_scrape = False
            
            if not should_scrape:
                continue # Pula a visita ao detalhe

            yield response.follow(
                full_link,
                callback=self.parse_remax_imovel,
                cb_kwargs={
                    'area': area_val, 'price': price_val, 
                    'freguesia': freguesia_val, 'link_completo': full_link,
                    'page_number': page_num # PASSADO PARA O LOG
                }
            )

        # PAGINAÇÃO
        next_button = response.css('button[aria-label="Go to next page"]')
        if next_button and 'Mui-disabled' not in next_button.attrib.get('class', ''):
            next_page = page_num + 1
            next_url = self.get_next_page_url(response.url)
            self.logger.info(f"➡️ A avançar para página {next_page}...")
            yield self.make_listing_request(next_url, next_page)
        else:
            self.logger.info("🏁 Última página atingida.")

    # ------------------------------
    # PARSE DO DETALHE (Extração de todos os campos brutos)
    # ------------------------------
    def parse_remax_imovel(self, response, area, price, freguesia, link_completo, page_number):
        item = ImovelItem()
        item['preco_atual'] = price
        item['area_bruta_m2'] = area
        item['freguesia'] = freguesia
        item['link'] = link_completo

        def clean_num(text):
            if not text: return 0
            clean = re.sub(r'[^\d]', '', text)
            return int(clean) if clean else 0

        def get_detail(label):
            return response.xpath(f"//span[contains(text(), '{label}')]/following-sibling::span/text()").get()

        # EXTRAÇÃO DA DESCRIÇÃO COMPLETA (CRÍTICO PARA O FILTRO TIMESHARE)
        desc_list = response.css('#description .custom-description *::text').getall()
        item['descricao_bruta'] = " ".join(desc_list).strip()

        # Áreas Detalhadas e Priorização
        area_priv = get_detail("Área Bruta Privativa")
        area_bruta = get_detail("Área Bruta")
        item['area_terreno_m2'] = clean_num(get_detail("Área Total do Lote"))
        item['area_util_m2'] = clean_num(get_detail("Área Útil"))
        
        if area_priv: item['area_bruta_m2'] = clean_num(area_priv)
        elif area_bruta: item['area_bruta_m2'] = clean_num(area_bruta)

        # Detalhes Técnicos
        item['ano_construcao'] = clean_num(get_detail("Ano de Construção"))
        item['num_quartos'] = clean_num(get_detail("Quartos"))
        item['num_wc'] = clean_num(get_detail("WC") or get_detail("Casas de banho"))
        item['estacionamento'] = get_detail("Estacionamento")
        item['elevador'] = get_detail("Elevador")
        
        # Tipologia e Certificado Energético
        page_title = response.css('title::text').get()
        item['tipologia'] = 'Desconhecida'
        if page_title:
            m = re.search(r'([A-Za-z]+)\s+(T\d+)', page_title)
            if m: item['tipologia'] = f"{m.group(1)} {m.group(2)}".strip()
            else: 
                tm = re.search(r'Venda-\s*([A-Za-z]+)', page_title)
                if tm: item['tipologia'] = tm.group(1).strip()

        item['certificado_energetico'] = response.xpath("//*[contains(text(), 'Eficiência energética')]/following-sibling::span//img/@alt").get()

        # DADOS DE CONTROLO
        id_match = re.search(r'/(\d+-\d+)$', response.url)
        item['url_id'] = id_match.group(1) if id_match else response.url
        item['data_publicacao'] = str(datetime.now().date())
        item['last_crawled'] = str(datetime.now())
        item['listing_page_number'] = page_number # ADICIONADO PARA O LOG
        
        self.items_processed += 1
        yield item