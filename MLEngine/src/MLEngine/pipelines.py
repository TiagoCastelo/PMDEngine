import psycopg2
from scrapy.utils.project import get_project_settings

class PostgresPipeline:
    def __init__(self):
        settings = get_project_settings()
        self.connection = psycopg2.connect(
            host=settings.get('PGHOST'),
            user=settings.get('PGUSER'),
            password=settings.get('PGPASSWORD'),
            dbname=settings.get('PGDATABASE'),
            port=settings.get('PGPORT')
        )
        self.cursor = self.connection.cursor()
        self.success_count = 0
        self.fail_count = 0

        # 1. CRIAR TABELA (Inclui descricao_bruta)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS imoveis (
                url_id VARCHAR PRIMARY KEY,
                link TEXT,
                last_crawled TIMESTAMP,
                data_publicacao DATE,
                
                preco_atual FLOAT,
                freguesia VARCHAR,
                tipologia VARCHAR,
                
                area_bruta_m2 FLOAT,
                area_util_m2 FLOAT,
                area_terreno_m2 FLOAT,
                
                ano_construcao INTEGER,
                num_quartos INTEGER,
                num_wc INTEGER,
                estacionamento VARCHAR,
                elevador VARCHAR,
                certificado_energetico VARCHAR,
                descricao_bruta TEXT
            );
        """)
        self.connection.commit()

    def process_item(self, item, spider):
        page_num = item.get('listing_page_number', 'N/A') # <--- LÊ O NÚMERO DA PÁGINA AQUI
        
        try:
            # 2. UPSERT (INSERT/UPDATE)
            self.cursor.execute("""
                INSERT INTO imoveis (
                    url_id, link, last_crawled, data_publicacao,
                    preco_atual, freguesia, tipologia,
                    area_bruta_m2, area_util_m2, area_terreno_m2,
                    ano_construcao, num_quartos, num_wc, 
                    estacionamento, elevador, certificado_energetico,
                    descricao_bruta
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url_id) DO UPDATE SET
                    link = EXCLUDED.link,
                    last_crawled = EXCLUDED.last_crawled,
                    -- ... (outros campos) ...
                    descricao_bruta = EXCLUDED.descricao_bruta;
            """, (
                item.get('url_id'), item.get('link'), item.get('last_crawled'), item.get('data_publicacao'),
                item.get('preco_atual'), item.get('freguesia'), item.get('tipologia'),
                item.get('area_bruta_m2'), item.get('area_util_m2'), item.get('area_terreno_m2'),
                item.get('ano_construcao'), item.get('num_quartos'), item.get('num_wc'),
                item.get('estacionamento'), item.get('elevador'), item.get('certificado_energetico'),
                item.get('descricao_bruta')
            ))
            self.connection.commit()
            self.success_count += 1
            
            # 🚨 LOG ATUALIZADO COM O NÚMERO DA PÁGINA
            spider.logger.info(f"[PIPELINE] Inserido/Atualizado: {item.get('url_id')} (Pág {page_num}) | Total inseridos: {self.success_count}")

        except Exception as e:
            self.connection.rollback()
            self.fail_count += 1
            spider.logger.error(f"[PIPELINE-ERROR] Falha ao salvar {item.get('url_id')} | Erro: {e} | Total falhas: {self.fail_count}")
            
        return item

    def close_spider(self, spider):
        self.cursor.close()
        self.connection.close()
        spider.logger.info(f"[PIPELINE] Conexão encerrada. Inseridos: {self.success_count}, Falhas: {self.fail_count}")