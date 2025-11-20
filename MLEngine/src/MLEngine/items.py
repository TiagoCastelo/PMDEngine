import scrapy

class ImovelItem(scrapy.Item):
    # Identificação e Link
    url_id = scrapy.Field()
    link = scrapy.Field()           # <--- NOVO CAMPO
    last_crawled = scrapy.Field()
    data_publicacao = scrapy.Field()
    listing_page_number = scrapy.Field() # <--- NOVO CAMPO
    # Dados Principais
    preco_atual = scrapy.Field()
    freguesia = scrapy.Field()
    tipologia = scrapy.Field()
    
    # Áreas
    area_bruta_m2 = scrapy.Field()
    area_util_m2 = scrapy.Field()
    area_terreno_m2 = scrapy.Field()
    
    # Detalhes Específicos
    ano_construcao = scrapy.Field()
    num_quartos = scrapy.Field()
    num_wc = scrapy.Field()
    estacionamento = scrapy.Field()
    elevador = scrapy.Field()
    
    # Energia
    certificado_energetico = scrapy.Field()
    # Nova Coluna para Filtro de Qualidade
    descricao_bruta = scrapy.Field() # <-- Campo para o texto longo do anúncio
    