import logging
import uuid

class RequestIdFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.request_id = None

    def filter(self, record):
        # Si no se ha seteado, asigna un UUID genérico
        record.request_id = self.request_id or "no-request-id"
        return True

# Configurar logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# Crear handler y formateador personalizado
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s {- %(request_id)s} - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Añadir filtro para request_id
request_id_filter = RequestIdFilter()
logger.addFilter(request_id_filter)
