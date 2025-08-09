from pydantic import BaseModel

class TextInput(BaseModel):
    texto: str

class SentimentResponse(BaseModel):
    texto: str
    positividad: float
    negatividad: float
    neutralidad: float
    rating: float
    fecha: str
    length: int
