from typing import Union
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None) -> Dict[str,Union[str, None]]:
    return {"item_id": item_id, "q": q}