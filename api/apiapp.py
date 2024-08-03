from fastapi import FastAPI

fastapiapp = FastAPI()
@fastapiapp.get("/")
async def index():
   return {"message": "Hello World"}