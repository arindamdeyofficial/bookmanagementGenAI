#region imports
import asyncio
from Repository.SqlAlchemySetup import SqlAlchemySetup
from apiapp import fastapiapp
from Controllers import BookController, UserController

#endregion imports
app = fastapiapp
async def main():
    sqlalchemy_setup = SqlAlchemySetup()
    await sqlalchemy_setup.create_async_tables()

if __name__ == "__main__":
    asyncio.run(main())
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8020, reload=True)
