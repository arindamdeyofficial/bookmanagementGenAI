from fastapi import HTTPException, status
from sqlalchemy import CheckConstraint, create_engine, Column, Integer, String, ForeignKey, func, or_, update, orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload, scoped_session, DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError, OperationalError

class SqlAlchemySetup:
    #region SQLAlchemy setup
    DATABASE_URL = "postgresql+asyncpg://postgres:nakshal01051987@localhost:5432/bookreviewmgmt"
    async_engine = create_async_engine(DATABASE_URL, connect_args={"command_timeout": 28.0})
    async_session_maker = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine, expire_on_commit=False)
    Base = declarative_base()

    async def create_async_tables(self):
        async with self.async_engine.begin() as conn:
            try:
                await conn.run_sync(self.Base.metadata.create_all)
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    #endregion SQLAlchemy setup