from datetime import datetime
from typing import Optional
from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class BaseEntity:
    created_by = Column(String(255), nullable=False, default='system')
    created_on = Column(DateTime, nullable=False, default= datetime.now())
    modified_by = Column(String(255), nullable=False, default='system')
    modified_on = Column(DateTime, nullable=False, default= datetime.now())