#region imports

from sqlalchemy import Column, ForeignKey, Integer, String
from BusinessObjects.BaseEntity import BaseEntity
from Repository.SqlAlchemySetup import SqlAlchemySetup

#endregion imports

#region Model def: start
class BookDto(BaseEntity, SqlAlchemySetup.Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    genre = Column(String(255))
    year_published = Column(Integer)
    summary = Column(String(1000))

class ReviewDto(BaseEntity, SqlAlchemySetup.Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    review_text = Column(String)
    rating = Column(Integer)

class UserDto(BaseEntity, SqlAlchemySetup.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)  # Store hashed password
    
#endregion Model def: End
