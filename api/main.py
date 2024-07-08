import asyncio
import json
from sqlite3 import IntegrityError
from typing import Union
from fastapi import Depends, status, FastAPI, Body, Path, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, constr, ValidationError
import spacy
from sqlalchemy import CheckConstraint, create_engine, Column, Integer, String, ForeignKey, func, or_, update, orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload, scoped_session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from automapper import mapper

# Database connection string (replace with your details)
DATABASE_URL = "postgresql+asyncpg://postgres:nakshal01051987@localhost:5432/bookreviewmgmt"

# SQLAlchemy setup
async_engine = create_async_engine(DATABASE_URL, connect_args={"command_timeout": 28.0})
async_session_maker = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine, expire_on_commit=False)
Base = declarative_base()

async def create_async_tables():
    async with async_engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

#region Model def: start
class BookDto(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    genre = Column(String(255))
    year_published = Column(Integer)
    summary = Column(String(1000))

class Book(BaseModel):
    id: int  # Assuming 'id' is an integer in the database
    title: str = Field(..., max_length=255, description="Title of the book")
    author: str = Field(..., max_length=255, description="Author of the book")
    genre: str | None = None  # Optional genre field with maximum length
    year_published: int | None = None  # Optional year published field (can be integer or None)
    summary: str | None = None  # Optional summary with maximum length

    def validate_string_length(value: str) -> str:
        if len(value) > 255:
            raise ValidationError("Field must be less than or equal to 255 characters")
        return value

    _pre_validators = {"title": validate_string_length, "author": validate_string_length, "summary": validate_string_length}
    class Config:
        orm_mode = True

class ReviewDto(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    review_text = Column(String)
    rating = Column(Integer)

class Review(BaseModel):
    id: int
    book_id: int = Field(..., foreign_key="Book.id")
    user_id: int
    review_text: str
    rating: int

    class Config:
        orm_mode = True

class UserDto(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)  # Store hashed password

class User(BaseModel):
    id: int
    username: str
    email: EmailStr  

    class Config:
        orm_mode = True
#endregion Model def: End

#region Automapper: start
mapper.add(Book, BookDto)
mapper.add(BookDto, Book)
mapper.add(Review, ReviewDto)
mapper.add(ReviewDto, Review)
mapper.add(User, UserDto)
mapper.add(UserDto, User)
#endregion Automapper: end

def to_dict(obj):
    dict = {}
    for field in obj.__dict__.keys():
        if not field.startswith("_"):  # Exclude private attributes (optional)
            value = getattr(obj, field)
            dict[field] = value
    return dict

# CRUD operations for books
async def create_book(book: Book):  
        async with async_session_maker() as db:        
            try:
                bdto = mapper.to(BookDto).map(book)
                async with db.begin():
                    db.add(bdto)
                await db.commit()
                await db.refresh(bdto)
                return book
            except IntegrityError as e:
                # Handle potential database constraint violations
                await db.rollback()
                raise ValueError(f"Error creating book: {e}") from e

async def get_all_books():    
        async with async_session_maker() as db:  
            try:
                books = await db.execute(select(BookDto)) 
                return books.scalars().all() 
            except Exception as e: 
                return {"error": str(e)}

async def get_book(book_id: int = Path(..., gt=0)):
    async with async_session_maker() as session:
        try:
            result = await session.execute(select(BookDto).filter(BookDto.id == book_id))
            book = result.scalars().first()
            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")
            return book
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def update_book(book_id: int = Path(..., gt=0), updated_data: Book = Body(...)):
    async with async_session_maker() as db:
        try:
            #session = scoped_session(db)
            result = await db.execute(select(BookDto).filter(BookDto.id == book_id))
            book = result.scalars().first()
            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")
            book = mapper.to(BookDto).map(updated_data)
            await db.execute(update(BookDto).where(BookDto.id == book_id).values(to_dict(book)))

            await db.commit()
            #await db.refresh(book)
            return book
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def delete_book(book_id: int = Path(..., gt=0)):
    async with async_session_maker() as db:
        try:  
            reviewresult = await db.execute(select(ReviewDto).filter(ReviewDto.book_id == book_id))
            reviews = reviewresult.scalars().all()
            if reviews:
                for review in reviews:
                    await db.delete(review)  
            await db.commit()
            result = await db.execute(select(BookDto).filter(BookDto.id == book_id))
            book = result.scalars().first()
            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")        
            await db.delete(book)
            await db.commit()
            return {"message": "Book deleted successfully"}
        
        except IntegrityError as e:  
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        except OperationalError as e:  
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except SQLAlchemyError as e:
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:  
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Endpoints for reviews
async def create_review(book_id: int = Path(..., gt=0), review: Review = Body(...)):
    async with async_session_maker() as db:
        try:
            review.book_id = book_id            
            rtodd = mapper.to(ReviewDto).map(review)
            if rtodd is None:
                raise ValueError("Failed to map review object")
            async with db.begin():
                db.add(rtodd)
            await db.commit()

            await db.refresh(rtodd)

            return review
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def get_book_reviews(book_id: int = Path(..., gt=0), rating: Union[int, None] = None):
    async with async_session_maker() as db:
        try:
            query = select(ReviewDto).filter(ReviewDto.book_id == book_id)
            if rating is not None:
                query = query.where(or_(Review.rating == rating, Review.rating - 1 == rating, Review.rating + 1 == rating))  # Example filter by rating (including +/- 1)

            results = await db.execute(query)
            reviews = results.scalars().all()
            return reviews
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def get_book_summary(book_id: int = Path(..., gt=0)):
    async with async_session_maker() as db:
        try:
            query = query = select(BookDto) \
            .options(joinedload(BookDto.reviews)) \
            .filter(BookDto.id == book_id)
            result = await db.execute(query)
            book = result.scalars().first()
            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")

            average_rating = None
            if book.reviews:
                average_rating_query = select(func.avg(Review.rating)).filter(Review.book_id == book.id)
                average_rating = await db.execute(average_rating_query)
                average_rating = average_rating.scalar()  # Extract single value

            summary = ""
            if book.summary:
                summary = book.summary
            else:
                if book.content:
                    try:
                        # Replace with your preferred asynchronous summarization logic (NLTK, Gensim, etc.)
                        summary = await summarize_text(book.content)  # Placeholder async function
                    except Exception as e:
                        print(f"Error during summarization: {e}")  # Log or handle the error

            return {"summary": summary, "average_rating": average_rating}
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def summarize_text(text: str):
    nlp = spacy.load("en_core_web_sm")  # Load a small English language model
    doc = nlp(text)  # Process the text with spaCy

    # Extract noun phrases and named entities (assuming these indicate key information)
    phrases = []
    for token in doc.ents:
        phrases.append(token.text)
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text)

    # Calculate sentence scores based on the presence of phrases
    sentence_scores = {i: len([p for p in phrases if p in sentence.text]) for i, sentence in enumerate(doc.sents)}

    # Select the top N sentences (adjust N for desired summary length)
    num_sentences = 3
    summary_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)[:num_sentences]

    # Join the top sentences to form the summary
    summary = ' '.join(sentence.text for sentence in doc.sents if (sentence.start, sentence.end) in summary_sentences)

    return summary

async def get_book_recommendations():
    async with async_session_maker() as db:
        # Recommend top 10 most rated books (replace with your desired criteria)
        query = select(BookDto) \
            .order_by(func.desc(BookDto.average_rating)) \
            .limit(10)
        results = await db.execute(query)
        books = results.scalars().all()

        return books

async def create_user():
    async with async_session_maker() as db:        
            try:
                user_data = [{"username": "user1", "email": "user1@example.com", "hashed_password": "hashed_password1"},
                             {"username": "user2", "email": "user2@example.com", "hashed_password": "hashed_password2"},
                             {"username": "user3", "email": "user3@example.com", "hashed_password": "hashed_password3"}]                
                for user in user_data:
                    new_user = UserDto(**user)
                    async with db.begin():
                        db.add(new_user)                    
                    
                await db.commit()  
                await db.refresh(new_user)  
                return user_data
            except Exception as e:  
                await db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def get_all_user():
    async with async_session_maker() as db:        
            try:
                users = await db.execute(select(UserDto)) 
                return users.scalars().all() 
            except Exception as e:  
                await db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


#region fastAPI: start
app = FastAPI()

@app.get("/")
async def index():
   return {"message": "Hello World"}

@app.post("/books", status_code=status.HTTP_201_CREATED)
async def postBooks(book: Book):
    return await create_book(book)

@app.get("/books")
async def getAllBooks():
    return await get_all_books()

@app.get("/books/{book_id}")
async def get_book_by_id(book_id: int):
    book = await get_book(book_id)
    return book

@app.put("/books/{book_id}")
async def updateBook(book_id: int, book: Book):
    return await update_book(book_id, book)

@app.delete("/books/{book_id}")
async def deleteBook(book_id: int):
    return await delete_book(book_id)

@app.post("/books/{book_id}/reviews", status_code=status.HTTP_201_CREATED)
async def createReview(book_id: int, review: Review):
    return await create_review(book_id, review)

@app.get("/books/{book_id}/reviews")
async def getBookReviews(book_id: int, rating: Union[int, None]):
    return await get_book_reviews(book_id, rating)

@app.get("/books/{book_id}/summary")
async def getBookSummary(book_id: int):
    return await get_book_summary(book_id)

@app.get("/recommendations")
async def getBookRecommendations():
    return await get_book_recommendations()

@app.put("/users")
async def createusers():
    return await create_user()

@app.get("/users")
async def getallusers():
    return await get_all_user()

#endregion fastAPI: end

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_async_tables())
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8020, reload=True)
