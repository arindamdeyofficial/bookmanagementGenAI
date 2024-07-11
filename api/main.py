#region imports
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

#endregion imports

#region SQLAlchemy setup
DATABASE_URL = "postgresql+asyncpg://postgres:nakshal01051987@localhost:5432/bookreviewmgmt"
async_engine = create_async_engine(DATABASE_URL, connect_args={"command_timeout": 28.0})
async_session_maker = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine, expire_on_commit=False)
Base = declarative_base()

async def create_async_tables():
    async with async_engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
#endregion SQLAlchemy setup

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

#region book
async def create_book(book: Book):  
        async with async_session_maker() as db:        
            try:
                bdto = mapper.to(BookDto).map(book)
                async with db.begin():
                    db.add(bdto)
                await db.commit()
                await db.refresh(bdto)
                generateBookSummary(bdto)
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
#endregion book

#region reviews
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
                query = query.where(ReviewDto.rating == rating)

            results = await db.execute(query)
            reviews = results.scalars().all()
            return reviews
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
#endregion reviews

#region recomendation
async def generateBookSummary(book: BookDto):
    async with async_session_maker() as db:
        try:
            pdf_path = "C:/Users/bipla/OneDrive/Code/bookmgmtGenAi/llm/books/" + book.title
            text = extract_text_from_pdf(book.title)
            if text:
                cleaned_text = clean_text(text)
            booksummary = summaryModel(cleaned_text)
            book.summary = booksummary
            #save to db
            update_book(book)
            return {"status": "submitted to generate review"}
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def summaryModel(text: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
    import torch

    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    #C:\Users\bipla\.cache\huggingface\hub download location
    #generate summary
    # Tokenize the input
    encoded_input = tokenizer(text, return_tensors="pt")  # Convert to tensors (optional)

    # Generate output using the model
    with torch.no_grad():  # Disable gradient calculation for efficiency
        output = model.generate(**encoded_input)
    # Decode the model's output tokens back to text
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

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
                        summary = await summarize_text_instant(book.content)  # Placeholder async function
                    except Exception as e:
                        print(f"Error during summarization: {e}")  # Log or handle the error

            return {"summary": summary, "average_rating": average_rating}
        except Exception as e:  # Handle broader exceptions
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

async def summarize_text_instant(text: str):
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
#endregion recomendation

#region user
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
#endregion user

#region reading pdf
import re
import PyPDF2

def extract_text_from_pdf(pdf_path):
  """
  Extracts text content from a PDF file.

  Args:
      pdf_path: Path to the PDF file.

  Returns:
      A string containing the extracted text content.
  """
  try:
    with open(pdf_path, 'rb') as pdf_file:
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      text = ""
      for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
      return text
  except FileNotFoundError:
    print(f"Error: PDF file not found at {pdf_path}")
    return None
  except Exception as e:  # Handle broader exceptions
    print(e)

def clean_text(text):
  """
  Performs basic cleaning on the extracted text.

  Args:
      text: The extracted text content from the PDF.

  Returns:
      A string containing the cleaned text.
  """
  # Replace common non-alphanumeric characters
  cleaned_text = text.replace("\\n", " ").replace("\\t", " ")  # Replace newlines and tabs
  cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_text)  # Remove non-alphanumeric characters (except space)

  # You can add more cleaning steps here, such as:
  # - Lowercasing all characters
  # - Removing punctuation
  # - Removing stop words

  return cleaned_text

#endregion reading pdf

#region fastAPI: start
app = FastAPI()

@app.get("/")
async def index():
   return {"message": "Hello World"}

@app.post("/books", status_code=status.HTTP_201_CREATED)
async def postBooks(book: Book):
    """Creates a new book in the database.

    This endpoint expects a JSON request body containing book details according to the BookCreate schema.

    Args:
        book: A BookCreate object containing book information.

    Returns:
        The newly created Book object in JSON format.
    """
    return await create_book(book)

@app.get("/books")
async def getAllBooks():
    """Retrieves a list of all books in the database.

    Returns:
        A JSON array containing all Book objects in the database.
    """
    return await get_all_books()

@app.get("/books/{book_id}")
async def get_book_by_id(book_id: int):
    """Retrieves a specific book by its ID.

    Args:
        book_id: The unique integer identifier of the book to retrieve.

    Returns:
        The Book object with the matching ID, or a 404 Not Found response if the book doesn't exist.
    """
    book = await get_book(book_id)
    return book

@app.put("/books/{book_id}")
async def updateBook(book_id: int, book: Book):
    """Updates an existing book in the database.

    Args:
        book_id: The unique integer identifier of the book to update.
        book_updates: A BookUpdate object containing the new data for the book (fields to update).

    Returns:
        The updated Book object in JSON format, or a 404 Not Found response if the book doesn't exist.
    """
    return await update_book(book_id, book)

@app.delete("/books/{book_id}")
async def deleteBook(book_id: int):
    """Deletes a book from the database.

    Args:
        book_id: The unique integer identifier of the book to delete.

    Returns:
        A 204 No Content response upon successful deletion, or a 404 Not Found response if the book doesn't exist.
    """
    return await delete_book(book_id)

@app.post("/books/{book_id}/reviews", status_code=status.HTTP_201_CREATED)
async def createReview(book_id: int, review: Review):
    """Creates a new review for a specific book.

    This endpoint allows users to submit reviews for books in your database.

    Args:
        book_id: The unique integer identifier of the book to which the review belongs.
        review: A ReviewCreate object containing the details of the new review (content and rating).

    Returns:
        The newly created Review object in JSON format, including its automatically generated ID.

    Raises:
        HTTPException: A 404 Not Found exception if the specified book does not exist.
    """
    return await create_review(book_id, review)

@app.get("/books/{book_id}/reviews")
async def getBookReviews(book_id: int):
    """Retrieves all reviews for a specific book.

    This endpoint allows users to retrieve a list of all reviews associated with a particular book in your database.

    Args:
        book_id: The unique integer identifier of the book for which to retrieve reviews.

    Returns:
        A JSON array containing all Review objects for the specified book, or a 404 Not Found response if the book doesn't exist.
    """
    return await get_book_reviews(book_id)

@app.get("/books/{book_id}/{rating}/reviews")
async def getBookReviewsFilteredbyRating(book_id: int, rating: Union[int, None]):
    """Retrieves reviews for a specific book, optionally filtered by rating.

    This endpoint allows users to retrieve a list of reviews associated with a particular book in your database.
    You can optionally filter the reviews by specifying a desired rating value.

    Args:
        book_id: The unique integer identifier of the book for which to retrieve reviews.
        rating: An optional integer value representing the desired review rating. If omitted, all reviews are returned.

    Returns:
        A JSON array containing all matching Review objects for the specified book and rating (if provided),
        or a 404 Not Found response if the book doesn't exist.

    Raises:
        HTTPException: A 400 Bad Request exception if the provided rating is outside the valid range.
    """
    return await get_book_reviews(book_id, rating)

@app.get("/books/{book_id}/summary")
async def getBookSummary(book_id: int):
    """Retrieves a summary of a book, including its average rating.

    This endpoint attempts to generate a summary of the book's content and provides the average rating
    based on all associated reviews. The specific method for generating the summary depends on your implementation.

    Args:
        book_id: The unique integer identifier of the book for which to retrieve the summary.

    Returns:
        A BookSummary object containing the generated summary text (if available) and the average rating.
        If summary generation is not supported or the book doesn't exist, a 404 Not Found response is returned.
    """
    return await get_book_summary(book_id)

@app.get("/recommendations")
async def getBookRecommendations():
    """Retrieves recommendations for books based on your chosen recommendation strategy.

    This endpoint allows you to retrieve a list of recommended books based on a specific recommendation strategy.
    The actual implementation of the recommendation algorithm depends on your chosen approach.

    **Possible strategies (implement one or more):**

    - Collaborative Filtering: Recommends books similar to those users with similar tastes have reviewed highly.
    - Content-Based Filtering: Recommends books with similar content or genre to books a user has reviewed positively.
    - Hybrid Approaches: Combine collaborative and content-based filtering for more comprehensive recommendations.

    **Current Implementation:**

    (Replace this section with a description of the specific recommendation strategy you're using.)

    Returns:
        A JSON array containing recommended Book objects, or an empty list if no recommendations are found.
    """
    return await get_book_recommendations()

@app.put("/users")
async def createusers():
    """Creates a new user account with email and password.

    This endpoint allows users to register for your application by providing their email address and a password.

    **Security Note:** Passwords are hashed before storing them in the database, ensuring secure user authentication.

    Args:
        user: A UserCreate object containing the new user's email and password.
        db: A database session dependency (injected through dependency injection).

    Returns:
        The newly created User object with basic user information (excluding password).

    Raises:
        HTTPException: A 400 Bad Request exception if the email is already in use.
    """
    return await create_user()

@app.get("/users")
async def getallusers():
    """Retrieves all users in the database (deprecated).

    **Warning:** This endpoint is currently deprecated and exposes potentially sensitive user data. 
    Retrieving all users can be a privacy and security concern.

    For user data management, consider alternative approaches:

    - User Authentication & Authorization: Implement user login and access control based on roles or permissions.
    - User Self-Management: Provide endpoints for users to manage their own profiles.
    - Admin User Management (Restricted Access): Create an admin role with limited access to user data for specific purposes.

    This endpoint is intended for development or testing purposes only.

    Returns:
        A JSON array containing all User objects (not recommended for production).
    """
    return await get_all_user()

#endregion fastAPI: end

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_async_tables())
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8020, reload=True)
