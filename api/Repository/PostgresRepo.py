from sqlite3 import IntegrityError, OperationalError
from typing import Union
from fastapi import Body, HTTPException, Path, status
import spacy
from sqlalchemy import func, update
from sqlalchemy.future import select
from Helper.CommonHelper import CommonHelper
from sqlalchemy.orm import joinedload
from MapperConfig import *
from Helper.PdfHelper import PdfHelper
from Repository.SqlAlchemySetup import SqlAlchemySetup


class PostgresRepo:
    #region book
    async def create_book(self, book: Book):  
            async with SqlAlchemySetup.async_session_maker() as db:        
                try:
                    bdto = mapper.to(BookDto).map(book)
                    async with db.begin():
                        db.add(bdto)
                    await db.commit()
                    await db.refresh(bdto)
                    self.generateBookSummary(bdto)
                    return book
                except IntegrityError as e:
                    # Handle potential database constraint violations
                    await db.rollback()
                    raise ValueError(f"Error creating book: {e}") from e

    async def get_all_books(self):    
            async with SqlAlchemySetup.async_session_maker() as db:  
                try:
                    books = await db.execute(select(BookDto)) 
                    return books.scalars().all() 
                except Exception as e: 
                    return {"error": str(e)}

    async def get_book(self, book_id: int = Path(..., gt=0)):
        async with SqlAlchemySetup.async_session_maker() as session:
            try:
                result = await session.execute(select(BookDto).filter(BookDto.id == book_id))
                book = result.scalars().first()
                if not book:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")
                return book
            except Exception as e:  # Handle broader exceptions
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def update_book(self, book_id: int = Path(..., gt=0), updated_data: Book = Body(...)):
        async with SqlAlchemySetup.async_session_maker() as db:
            try:
                #session = scoped_session(db)
                result = await db.execute(select(BookDto).filter(BookDto.id == book_id))
                book = result.scalars().first()
                if not book:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {book_id} not found")
                book = mapper.to(BookDto).map(updated_data)
                await db.execute(update(BookDto).where(BookDto.id == book_id).values(CommonHelper.to_dict(book)))

                await db.commit()
                #await db.refresh(book)
                return book
            except Exception as e:  # Handle broader exceptions
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def delete_book(self, book_id: int = Path(..., gt=0)):
        async with SqlAlchemySetup.async_session_maker() as db:
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
            except Exception as e:  
                await db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    #endregion book

    #region user
    async def create_user(self):
        async with SqlAlchemySetup.async_session_maker() as db:        
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

    async def get_all_user(self):
        async with SqlAlchemySetup.async_session_maker() as db:         
            try:
                users = await db.execute(select(UserDto)) 
                return users.scalars().all() 
            except Exception as e:  
                await db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    #endregion user

    #region reviews
    async def create_review(self, book_id: int = Path(..., gt=0), review: Review = Body(...)):
        async with SqlAlchemySetup.async_session_maker() as db:
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

    async def get_book_reviews(self, book_id: int = Path(..., gt=0), rating: Union[int, None] = None):
        async with SqlAlchemySetup.async_session_maker() as db:
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
    async def generateBookSummary(self, book: BookDto):
        async with SqlAlchemySetup.async_session_maker() as db:
            try:
                pdf_path = "C:/Users/bipla/OneDrive/Code/bookmgmtGenAi/llm/books/" + book.title
                text = PdfHelper.extract_text_from_pdf(book.title)
                if text:
                    cleaned_text = PdfHelper.clean_text(text)
                booksummary = self.summaryModel(cleaned_text)
                book.summary = booksummary
                #save to db
                self.update_book(book)
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

    async def get_book_summary(self, book_id: int = Path(..., gt=0)):
        async with SqlAlchemySetup.async_session_maker() as db:
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
                            summary = await self.summarize_text_instant(book.content)  # Placeholder async function
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

    async def get_book_recommendations(self):
        async with SqlAlchemySetup.async_session_maker() as db:
            # Recommend top 10 most rated books (replace with your desired criteria)
            query = select(BookDto) \
                .order_by(func.desc(BookDto.average_rating)) \
                .limit(10)
            results = await db.execute(query)
            books = results.scalars().all()

            return books
#endregion recomendation