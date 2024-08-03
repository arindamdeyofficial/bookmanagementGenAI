from typing import Union

from fastapi.responses import JSONResponse
from BusinessObjects.models import Book, Review
from fastapi import Depends, FastAPI, status
from Repository.PostgresRepo import PostgresRepo
from apiapp import fastapiapp

app = fastapiapp

@app.post("/books", status_code=status.HTTP_201_CREATED)
async def postBooks(book: Book, repo: PostgresRepo = Depends()):
    """Creates a new book in the database.

    This endpoint expects a JSON request body containing book details according to the BookCreate schema.

    Args:
        book: A BookCreate object containing book information.

    Returns:
        The newly created Book object in JSON format.
    """
    try:
        new_book = await repo.create_book(book)
        return new_book
    except Exception as e:
        return str(e)

@app.get("/books")
async def getAllBooks(repo: PostgresRepo = Depends()):
    """Retrieves a list of all books in the database.

    Returns:
        A JSON array containing all Book objects in the database.
    """
    return await repo.get_all_books()

@app.get("/books/{book_id}")
async def get_book_by_id(book_id: int, repo: PostgresRepo = Depends()):
    """Retrieves a specific book by its ID.

    Args:
        book_id: The unique integer identifier of the book to retrieve.

    Returns:
        The Book object with the matching ID, or a 404 Not Found response if the book doesn't exist.
    """
    book = await repo.get_book(book_id)
    return book

@app.put("/books/{book_id}")
async def updateBook(book_id: int, book: Book, repo: PostgresRepo = Depends()):
    """Updates an existing book in the database.

    Args:
        book_id: The unique integer identifier of the book to update.
        book_updates: A BookUpdate object containing the new data for the book (fields to update).

    Returns:
        The updated Book object in JSON format, or a 404 Not Found response if the book doesn't exist.
    """
    return await repo.update_book(book_id, book)

@app.delete("/books/{book_id}")
async def deleteBook(book_id: int, repo: PostgresRepo = Depends()):
    """Deletes a book from the database.

    Args:
        book_id: The unique integer identifier of the book to delete.

    Returns:
        A 204 No Content response upon successful deletion, or a 404 Not Found response if the book doesn't exist.
    """
    return await repo.delete_book(book_id)

@app.post("/books/{book_id}/reviews", status_code=status.HTTP_201_CREATED)
async def createReview(book_id: int, review: Review, repo: PostgresRepo = Depends()):
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
    return await repo.create_review(book_id, review)

@app.get("/books/{book_id}/reviews")
async def getBookReviews(book_id: int, repo: PostgresRepo = Depends()):
    """Retrieves all reviews for a specific book.

    This endpoint allows users to retrieve a list of all reviews associated with a particular book in your database.

    Args:
        book_id: The unique integer identifier of the book for which to retrieve reviews.

    Returns:
        A JSON array containing all Review objects for the specified book, or a 404 Not Found response if the book doesn't exist.
    """
    return await repo.get_book_reviews(book_id)

@app.get("/books/{book_id}/{rating}/reviews")
async def getBookReviewsFilteredbyRating(book_id: int, rating: Union[int, None], repo: PostgresRepo = Depends()):
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
    return await repo.get_book_reviews(book_id, rating)

@app.get("/books/{book_id}/summary")
async def getBookSummary(book_id: int, repo: PostgresRepo = Depends()):
    """Retrieves a summary of a book, including its average rating.

    This endpoint attempts to generate a summary of the book's content and provides the average rating
    based on all associated reviews. The specific method for generating the summary depends on your implementation.

    Args:
        book_id: The unique integer identifier of the book for which to retrieve the summary.

    Returns:
        A BookSummary object containing the generated summary text (if available) and the average rating.
        If summary generation is not supported or the book doesn't exist, a 404 Not Found response is returned.
    """
    return await repo.get_book_summary(book_id)

@app.get("/recommendations")
async def getBookRecommendations(repo: PostgresRepo = Depends()):
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
    return await repo.get_book_recommendations()