import select
from fastapi import Depends, HTTPException, status
from Repository.PostgresRepo import PostgresRepo
from apiapp import fastapiapp

app = fastapiapp

@app.put("/users")
async def createusers(repo: PostgresRepo = Depends()):
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
    return await repo.create_user()

@app.get("/users")
async def getallusers(repo: PostgresRepo = Depends()):
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
    return await repo.get_all_user()