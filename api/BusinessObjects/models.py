#region imports

from pydantic import BaseModel, EmailStr, Field, ValidationError

#endregion imports

#region Model def: start

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

class Review(BaseModel):
    id: int
    book_id: int = Field(..., foreign_key="Book.id")
    user_id: int
    review_text: str
    rating: int

    class Config:
        orm_mode = True

class User(BaseModel):
    id: int
    username: str
    email: EmailStr  

    class Config:
        orm_mode = True
        
#endregion Model def: End
