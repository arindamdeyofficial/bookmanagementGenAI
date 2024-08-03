from automapper import mapper
from BusinessObjects.dbModels import *
from BusinessObjects.models import *

class MapperConfig:
    #region Automapper: start
    mapper.add(Book, BookDto)
    mapper.add(BookDto, Book)
    mapper.add(Review, ReviewDto)
    mapper.add(ReviewDto, Review)
    mapper.add(User, UserDto)
    mapper.add(UserDto, User)
    #endregion Automapper: end