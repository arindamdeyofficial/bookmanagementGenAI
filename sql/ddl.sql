INSERT INTO users
(id, username, email, hashed_password)
VALUES(nextval('users_id_seq'::regclass), 'user1@example.com', 'user1', 'dghtrrtrh')

INSERT INTO users (username, email)
VALUES ('user1@example.com', 'user1'),
       ('user2@example.com', 'user2'),
       ('user3@example.com', 'user3');

CREATE TABLE books (
  id SERIAL  PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  author VARCHAR(255) NOT NULL,
  genre VARCHAR(50),
  year_published INT,
  summary TEXT
);

INSERT INTO books (title, author, genre, year_published, summary)
VALUES ('The Hitchhiker Guide to the Galaxy', 'Douglas Adams', 'Science Fiction Comedy', 1979, 'On the verge of Earths demolition for a hyperspace bypass, Arthur Dent is whisked away by his friend Ford Prefect, a researcher for the titular Hitchhikers Guide to the Galaxy.'),
       ('Pride and Prejudice', 'Jane Austen', 'Romance', 1813, 'In an English country village, the wealthy and arrogant Mr. Darcy meets the lively Elizabeth Bennet. Sparks fly as their initial dislike slowly transforms into something deeper.'),
       ('The Lord of the Rings: The Fellowship of the Ring', 'J.R.R. Tolkien', 'Epic Fantasy', 1954, 'Frodo Baggins inherits the One Ring, an evil artifact of the Dark Lord Sauron. He embarks on a quest to destroy the Ring and ensure the future of Middle-earth.'),
       ('To Kill a Mockingbird', 'Harper Lee', 'Historical Fiction', 1960, 'Scout Finch, a young girl living in Alabama during the 1930s, witnesses the trial of a black man wrongly accused of assaulting a white woman. Through her innocent eyes, we see the complexities of racism in the Deep South.'),
       ('The Catcher in the Rye', 'J.D. Salinger', 'Coming-of-Age Fiction', 1951, 'Holden Caulfield, a cynical teenager, is expelled from boarding school. He wanders the streets of New York City, reflecting on his disillusionment with society.');

CREATE TABLE reviews (
  id SERIAL PRIMARY KEY,
  book_id INT NOT NULL,
  user_id INT NOT NULL,
  review_text TEXT,
  rating DECIMAL(2, 1),  -- Allows ratings with one decimal place (e.g., 4.5)
  FOREIGN KEY (book_id) REFERENCES books(id),
  FOREIGN KEY (user_id) REFERENCES users(id)  -- Assuming a users table exists
);

INSERT INTO reviews (book_id, user_id, review_text, rating)
VALUES
  (1, 1, 'A fantastic and witty sci-fi adventure!', 4.8),
  (1, 2, 'A must-read for any Adams fan!', 5.0),
  (2, 3, 'A timeless classic of romance and social commentary.', 4.5),
  (2, 1, 'Witty and engaging story of love and misunderstanding.', 4.2),
  (3, 2, 'An epic journey filled with adventure and heroism.', 4.7),
  (3, 1, 'A captivating tale that explores the themes of good vs. evil.', 4.9);

