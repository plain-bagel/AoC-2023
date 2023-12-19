module StringParser (
    Parser,
    runParser,
    charP,
    stringP,
    spanP,
    notNull,
    intP,
    sepBy,
    ws,
) where

import Control.Applicative
import Data.Char (isDigit, isSpace)

{-
type : Type aliases
data : new data types, complex types with multiple constructors (See Colour)
newtype : distinct data type from existing type (wraps), single constructor, efficient
-}
-- Define Parser type for generic type a
newtype Parser a = Parser
    { runParser :: String -> Maybe (a, String)
    -- Given a string, return parsed value and remaining string
    }

{-
For each field, Haskell creates a function with the same name as the field to extract the value
runParser :: Parser a -> String -> Maybe (a, String)
so you can wrap and unwrap the value with Parser and runParser
-}

-- (from stringP) To prove Parser is an Applicative, we need to prove it is a Functor first (class Functor f => Applicative f where ...)
-- minimal impl. of Functor is fmap (fmap :: (a -> b) -> f a -> f b), Inject function into the functor
instance Functor Parser where
    fmap f (Parser p) = Parser $ \input ->
        -- Wrap input(String) with given Parser, returning Maybe(a, String)
        case p input of
            Just (y, ys) -> Just (f y, ys)
            Nothing -> Nothing

-- Now that Parser is a Functor, we can prove that it is an Applicative
instance Applicative Parser where
    pure p = Parser $ \input -> Just (p, input)
    (Parser p1) <*> (Parser p2) = Parser $ \input ->
        case p1 input of
            Just (f, rest) ->
                case p2 rest of
                    Just (x, leftover) -> Just (f x, leftover)
                    Nothing -> Nothing
            Nothing -> Nothing

-- Alternative is a typeclass that represents a choice between two values
instance Alternative Parser where
    empty = Parser $ \_ -> Nothing -- always fail
    (Parser p1) <|> (Parser p2) = Parser $ \input ->
        case p1 input of
            Just (x, leftover) -> Just (x, leftover)
            Nothing -> p2 input

-- To parse colour, we need to parse the string "blue", "red", "green"
-- To parse a string, we need to parse a list of characters, and thus a character first

-- Parser for character (parameterized by character that we want to parse, returns a Parser that parses that character)
charP :: Char -> Parser Char
charP x = Parser f
  where
    f (y : ys)
        | y == x = Just (x, ys)
        | otherwise = Nothing
    f [] = Nothing

{-
We can `map charP` over list of characters, but that gives us `[Parser Char]`
We want `Parser [Char]`

Haskell has a function called `sequenceA`
  sequenceA :: Applicative f => t (f a) -> f (t a)
It takes a traversable of applicatives, and returns an Applicative of traversables (turns type inside out)
list is traversable, so if Parser is an applicative, we can use sequenceA to turn [Parser Char] into Parser [Char]
-}
-- Parser for string (parameterized by string that we want to parse, returns a Parser that parses that string)
stringP :: String -> Parser String
-- Base case, minimally wrap an empty string into an Applicative Parser instance
stringP [] = pure []
stringP (c : cs) = (:) <$> charP c <*> stringP cs -- Recursive chaining of charP
-- alt: stringP = sequenceA . map charP

-- Parse based on predicate condition
spanP :: (Char -> Bool) -> Parser String
spanP f = Parser $ \input -> Just (span f input)

-- Parser that returns Nothing if the result is empty
notNull :: Parser [a] -> Parser [a]
notNull (Parser p) = Parser $ \input ->
    case p input of
        Just ([], _) -> Nothing
        Just (x, xs) -> Just (x, xs)
        Nothing -> Nothing

-- Parser for integer
intP :: Parser Int
intP = read <$> notNull (spanP isDigit)

-- First create a parser that parses an element with a separator, and returns a list of elements separated by the separator
sepBy :: Parser a -> Parser b -> Parser [b]
sepBy sep element = (:) <$> element <*> many (sep *> element) <|> pure [] -- Inject (:) into the functor, then apply to element, then apply to many (sep *> element)
-- If the parser fails, we want to return an empty list so provide an alternative with `pure []`

-- Handle whitespace around delimiters
ws :: Parser String
ws = spanP isSpace
