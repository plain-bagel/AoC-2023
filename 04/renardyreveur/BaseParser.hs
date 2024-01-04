module BaseParser (
    Parser (..),
    charP,
    stringP,
    spanP,
    notNull,
    intP,
    sepBy,
    ws,
    fromMaybe,
    sHead,
    sTail,
) where

import Control.Applicative
import Data.Char (isDigit, isSpace)

-- Define Parser type for generic type a
newtype Parser a = Parser
    { runParser :: String -> Maybe (a, String)
    }

instance Functor Parser where
    fmap f (Parser p) = Parser $ \input ->
        case p input of
            Just (y, ys) -> Just (f y, ys)
            Nothing -> Nothing

instance Applicative Parser where
    pure p = Parser $ \input -> Just (p, input)
    (Parser p1) <*> (Parser p2) = Parser $ \input ->
        case p1 input of
            Just (f, rest) ->
                case p2 rest of
                    Just (x, leftover) -> Just (f x, leftover)
                    Nothing -> Nothing
            Nothing -> Nothing

instance Alternative Parser where
    empty = Parser $ \_ -> Nothing -- always fail
    (Parser p1) <|> (Parser p2) = Parser $ \input ->
        case p1 input of
            Just (x, leftover) -> Just (x, leftover)
            Nothing -> p2 input

charP :: Char -> Parser Char
charP x = Parser f
  where
    f (y : ys)
        | y == x = Just (x, ys)
        | otherwise = Nothing
    f [] = Nothing

stringP :: String -> Parser String
stringP [] = pure []
stringP (c : cs) = (:) <$> charP c <*> stringP cs -- Recursive chaining of charP

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

-- Handle whitespace around delimiters
ws :: Parser String
ws = spanP isSpace

fromMaybe :: Maybe a -> a
fromMaybe (Just x) = x
fromMaybe Nothing = error "Nothing"

sHead :: [a] -> a
sHead [] = error "Empty list"
sHead (x : _) = x

sTail :: [a] -> [a]
sTail [] = error "Empty list"
sTail (_ : xs) = xs
