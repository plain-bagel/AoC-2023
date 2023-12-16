module BaseParser (
    Parser (..),
    charP,
    spanP,
) where

import Control.Applicative

-- Define Parser type for generic type a
newtype Parser a = Parser
    { runParser :: String -> Maybe (a, String)
    }

-- Apply a function to the result of a parser
instance Functor Parser where
    fmap f (Parser p) = Parser $ \input ->
        case p input of
            Just (y, ys) -> Just (f y, ys)
            Nothing -> Nothing

-- Apply a parser to the result of another parser
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

-- Parser for character
charP :: Char -> Parser Char
charP x = Parser f
  where
    f (y : ys)
        | y == x = Just (x, ys)
        | otherwise = Nothing
    f [] = Nothing

-- Parse based on predicate condition
spanP :: (Char -> Bool) -> Parser String
spanP f = Parser $ \input ->
    case input of
        (x : xs) | f x -> Just (span f input)
        _ -> Nothing
