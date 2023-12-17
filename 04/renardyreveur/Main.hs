module Main where

import BaseParser
import qualified Data.Map as Map

-- Define Scratchcard data type
data Scratchcard = Scratchcard
    { cardID :: Int
    , winners :: [Int]
    , numbers :: [Int]
    }
    deriving (Show)

-- Scratchcard parser
idP :: Parser Int
idP = stringP "Card" *> ws *> intP <* ws <* charP ':' <* ws

winP :: Parser [Int]
winP = ws *> sepBy ws intP <* ws <* charP '|'

numP :: Parser [Int]
numP = ws *> sepBy ws intP <* ws

scratchcard :: Parser Scratchcard
scratchcard = Scratchcard <$> idP <*> winP <*> numP

-- Scratchcard functions
winCards :: Scratchcard -> [Int]
winCards (Scratchcard i w n) = [i + 1 .. i + length (filter (`elem` w) n)]

recursiveWins :: [Scratchcard] -> Int -> Map.Map Int Int -> Map.Map Int Int
recursiveWins deck idx acc =
    let
        -- Find card with index idx
        card =
            case filter (\x -> cardID x == idx) deck of
                [] -> Nothing
                (x : xs) -> Just x

        -- Find all new cards won by card
        wins = winCards $ fromMaybe card

        -- Update Map with new cards won
        numCard = fromMaybe $ Map.lookup idx acc
        newAcc = foldl (\ac x -> Map.insertWith (+) x (1 * numCard) ac) acc wins
     in
        -- Stop if no card is found
        if null card then acc else recursiveWins deck (idx + 1) newAcc

-- Main Function
main :: IO ()
main = do
    -- Read Scratchcards, and parse them line by line
    scratchcards <- readFile "04/input.txt"
    let card = lines scratchcards

    -- Parse Scratchcards
    let parsed = map (fst . fromMaybe . runParser scratchcard) card

    -- Compare Winners and Numbers
    let points = foldl (\acc (Scratchcard i w n) -> (floor $ 2 ^^ ((length (filter (`elem` w) n)) - 1) :: Int) + acc) 0 parsed
    putStrLn $ "Scratchcard points worth in total: " ++ show points

    -- Part 2
    let cardNumMap = Map.fromList $ map (\x -> (cardID x, 1)) parsed
    let cardsWon = recursiveWins parsed 1 cardNumMap

    -- Sum all values in Map
    let total = Map.foldl (+) 0 cardsWon
    putStrLn $ "Total number of cards won: " ++ show total
