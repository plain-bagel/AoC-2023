module Main where

import Control.Applicative
import StringParser

-- Abstract syntax tree for Cube Game Record
-- Sample string to parse "Game 1: 4 blue, 4 red, 16 green; 14 green, 5 red; 1 blue, 3 red, 5 green"
data CubeGame = CubeGame Int [Trials] deriving (Show, Eq)
type Trials = [ColourCount]
data ColourCount = ColourCount Colour Int deriving (Show, Eq)
instance Ord ColourCount where
    (ColourCount c1 a) `compare` (ColourCount c2 b)
        | c1 == c2 = a `compare` b
        | otherwise = a `compare` a -- Different colours should not be compared
data Colour = Blue | Red | Green deriving (Show, Eq)

-- Parser for Colour
colourP :: Parser Colour
colourP =
    -- \*> is a combinator that evaluates but ignores the result of the first parser and returns the result of the second parser
    (stringP "blue" *> pure Blue)
        <|> (stringP "red" *> pure Red)
        <|> (stringP "green" *> pure Green)

-- Parser for ColourCount (e.g "4 blue")
colourCount :: Parser ColourCount
colourCount = flip ColourCount <$> (intP <* (charP ' ')) <*> colourP -- Apply ColourCount constructor to Colour and Int

{-
Trials are many ColourCounts separated by commas
The Alternative instance for Parser has a `many` function that looks like
many :: Alternative f => f a -> f [a]

So what we want to do is parse a ColourCount, then a comma, and tell the parser there could be more ColourCounts
-}
-- Parser for Trials (e.g "4 blue, 4 red, 16 green;")
trialParser :: Parser Trials
trialParser = sepBy (charP ',' <* ws) colourCount

-- Parser for CubeGame
cubeGameParser :: Parser CubeGame
cubeGameParser = CubeGame <$> (stringP "Game " *> intP <* charP ':' <* ws) <*> sepBy (charP ';' <* ws) trialParser

-- Check valid games
validTrial :: [ColourCount] -> Maybe (CubeGame, String) -> Bool
validTrial limits game =
    case game of
        Just (CubeGame _ trials, _) -> all isValidTrial trials
          where
            isValidTrial trial = all (\cc -> all (cc <=) limits) trial
        Nothing -> False

-- Find maximum count for each colour in a game
maxColourCount :: CubeGame -> [ColourCount]
maxColourCount (CubeGame _ trials) =
    -- Iterate over all possible colours, and find the maximum count for each colour
    map (\c -> ColourCount c (maximum [x | ColourCount c2 x <- trial, c == c2])) [Blue, Green, Red]
  where
    trial = concat trials

-- Main
main :: IO ()
main = do
    -- Read game records from file
    gameRecords <- readFile "02/input.txt"

    -- Parse game records into a list of CubeGames
    let parsedGameRecords = map (runParser cubeGameParser) $ lines gameRecords

    -- Part 1
    let blueLimit = ColourCount Blue 14
    let greenLimit = ColourCount Green 13
    let redLimit = ColourCount Red 12

    -- Filter out invalid games and sum the game IDs
    let validGames = filter (validTrial [blueLimit, greenLimit, redLimit]) parsedGameRecords
    let sumOfValidGameIDs = sum $ map f validGames
          where
            f (Just (CubeGame gameId _, _)) = gameId
            f Nothing = 0
    putStrLn $ "Sum of Valid Game IDs: " ++ show sumOfValidGameIDs

    -- Part 2
    -- Find minimum limit for each colour per game
    let minLimits = map maxColourCount $ map f parsedGameRecords
          where
            f (Just (CubeGame _ trials, _)) = CubeGame 0 trials
            f Nothing = CubeGame 0 []
    let cubesetPower = map (\cc -> product [x | ColourCount _ x <- cc]) minLimits
    let sumOfCubesetPower = sum cubesetPower
    putStrLn $ "Sum of Cubeset Powers: " ++ show sumOfCubesetPower
