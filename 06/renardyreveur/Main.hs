module Main where

import BaseParser

data Race = Race
    { time :: Int
    , distance :: Int
    }
    deriving (Show)

-- Parsers
parseTime :: Parser [Int]
parseTime = stringP "Time:" *> ws *> sepBy ws intP

parseDistance :: Parser [Int]
parseDistance = stringP "\nDistance:" *> ws *> sepBy ws intP

fromDigits :: [Int] -> Int
fromDigits = read . concatMap show

-- Helpers
getDistance :: Int -> Int -> Int
getDistance t h = (t - h) * h

getPossibleRaceDistances :: Race -> [Int]
getPossibleRaceDistances race = map (\i -> getDistance (time race) i) [0 .. time race]

-- Solve Quadratic equation (t-h)*h = d to find minimum hold
getMinHoldforTarget :: Int -> Int -> Int
getMinHoldforTarget t d = ceiling ((fromIntegral t + sqrt (fromIntegral t ^ 2 - 4 * fromIntegral d)) / 2)

-- Main function
main :: IO ()
main = do
    -- Read input
    input <- readFile "06/input.txt"

    -- Parse input
    let races = [Race t d | (t, d) <- zip times distances]
          where
            (times, dist_str) = fromMaybe $ runParser parseTime input
            (distances, _) = fromMaybe $ runParser parseDistance dist_str

    -- Get winning distances
    let distances = foldl (\acc x -> (filter (> distance x) $ getPossibleRaceDistances x) : acc) [] races
    putStrLn $ "Number of ways you could beat the record: " ++ show (product $ map length distances) ++ "\n"

    -- Part 2
    -- Parse input
    let oneRace = Race (fromDigits times) (fromDigits distances)
          where
            (times, dist_str) = fromMaybe $ runParser parseTime input
            (distances, _) = fromMaybe $ runParser parseDistance dist_str

    -- Get minimum hold
    let minHold = getMinHoldforTarget (time oneRace) (distance oneRace)
    putStrLn $ "Minimum hold to beat the record: " ++ show minHold

    -- Get number of winning distances (symmetric nature of (t-h)*h = d))
    let numWinningDistances = (time oneRace) - 2 * (time oneRace - minHold) - 1
    putStrLn $ "Number of ways you could beat the record with the new input: " ++ show numWinningDistances

    -- Alt - Bruteforce, doesn't take too long
    let winningDistances = length . filter (> distance oneRace) $ getPossibleRaceDistances oneRace
    putStrLn $ "Number of ways you could beat the record with the new input (bruteforce): " ++ show winningDistances
