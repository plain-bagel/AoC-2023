module Main where

import BaseParser
import Control.Applicative

-- Parsers
parseSequence :: Parser [Int]
parseSequence = sepBy ws (intP <|> intNegP)

-- Get Difference Sequence
getDiffs :: [Int] -> [Int]
getDiffs [] = []
getDiffs [x] = []
getDiffs (x : y : xs) = (y - x) : getDiffs (y : xs)

-- Solutions
part1Solution :: [[Int]] -> Int
part1Solution history = sum nexts
  where
    -- Recursively run getDiffs until the sum of the sequence is 0, then return all intermediate sequences
    -- Find next in sequence by adding the last element of the last sequence to the last element of the next sequence, all the way to the orignal sequence
    allDiffs = map (\y -> takeWhile (any (/= 0)) (iterate getDiffs y)) history
    nexts = map (\x -> foldl (\acc y -> acc + last y) 0 (reverse x)) $ allDiffs

part2Solution :: [[Int]] -> Int
part2Solution history = sum previouses
  where
    -- Just do it backwards
    allDiffs = map (\y -> takeWhile (any (/= 0)) (iterate getDiffs y)) history
    previouses = map (\x -> foldl (\acc y -> (sHead y) - acc) 0 (reverse x)) $ allDiffs

-- Main
main :: IO ()
main = do
    -- Read input file
    history <- readFile "09/input.txt"

    -- Parse each line as a sequence of integers
    let parsed = map (fst . fromMaybe . runParser parseSequence) (lines history)

    -- Part 1
    putStrLn $ "Part 1: " ++ show (part1Solution parsed)

    -- Part 2
    putStrLn $ "Part 2: " ++ show (part2Solution parsed)
