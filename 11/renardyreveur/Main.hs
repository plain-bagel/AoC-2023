module Main where

import BaseParser
import Control.Applicative
import Data.Array

-- Define type
data Cosmos = Space | Galaxy deriving (Show, Eq)

-- Parser for cosmos
parseCosmos :: Parser [Cosmos]
parseCosmos = many (charP '.' *> pure Space <|> charP '#' *> pure Galaxy)

-- Cosmos Expansion
isSpaceRow :: Array (Int, Int) Cosmos -> Int -> Bool
isSpaceRow starMap row = all (== Space) $ map (starMap !) [(row, col) | col <- [0 .. (snd . snd . bounds) starMap]]

isSpaceCol :: Array (Int, Int) Cosmos -> Int -> Bool
isSpaceCol starMap col = all (== Space) $ map (starMap !) [(row, col) | row <- [0 .. (fst . snd . bounds) starMap]]

-- find row numbers that are only composed of spaces
findSpaceRows :: Array (Int, Int) Cosmos -> [Int]
findSpaceRows starMap = filter (isSpaceRow starMap) [minRow .. maxRow]
  where
    ((minRow, _), (maxRow, _)) = bounds starMap

-- find col numbers that are only composed of spaces
findSpaceCols :: Array (Int, Int) Cosmos -> [Int]
findSpaceCols starMap = filter (isSpaceCol starMap) [minCol .. maxCol]
  where
    ((_, minCol), (_, maxCol)) = bounds starMap

getDistance :: (Int, Int) -> (Int, Int) -> [Int] -> [Int] -> Int -> Int
getDistance (r1, c1) (r2, c2) srow scol expandR =
    -- Find number of srow/scol elements between r1 and r2 / c1 and c2
    let
        rspaces = filter (\x -> x > min r1 r2 && x < max r1 r2) srow
        cspaces = filter (\x -> x > min c1 c2 && x < max c1 c2) scol
     in
        (expandR - 1) * (length rspaces + length cspaces) + abs (r1 - r2) + abs (c1 - c2)

-- Find galaxy pairs
combinations :: [a] -> [(a, a)]
combinations [] = []
combinations (x : xs) = map (\y -> (x, y)) xs ++ combinations xs

-- Main Function
main :: IO ()
main = do
    -- Read telescope data
    input <- readFile "11/input.txt"

    -- Parse telescope data
    let starMap = map (fst . fromMaybe . runParser parseCosmos) $ lines input

    -- Into Data.Array
    let starMapArray = listArray ((0, 0), (length starMap - 1, length (sHead starMap) - 1)) $ concat starMap

    -- Find all rows and columns that are only composed of spaces
    let spaceRows = findSpaceRows starMapArray
    let spaceCols = findSpaceCols starMapArray

    -- Find all Galaxies and their coordinates
    let galaxies = filter ((== Galaxy) . snd) $ assocs starMapArray

    -- Pair each galaxy up combination (g1, g2) (g2, g1) are the same
    let galaxyPairs = combinations galaxies

    -- Find distance between each pair
    let distance = map (\((g1, _), (g2, _)) -> getDistance g1 g2 spaceRows spaceCols 2) galaxyPairs
    putStrLn $ "Part 1: " ++ show (sum distance)

    -- Find distance between each pair (Part 2)
    let distance2 = map (\((g1, _), (g2, _)) -> getDistance g1 g2 spaceRows spaceCols 1000000) galaxyPairs
    putStrLn $ "Part 2: " ++ show (sum distance2)
