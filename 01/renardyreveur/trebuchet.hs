module Day1 where

import Data.List (isPrefixOf)
import qualified Data.Map as Map
import Data.Maybe (fromMaybe)
import System.IO ()

-- Type Aliases
type Index = Int
type Dictionary = Map.Map String Int

-- Dictionaries for numerical words
numberMap :: Dictionary
numberMap = Map.fromList [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)]

reverseNumberMap :: Dictionary
reverseNumberMap = Map.fromList [(reverse k, v) | (k, v) <- Map.toList numberMap]

-- Find the first number in a string
findFirstNumber :: String -> Dictionary -> Int
findFirstNumber [] _ = 0
findFirstNumber (x : xs) dict
    -- Return as soon as a number or a number word is found
    | search /= Nothing = fromMaybe 0 $ Map.lookup (fromMaybe "" search) dict
    | x `elem` ['0' .. '9'] = read [x] :: Int
    | otherwise = findFirstNumber xs dict -- Otherwise, recurse on the rest of the string
  where
    -- Helper function
    findKey :: [String] -> String -> Maybe String
    findKey [] _ = Nothing
    findKey (k : ks) s
        | k `isPrefixOf` s = Just k
        | otherwise = findKey ks s

    -- Search for a number word (as provided by the keys of the dictionary) from the beginning of the string
    search = findKey (Map.keys dict) (x : xs)

-- Main function
main :: IO ()
main = do
    -- Read input file
    calibrationDocument <- readFile "01/input.txt"
    -- TODO: Handle exception

    -- Split the sting by lines, make a list of strings
    let calibrationList = lines calibrationDocument

    -- Map and accumulate results over calibrationList (Part 1)
    let result = foldl (\acc x -> acc + (10 * (findFirstNumber x Map.empty)) + (findFirstNumber (reverse x) Map.empty)) 0 calibrationList

    -- Map and accumulate results over calibrationList (Part 2)
    let snd_result = foldl (\acc x -> acc + (10 * (findFirstNumber x numberMap)) + (findFirstNumber (reverse x) reverseNumberMap)) 0 calibrationList

    -- Print Part 1 solution
    putStrLn $ "The sum of all calibration values are: " ++ (show result) ++ "\n"

    -- Print Part 2 solution
    putStrLn $ "The sum of all calibration values considering spelled out numbers are: " ++ show snd_result
