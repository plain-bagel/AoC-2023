module Day1 where

import qualified Data.Map as Map
import qualified Data.Text as T
import Data.Text.Internal.Search as Ts (indices)
import System.IO ()

-- Find the first number in a string
type Index = Int
findFirstNumber :: T.Text -> (Index, Int)
findFirstNumber text = findFirstNumber' text 0
  where
    findFirstNumber' :: T.Text -> Index -> (Index, Int)
    findFirstNumber' text index
        | T.null text = (index, 0)
        | T.head text `elem` ['0' .. '9'] = (index, read . T.unpack $ T.take 1 text)
        | otherwise = findFirstNumber' (T.tail text) (index + 1)

-- Find the last number in a string
findLastNumber :: T.Text -> (Index, Int)
findLastNumber = findFirstNumber . T.reverse

-- Part 2 --
-- Define numerical words map
numberMap :: Map.Map T.Text Int
numberMap = Map.fromList $ map (\x -> (T.pack . fst $ x, snd x)) [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)]

-- Lookup function for numerical words
lookupNumber :: T.Text -> Int
lookupNumber word = Map.findWithDefault 0 word numberMap

-- Helper functions for finding the first and last number/numerical word in a string
-- A much better way of doing this would be to use a Maybe type, and properly handle it with the minimum and maximum functions
getFirstElement :: [Int] -> Int
getFirstElement [] = 100000000
getFirstElement (x : xs) = x

getLastElement :: [Int] -> Int
getLastElement [] = 0
getLastElement xs = last xs

-- Find the first number/numerical word in a string
findFirstNumberOrWord :: T.Text -> Int
findFirstNumberOrWord text = firstOverall
  where
    -- First number that appears in the string
    firstNumber = findFirstNumber text
    -- First numerical word that appears in the string
    firstWord = minimum $ map (\x -> (getFirstElement (Ts.indices x text), lookupNumber x)) $ Map.keys numberMap
    -- Which one appears first overall
    firstOverall
        | fst firstNumber < fst firstWord = snd firstNumber
        | otherwise = snd firstWord

-- Find the last number/numerical word in a string
findLastNumberOrWord :: T.Text -> Int
findLastNumberOrWord text = lastOverall
  where
    -- Last number that appears in the string, subtract from length of text otherwise it will return the index from the end of the string
    lastNumber = ((T.length text) - (fst . findLastNumber $ text), snd . findLastNumber $ text)
    -- Last numerical word that appears in the string
    lastWord = maximum $ map (\x -> (getLastElement (Ts.indices x text), lookupNumber x)) $ Map.keys numberMap
    -- Which one appears last overall
    lastOverall
        | fst lastNumber > fst lastWord = snd lastNumber
        | otherwise = snd lastWord

-- Main function
main :: IO ()
main = do
    -- Read input file
    calibrationDocument <- readFile "01/input.txt"
    -- TODO: Handle exception

    -- Split the sting by lines, make a list of strings
    let calibrationList = lines calibrationDocument

    -- Map and accumulate results over calibrationList (Part 1)
    let result = foldl (\acc x -> acc + (10 * (snd . findFirstNumber $ T.pack x)) + (snd . findLastNumber $ T.pack x)) 0 calibrationList

    -- Map and accumulate results over calibrationList (Part 2)
    let snd_result = foldl (\acc x -> acc + (10 * (findFirstNumberOrWord $ T.pack x)) + (findLastNumberOrWord $ T.pack x)) 0 calibrationList

    -- Print Part 1 solution
    putStrLn $ "The sum of all calibration values are: " ++ (show result) ++ "\n"

    -- Print Part 2 solution
    putStrLn $ "The sum of all calibration values considering spelled out numbers are: " ++ show snd_result
