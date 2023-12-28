module Main where

import BaseParser
import Control.Applicative
import Data.Char
import qualified Data.Map as M

-- Define Direction type
data Direction = R | L deriving (Show, Eq)

-- Parse "RLRL" into [R, L, R, L]
parseDirection :: Parser [Direction]
parseDirection = many $ (charP 'R' *> pure R) <|> (charP 'L' *> pure L)

parseNodeNames :: Parser String
parseNodeNames = some (letterP <|> (intToDigit <$> digitP))

-- Parse "AAA = (BBB, CCC)" into ("AAA", "BBB", "CCC"), applicative parser
parseMap :: Parser (String, String, String)
parseMap =
    (,,)
        <$> (parseNodeNames <* ws)
        <*> (charP '=' *> ws *> charP '(' *> parseNodeNames <* charP ',' <* ws)
        <*> (parseNodeNames <* charP ')')

-- Create Left/Right Direction Map
createLeftMap :: [(String, String, String)] -> M.Map String String
createLeftMap = foldl (\acc (k, v, _) -> M.insert k v acc) M.empty

createRightMap :: [(String, String, String)] -> M.Map String String
createRightMap = foldl (\acc (k, _, v) -> M.insert k v acc) M.empty

-- Solutions
part1Solution :: [Direction] -> [(String, String, String)] -> String -> (String -> Bool) -> Int
part1Solution directions network startNode endCond = length result
  where
    -- Start from 'AAA', go through network in Direction list order
    -- Stop when 'ZZZ' is reached
    leftMap = createLeftMap network
    rightMap = createRightMap network

    -- Traverse network
    traverseNetwork :: String -> Direction -> String
    traverseNetwork node dir
        | dir == R = rightMap M.! node
        | dir == L = leftMap M.! node

    result = takeWhile (endCond) $ scanl (\acc x -> traverseNetwork acc x) startNode (cycle directions)

part2Solution :: [Direction] -> [(String, String, String)] -> Int
part2Solution directions network = result
  where
    -- Find startNodes that ends in 'A', go through network in Direction list order, find length of each startNode to their respective ends
    -- Find LCM of all lengths, loops are cyclic
    leftMap = createLeftMap network
    rightMap = createRightMap network

    -- Find Start Nodes
    startNodes = (\(a, _, _) -> a) <$> filter (\(n, _, _) -> last n == 'A') network

    -- Recursive Least Common Multiplier through list
    recursiveLCM [] = 1
    recursiveLCM (x : xs) = lcm x (recursiveLCM xs)

    -- Use Part 1 to find length of each startNode to their respective ends
    result = recursiveLCM $ map (\sn -> part1Solution directions network sn (\n -> last n /= 'Z')) startNodes

-- Main Function
main :: IO ()
main = do
    -- Read map
    input <- readFile "08/input.txt"
    let (directions : _ : network) = lines input

    -- Parse direction
    let parsedD = fst . fromMaybe $ runParser parseDirection directions

    -- Parse map
    let parsedM = map (fst . fromMaybe . runParser parseMap) $ network

    -- Part 1
    let part1 = part1Solution parsedD parsedM "AAA" (/= "ZZZ")
    putStrLn $ "It takes " ++ show part1 ++ " steps to reach ZZZ"

    -- Part 2
    let part2 = part2Solution parsedD parsedM
    putStrLn $ "It takes " ++ show part2 ++ " steps for all startNodes to reach a node that ends with Z"

    print "Done"
