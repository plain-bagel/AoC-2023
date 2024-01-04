module Main where

import BaseParser (Parser (..), charP, runParser, spanP)
import Control.Applicative (Alternative (many, (<|>)))
import Data.Char (isDigit)
import Data.List (intersect, nub)
import Data.Maybe ()

-- Type Definitions and Data Constructors
type NumDots = Int
type Coord = (Int, Int)
data Part = PartNum String | PartSymbol String | PartDot Char deriving (Show, Eq)

-- Parser Functions
dotP :: Parser Char
dotP = charP '.'

symbolP :: Parser String
symbolP = spanP (\x -> not (x == '.' || isDigit x))

numP :: Parser String
numP = spanP isDigit

partP :: Parser Part
partP = PartNum <$> numP <|> PartSymbol <$> symbolP <|> PartDot <$> dotP

partsP :: Parser [Part]
partsP = many partP

-- Helper functions
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x : _) = Just x

safeTail :: [a] -> Maybe [a]
safeTail [] = Nothing
safeTail (_ : xs) = Just xs

fromMaybe :: Maybe a -> a
fromMaybe Nothing = error "Nothing"
fromMaybe (Just x) = x

sHead :: [c] -> c
sHead = fromMaybe . safeHead

sTail :: [a] -> [a]
sTail = fromMaybe . safeTail

processPart :: Part -> String
processPart (PartNum x) = x
processPart (PartSymbol x) = x
processPart (PartDot x) = [x]

replicatePart :: Part -> [(Part, Int)]
replicatePart (PartNum x) = zip (replicate (length x) (PartNum x)) [0 .. (length x) - 1]
replicatePart (PartSymbol x) = zip (replicate (length x) (PartSymbol x)) [0 .. (length x) - 1]
replicatePart (PartDot x) = zip (replicate 1 (PartDot x)) [0]

sameRow :: Coord -> Coord -> Bool
sameRow (x1, _) (x2, _) = x1 == x2

connectedCol :: Coord -> Int -> Coord -> Bool
connectedCol (_, y1) idx (_, y2) = any (\l -> y1 == (y2 - l)) [1 .. idx]

processPartNums :: [(((Part, Int), Coord), [Coord])] -> [(((Part, Int), Coord), [Coord])] -> [(((Part, Int), Coord), [Coord])]
processPartNums acc (x@(((p2, idx), coord), star) : xs) =
    if sameRow (snd . fst . sHead $ acc) coord && connectedCol (snd . fst . sHead $ acc) idx coord
        then processPartNums acc xs
        else processPartNums (x : acc) xs
processPartNums acc [] = acc

groupKeys :: [(Part, [Coord])] -> [[Part]]
groupKeys list = nub [[k1, k2] | (k1, pairs1) <- list, (k2, pairs2) <- list, k1 /= k2, not . null $ intersect pairs1 pairs2]

-- Main Function
main :: IO ()
main = do
    -- Read file and split into lines
    engineSchematic <- readFile "03/input.txt"
    let engineSchematicArray = lines engineSchematic

    -- Create a list of coordinates
    let coords = [(x, y) | x <- [0 .. (schlen - 1)], y <- [0 .. (schlen - 1)]]
          where
            schlen = length . sHead $ engineSchematicArray

    -- Parse the engine schematic, and fit the parts into the coordinates
    let parts = fst . fromMaybe . runParser partsP $ concat engineSchematicArray
    let partCoords = zip partsExpanded coords
          where
            partsExpanded = concat $ map replicatePart parts

    -- Get the PartNums, and their coordinates, and their neighbors
    let partNums = filter (\(x, _) -> case fst x of PartNum _ -> True; _ -> False) partCoords
    let partNumCoords = map snd partNums
    let partNumNeighbors = map (\(x, y) -> [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]) partNumCoords

    -- Get the coordinates of the PartSymbols
    let partSymbolCoords = map snd $ filter (\(x, _) -> case fst x of PartSymbol _ -> True; _ -> False) partCoords

    -- Get the intersection of the PartNum neighbors and PartSymbol coordinates
    let partNumNeighborsWithSymbols = map (\x -> filter (\y -> y `elem` partSymbolCoords) x) partNumNeighbors

    -- Filter out the PartNums that have no neighbors
    let filterdPartNums = filter (\(x, y) -> length y > 0) $ zip partNums partNumNeighborsWithSymbols

    -- Remove PartNum entry from the list if the next entry is the neighbor of the previous entry
    let finalPartNums = foldl (\acc (((p2, idx), coord2), _) -> if sameRowcond acc coord2 && previousColcond acc idx coord2 then acc else ((p2, idx), coord2) : acc) [firstPartNum] $ sTail filterdPartNums
          where
            sameRowcond a c = (fst . snd $ sHead a) == (fst c)
            previousColcond a i c = any (\x -> (x == True)) [(snd . snd $ sHead a) == (snd c - l) | l <- [1 .. i]]
            firstPartNum = fst . sHead $ filterdPartNums

    putStrLn $ "Sum of all the part numbers in the engine schematic: " ++ show (sum $ map (read . processPart . fst . fst) finalPartNums)

    -- Part 2
    -- Get PartSymbol Coords only for the '*' symbol
    let partSymbolCoordsStar = map snd $ filter (\(x, _) -> case fst x of PartSymbol "*" -> True; _ -> False) partCoords

    -- Get the intersection of the PartNum neighbors and PartSymbol coordinates
    let partNumNeighborsWithSymbolsStar = map (\x -> filter (\y -> y `elem` partSymbolCoordsStar) x) partNumNeighbors

    -- Filtered PartNums
    let filterdPartNumsStar = filter (\(x, y) -> length y > 0) $ zip partNums partNumNeighborsWithSymbolsStar

    -- Remove PartNum entry from the list if the next entry is the neighbor of the previous entry
    let finalPartNumsStar = [(fst . fst . fst $ x, snd x) | x <- processPartNums [sHead filterdPartNumsStar] $ sTail filterdPartNumsStar]

    -- Get the group PartNums that are connected to the '*' symbol
    let b = groupKeys finalPartNumsStar

    -- Sum the products and halve it as the pairs are counted twice
    let c = div (sum $ map (\x -> product $ map (read . processPart) x) b) 2

    putStrLn $ "Sum of all the gear ratios (product of pairs sharing adjacent '*') in the engine schematic: " ++ show c
