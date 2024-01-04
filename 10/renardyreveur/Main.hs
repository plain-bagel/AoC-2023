module Main where

import BaseParser

import Data.Array
import qualified Data.Map as M

-- Define Data type for pipes
data PipeType = V | H | NE | NW | SW | SE | G | S deriving (Show, Eq)

-- PipeType Map
pipeTypeMap :: M.Map Char PipeType
pipeTypeMap =
    M.fromList
        [ ('|', V)
        , ('-', H)
        , ('.', G)
        , ('L', NE)
        , ('J', NW)
        , ('7', SW)
        , ('F', SE)
        , ('S', S)
        ]

-- Create PipeType Array from String
parsePipeMap :: String -> Array (Int, Int) PipeType
parsePipeMap str = listArray bounds [pipeTypeMap M.! col | row <- lines str, col <- row]
  where
    bounds = ((0, 0), (length (lines str) - 1, length (sHead (lines str)) - 1))

-- Processers

boundsFilter :: (Int, Int) -> (Int, Int) -> Bool
boundsFilter (row, col) (mRow, mCol) = row >= 0 && row <= mRow && col >= 0 && col <= mCol

nextCoordinates :: Array (Int, Int) PipeType -> (Int, Int) -> [(Int, Int)]
nextCoordinates pipeArray (row, col) =
    let pipeType = pipeArray ! (row, col)
     in case pipeType of
            -- Filter out coordinates that are out of bounds
            V -> [(row - 1, col), (row + 1, col)]
            H -> [(row, col - 1), (row, col + 1)]
            NE -> [(row - 1, col), (row, col + 1)]
            NW -> [(row - 1, col), (row, col - 1)]
            SW -> [(row + 1, col), (row, col - 1)]
            SE -> [(row + 1, col), (row, col + 1)]
            S -> [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            G -> []

-- Traverse Pipe
traversePipe :: Array (Int, Int) PipeType -> (Int, Int) -> [(Int, Int)] -> [(Int, Int)]
traversePipe pipeArray coord visited =
    let nextCoords = filter (\(r, c) -> boundsFilter (r, c) (snd $ bounds pipeArray)) $ nextCoordinates pipeArray coord
        validNextCoords = filter (\coord -> pipeArray ! coord /= G && pipeArray ! coord /= S && (coord `notElem` visited)) nextCoords
        newVisited = coord : visited
     in if null validNextCoords
            then newVisited
            else concatMap (\coord -> traversePipe pipeArray coord newVisited) validNextCoords


-- Main Function
main :: IO ()
main = do
    -- Read Pipe Map
    pipeMap <- readFile "10/input.txt"

    -- Parse Pipe Map
    let parsedPipeMap = parsePipeMap pipeMap

    -- Find Starting Coordinate (where 'S' is)
    let startCoord = sHead $ filter (\coord -> parsedPipeMap ! coord == S) $ indices parsedPipeMap
    
    -- Choose a direction and find the loop
    let nextCoords = nextCoordinates parsedPipeMap startCoord
    let nextCoords' = filter (\coord -> boundsFilter coord (snd $ bounds parsedPipeMap) && parsedPipeMap ! coord /= G) nextCoords

    -- Go through the pipes from starting coordinates
    let result = traversePipe parsedPipeMap (sHead nextCoords') []

    -- Part 1 (Full length of loop divided by 2 is furthest point)
    putStrLn $ "Steps it takes to point furthest away from starting point: " ++ show ((length result +1) `div` 2)
    
    
    -- Part 2 (Area of space enclosed by pipe loop)
    -- Go through each coordinate, check if it is enclosed by the loop -> Try Ray Casting (hope this is fast enough)
    -- Find the number of coordinates 'interior' to the loop = Area of space enclosed by loop