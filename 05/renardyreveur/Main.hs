module Main where

import BaseParser

-- import Debug.Trace
import Data.List

-- ----- Define Almanac Map -----
data Dict = Dict
    { dest :: Int
    , source :: Int
    , range :: Int
    }
    deriving (Show)

-- ----- Almanac Parser -----
seednumP :: Parser [Int]
seednumP = stringP "seeds:" *> spaceP *> sepBy spaceP intP <* stringP "\n\n"

-- Parse map string to Dict (Source -> Dest) but using sepBy spaceP intP
mapP :: Parser Dict
mapP = Dict <$> (spaceP *> intP <* spaceP) <*> (spaceP *> intP <* spaceP) <*> (spaceP *> intP <* spaceP)

-- Parser Almanac Map (dest, source, range)
aToB :: String -> Parser [Dict]
aToB s = stringP s *> stringP "\n" *> sepBy (stringP "\n") mapP <* ws

-- Recursive chaining of aToB parsers
chainedAToB :: [String] -> Parser [[Dict]]
chainedAToB [] = pure []
chainedAToB (s : ss) = (:) <$> aToB s <*> chainedAToB ss

-- ----- Process Maps ------
-- Fold over list of Maps, starting with seed number
processMaps :: Int -> [[Dict]] -> [Int]
processMaps seed maps = reverse . snd $ foldl f (seed, [seed]) maps -- Fold over maps starting with seed number, return accumulator in order
  where
    -- If input is within the range of source, transform to dest, else return identity, recurse on rest of maps
    f (s, acc) (m : ms) =
        case source m <= s && s < source m + range m of
            True -> (s - source m + dest m, s - source m + dest m : acc) --
            False -> f (s, acc) ms -- Try different ruleset if not in range
            -- If no ruleset satisfies the input number, return Identity
    f (s, acc) [] = (s, s : acc)

--  ----- Part 2 ------
-- To Parser Seed Ranges (start, range)
pairUp :: [a] -> [(a, a)]
pairUp [] = []
pairUp [_] = []
pairUp (x : y : xs) = (x, y) : pairUp xs

-- Apply map(source to dest) on input
transform :: Dict -> Int -> Maybe Int
transform (Dict dest source range) input
    | (source <= input) && (input <= source + range - 1) = Just $ input - source + dest
    | otherwise = Nothing

-- Given full interval, and list of already found intervals, find missing intervals
findMissingIntervals :: (Int, Int) -> [(Int, Int)] -> [(Int, Int)]
findMissingIntervals (start, end) found =
    let sortedFound = sort found
        gaps = zip (start : map snd sortedFound) (map fst sortedFound ++ [end])
     in filter (uncurry (<)) gaps

-- Find intersection between source-dest map and actual given input
findSourceIntersection :: (Int, Int) -> Dict -> Maybe (Int, Int)
findSourceIntersection (start, end) (Dict dest source range) =
    let lb
            | start > source + range - 1 = -1
            | start >= source = start
            | otherwise = source
        ub
            | end > source + range - 1 = source + range - 1
            | end >= source = end
            | otherwise = -1
     in if lb == -1 || ub == -1 then Nothing else Just (lb, ub)

-- Step for finding seed ranges after applying map
seedRangeStep :: (Int, Int) -> [Dict] -> [(Int, Int)]
seedRangeStep (start, end) m = new_range
  where
    -- Find intersections
    sintersections = map (findSourceIntersection (start, end)) m

    -- Filter for valid intersections and apply transformation
    zipped = zip m sintersections
    valid_intersections = [(d, ie) | (d, Just ie) <- zipped]
    transformed = map (\(t, i) -> (fromMaybe $ transform t (fst i), fromMaybe $ transform t (snd i))) valid_intersections

    -- Find missing intervals
    leftover = findMissingIntervals (start, end) $ map snd valid_intersections

    -- New range
    new_range = leftover ++ transformed

-- Get next seed ranges from previous seed ranges, update seedInterval
applySeedRangeStep :: [(Int, Int)] -> [Dict] -> [(Int, Int)]
applySeedRangeStep seedInterval trsfm =
    concatMap (\(start, end) -> seedRangeStep (start, end) trsfm) seedInterval

-- Main Function
main :: IO ()
main = do
    -- Read alamanc file
    almanac <- readFile "05/input.txt"

    -- Parse seed number
    let (seeds, mapInfo) = fromMaybe $ runParser seednumP almanac

    -- Define map names
    let maps =
            [ "seed-to-soil map:"
            , "soil-to-fertilizer map:"
            , "fertilizer-to-water map:"
            , "water-to-light map:"
            , "light-to-temperature map:"
            , "temperature-to-humidity map:"
            , "humidity-to-location map:"
            ]

    -- Chain Applicative parsers together (Create list of 'Dict's per stage)
    let result = fst . fromMaybe $ runParser (chainedAToB maps) mapInfo

    -- Run the mapping process on each seed, solve part 1
    let res = map (\s -> processMaps s result) seeds
    putStrLn $ "Lowest location number of initial seeds: " ++ show (minimum $ map last res) ++ "\n"

    -- ----- Part 2 -----
    -- Test on first seed Bucket
    let seedInterval = [(x, x + y) | (x, y) <- pairUp seeds]

    -- Chain result of seedRangeStep through all maps, find final seed ranges
    let ff = foldl applySeedRangeStep seedInterval result

    -- Find minimum from either fst or snd
    putStrLn $ "Lowest location number of initial seed ranges: " ++ show (minimum $ map fst ff ++ map snd ff) ++ "\n"
