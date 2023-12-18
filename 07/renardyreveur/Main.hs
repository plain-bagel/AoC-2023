module Main where

import BaseParser
import Data.List (nub, sort)

-- Create Data Structure
data HandType = HighCard | OnePair | TwoPair | ThreeofAKind | FullHouse | FourofAKind | FiveofAKind deriving (Show, Eq, Ord)

-- data CardType = N | T | J | Q | K | A deriving (Show, Eq, Ord) --- Part 1
data CardType = J | N | T | Q | K | A deriving (Show, Eq, Ord) -- Part 2

data Hand = Hand
    { cards :: String
    , cardtype :: [CardType]
    , bid :: Int
    , handtype :: HandType
    }
    deriving (Show)
instance Eq Hand where
    (==) (Hand s1 _ _ _) (Hand s2 _ _ _) = s1 == s2
instance Ord Hand where
    compare h1 h2 = compareHand h1 h2

-- Classify Hands into type
classifyHand :: String -> HandType
classifyHand h =
    case nub h of
        [c] -> FiveofAKind
        [c1, c2] -> if any (\ct -> (length (filter (== ct) h)) == 4) [c1, c2] then FourofAKind else FullHouse
        [c1, c2, c3] -> if any (\ct -> (length (filter (== ct) h)) == 3) [c1, c2, c3] then ThreeofAKind else TwoPair
        [c1, c2, c3, c4] -> if any (\ct -> (length (filter (== ct) h)) == 2) [c1, c2, c3, c4] then OnePair else HighCard
        _ -> HighCard

-- Classify Cards into type
classifyCard :: Char -> CardType
classifyCard c =
    case c of
        'T' -> T
        'J' -> J
        'Q' -> Q
        'K' -> K
        'A' -> A
        _ -> N

-- Compare single card
compareCard :: (CardType, Char) -> (CardType, Char) -> Ordering
compareCard (c1, s1) (c2, s2)
    | c1 > c2 = GT
    | c1 < c2 = LT
    | otherwise =
        -- 2 cases: either both are N or both are same non-N card type
        case (c1, c2) of
            (N, N) -> compare s1 s2
            _ -> EQ

-- Compare hands
compareHand :: Hand -> Hand -> Ordering
compareHand (Hand [] [] b1 h1) (Hand [] [] b2 h2) = EQ
compareHand (Hand (s1 : ss1) (c1 : cc1) b1 h1) (Hand (s2 : ss2) (c2 : cc2) b2 h2)
    | (h1 > h2) = GT
    | (h1 < h2) = LT
    | otherwise =
        case compareCard (c1, s1) (c2, s2) of
            GT -> GT
            EQ -> compareHand (Hand ss1 cc1 b1 h1) (Hand ss2 cc2 b2 h2)
            LT -> LT

-- Parsers
parseHand :: Parser (String, Int)
parseHand = (,) <$> stringLiteralP <* ws <*> intP

-- Main
main :: IO ()
main = do
    -- Read Input file
    hands <- readFile "07/input.txt"

    -- Card hands
    let cardHands = map (fst . fromMaybe . runParser parseHand) (lines hands)

    -- Classify hands
    let classifiedHands = map (\(h, b) -> Hand h (map classifyCard h) b (classifyHand h)) cardHands

    -- Sort hands from weakest to strongest
    let sortedHands = sort classifiedHands

    -- Total Winning Bid
    let totalBid = sum $ [rank * b | (rank, b) <- zip [1 .. length cardHands] (map bid sortedHands)]

    -- Print total winning bid
    putStrLn $ "Total Winning Bid: " ++ show totalBid

    -- Part2
    let subs = "23456789TQKA"

    -- First Try: I tried promoting HandTypes by the number of J's in the hand, realized it wouldn't work as there isn't a single promotion order
    -- As soon as there are more than three (JJJ), promotion order is not clear

    -- Try replacing 'J' with each character in subs, get best hand type (thankful for the Ord instance of HandType > <)
    let cardHands2 = map (\(h, _) -> maximum [classifyHand $ map (\c -> if c == 'J' then s else c) h | s <- subs]) cardHands

    -- Newly classified hands
    let classifiedHands2 = map (\((s, b), h) -> Hand s (map classifyCard s) b h) $ zip cardHands cardHands2

    -- Sort hands from weakest to strongest
    let sortedHands2 = sort classifiedHands2

    -- Total Winning Bid
    let totalBid2 = sum $ [rank * b | (rank, b) <- zip [1 .. length cardHands] (map bid sortedHands2)]

    -- Print total winning bid
    putStrLn $ "Total Winning Bid with Joker J: " ++ show totalBid2
