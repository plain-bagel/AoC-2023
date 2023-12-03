# 2023 Day 1: Trebuchet

## Prerequisites

* 하스켈 GHC 컴파일러와 기본적인 툴스택을 설치하는데 제일 쉬운 방법은 [GHCup](https://www.haskell.org/ghcup/) 입니다.

* 현재 `ghc-9.8.1`  버젼을 활용하고 있습니다.



## How to run

1. 퍼즐이 있는 디렉토리로 이동합니다.

    ```bash
    cd ~/AoC-2023/01
    ```
2. `trebuchet.hs` 를 실행합니다.
   
   `ghc` 컴파일러를 직접 사용하거나, `stack`, `cabal` 을 활용하여 패키징 하여 돌릴 수 있으나, `Main` 모듈을 안 쓰고 있기 때문에 그냥 `runhaskell` 로 실행합니다. (혹은 `ghci` 안에서 로드하여 실행해도 됩니다.)

    ```bash
    runhaskell renardyreveur/trebuchet.hs
    ```

## But I'm an AI Engineer...

> [!NOTE]
> `rule_learning` 폴더 내부의 AoC 2023 Day 1, Part 1 퍼즐을 ***딥러닝으로*** 푼 코드를 참고하세요.



## Tools Used

* Compiler and Evaluator Daemon: [ghcid](https://github.com/ndmitchell/ghcid)
* Formatter: [fourmolu](https://github.com/fourmolu/fourmolu)
* Training: [JAX](https://github.com/google/jax), no other dependencies