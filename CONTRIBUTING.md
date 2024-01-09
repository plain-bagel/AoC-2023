# AoC 참여 방법

한 해를 마무리하는 12월, [Advent of Code][aoc-official]에 참여하고 싶은 베이글러를 위한 안내서입니다.

## How to participate in AoC?

### Prerequisites

#### 1. Be a BAGELER

1. [플레인베이글][pbagel-career]을 확인하세요.
2. 원하는 직무를 선택하고, `지원하기` 를 눌러 베이글러가 되세요. 🥯🧑‍🍳

> [!IMPORTANT]
> 베이글러의 이야기가 궁금하다면, [베이글러 이야기][pbagel-blog]을 확인하세요.

#### 2. Sign up for Advent of Code

1. Go to [Advent of Code][aoc-official]
2. 상단 `[ Log In ]` 버튼을 클릭합니다.
3. 아래에 다음과 같은 메시지가 뜨면, 사용하시는 서비스의 계정을 활용해서 로그인 합니다.

   ```text
   To play, please identify yourself via one of these services:
   
   [GitHub] [Google] [Twitter] [Reddit] - [How Does Auth Work?]
   ```

#### 3. Join the leaderboard

1. 로그인이 완료되면, 상단에서 `[ Leaderboard ]` 버튼을 클릭합니다.
2. 아래에 다음과 같은 메시지가 뜬다면, `[ Join Private Leaderboard ]` 버튼을 클릭합니다.

    ```text
    Nothing to show on the leaderboard... yet.
    If you like, you can set up or join a **[Private Leaderboard]** in advance.
    ```
3. 아래에 다음과 같은 메시지가 뜨면, 입력창에 코드를 입력합니다. (코드는 슬랙 채널에서 확인 가능합니다.)

    ```text
    You can join a private leaderboard by entering its join code here:
    <입력창>
    ```

### Solve the puzzles

2023년 12월 1일부터 12월 25일까지 [Advent of Code][aoc-official]에 매일 UTC+9 14:00 에 새로운 문제가 출제됩니다.

원하는 프로그래밍 언어를 사용해서 문제를 해결하세요.

### Share your solution

문제가 공개되면, 해당 날짜의 폴더에 `README.md`와 `input.txt` 파일이 생성됩니다.

* `README.md` 파일에는 문제가 설명되어 있습니다.
* `input.txt` 파일에는 문제를 해결하기 위한 입력값이 저장되어 있습니다.

1. 해당 문제 브랜치로 이동합니다.

    ```shell
    git checkout -b day-00
    ```

2. 본인의 닉네임으로 된 브랜치를 생성합니다.

    ```shell
    git branch -b day-00/your-nickname
    ```

3. 해당 날짜의 폴더에 본인의 닉네임으로 된 폴더를 생성하고, 그 안에 문제 해결 방법을 공유해주세요.

   다음 명령어를 실행하여, 해당 날짜의 폴더에 본인의 닉네임으로 된 폴더를 생성할 수 있습니다.

    ```shell
    cd 00 && mkdir <YOUR_NICKNAME> && cd <YOUR_NICKNAME>
    ```

4. 본인의 닉네임으로 된 폴더에 문제 해결 방법을 Commit 해주세요.
5. 문제 해결 방법이 담긴 Commit을 [plain-bagel/AoC-2023]으로 Push 합니다.
6. Push 한 브랜치로 해당 날짜 문제에 대한 PR을 생성합니다.
7. PR은 모더레이터[@plain-bagel/aoc](https://github.com/orgs/plain-bagel/teams/aoc)가 확인 후 Merge합니다.

[aoc-official]: https://adventofcode.com

[pbagel-career]: https://pbagel.com/27040452

[pbagel-blog]: https://pbagel.com/27039550
