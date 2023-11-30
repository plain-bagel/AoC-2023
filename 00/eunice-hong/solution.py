import sys

# 입력값 받기
with open(sys.argv[1], "r") as f:
    data = f.read().strip()

# Elf 별로 나누기
elves = data.split("\n\n")

# 각 Elf 별로 합 구하기
cals_per_elf = []

for elf in elves:
    calories = sum(int(x) for x in elf.split("\n"))
    cals_per_elf += [calories]

# 합이 큰 순서대로 정렬
cals_per_elf.sort(reverse=True)

# 첫번째 문제에 대한 답 출력
print(f"Part 1: {cals_per_elf[0]}")

# 두번째 문제에 대한 답 출력
print(f"Part 2: {sum(cals_per_elf[:3])}")
