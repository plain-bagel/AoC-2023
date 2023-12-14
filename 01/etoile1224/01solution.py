def sum_calibration_values_from_file(file_path):
    total_sum = 0
    with open(file_path, 'r') as file:
        for line in file:
            digits = [char for char in line if char.isdigit()]
            if len(digits) >= 2:
                # 첫 번째와 마지막 숫자 사용
                total_sum += int(digits[0] + digits[-1])
            elif len(digits) == 1:
                # 한 개의 숫자만 있는 경우, 이를 두 번 사용
                total_sum += int(digits[0] * 2)
            # 숫자가 없는 경우는 무시
    return total_sum
file_path = "/Users/kimsaebyol/AoC-2023-1/01/etoile1224/input.txt"
result = sum_calibration_values_from_file(file_path)
print(result)