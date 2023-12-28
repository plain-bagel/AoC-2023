def is_set_possible(red, green, blue, red_limit, green_limit, blue_limit):
    return red <= red_limit and green <= green_limit and blue <= blue_limit

def extract_color_count(color_info):
    parts = color_info.strip().split(' ')
    return int(parts[0])

def process_games(file_path):
    with open(file_path, 'r') as file:
        games = file.readlines()

    possible_game_ids = []
    for game in games:
        game_id, game_content = game.split(':')
        game_id = int(game_id.strip().split(' ')[1])

        subsets = game_content.strip().split(';')
        game_possible = True
        for subset in subsets:
            colors = subset.strip().split(',')
            red = sum(extract_color_count(x) for x in colors if 'red' in x)
            green = sum(extract_color_count(x) for x in colors if 'green' in x)
            blue = sum(extract_color_count(x) for x in colors if 'blue' in x)

            if not is_set_possible(red, green, blue, 12, 13, 14):
                game_possible = False
                break

        if game_possible:
            possible_game_ids.append(game_id)

    return sum(possible_game_ids)

file_path = "/Users/kimsaebyol/AoC-2023/AoC-2023/02/input.txt"
result = process_games(file_path)
print(result)
