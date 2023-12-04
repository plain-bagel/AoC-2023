import 'dart:io';
import 'dart:math';

void main(List<String> arguments) {
  if (arguments.length != 1) {
    print('Usage: dart solution.dart <filename>');
    return;
  }

  String fileName = arguments[0];

  // Get the current directory path
  String currentDirectory = Directory.current.path;

  // File path of the specified file in the current directory
  String filePath = '$currentDirectory/$fileName';

  // Read the contents of the file
  try {
    final file = File(filePath);
    if (file.existsSync()) {
      // Read file contents
      String contents = file.readAsStringSync();

      // Print solutions
      printSolutions(contents);
    } else {
      print('File not found: $fileName');
    }
  } catch (e) {
    print('Error reading file: $e');
  }
}

/// Regular expression to match the game id
final idRegex = RegExp(r'(?<=Game\s)\d+');

/// Regular expressions to match the number of Red color
final redRegex = RegExp(r'\d+(?=\sred)');

/// Regular expressions to match the number of Green color
final greenRegex = RegExp(r'\d+(?=\sgreen)');

/// Regular expressions to match the number of Blue color
final blueRegex = RegExp(r'\d+(?=\sblue)');

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  return input
      .split('\n')
      .map((game) {
        final numbersOfRed = getNumberOfColor(game, redRegex);
        final numbersOfGreen = getNumberOfColor(game, greenRegex);
        final numbersOfBlue = getNumberOfColor(game, blueRegex);
        if (numbersOfRed <= 12 && numbersOfGreen <= 13 && numbersOfBlue <= 14) {
          final id = idRegex.firstMatch(game)?.group(0);
          return id != null ? int.parse(id) : 0;
        } else {
          return 0;
        }
      })
      .fold(0, (previousValue, element) => previousValue + element)
      .toString();
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  return input
      .split('\n')
      .map((game) {
        final numbersOfRed = getNumberOfColor(game, redRegex);
        final numbersOfGreen = getNumberOfColor(game, greenRegex);
        final numbersOfBlue = getNumberOfColor(game, blueRegex);
        final redPower = numbersOfRed <= 0 ? 1 : numbersOfRed;
        final greenPower = numbersOfGreen <= 0 ? 1 : numbersOfGreen;
        final bluePower = numbersOfBlue <= 0 ? 1 : numbersOfBlue;
        return redPower * greenPower * bluePower;
      })
      .fold(0, (previousValue, element) => previousValue + element)
      .toString();
}

/// Get the number of color
///
/// [game] is the game string. ex. 'Game 1: 1 red, 2 green, 3 blue; 2 red, 4 blue'
///
/// [regex] is a regular expression to match the number of color
///
/// Returns the number of color
int getNumberOfColor(String game, RegExp regex) {
  return regex
      .allMatches(game)
      .map((match) => int.parse(match.group(0) ?? '0'))
      .fold(0, (previousValue, element) => max(previousValue, element));
}

void printSolutions(String input) {
  print('Part 1: ${solution1(input)}');
  print('Part 2: ${solution2(input)}');
}
