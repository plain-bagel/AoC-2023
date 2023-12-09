import 'dart:io';

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

void printSolutions(String input) {
  print('Part 1: ${solution1(input)}');
  print('Part 2: ${solution2(input)}');
}

final digitRegex = RegExp(r'-*\d+');

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  return input
      .split('\n')
      .fold(0, (acc, curr) => acc + getExtrapolatedTail(parseHistory(curr)))
      .toString();
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  return input
      .split('\n')
      .fold(0, (acc, curr) => acc + getExtrapolatedHead(parseHistory(curr)))
      .toString();
}

/// Parses the history of the given [line]
List<int> parseHistory(String line) => digitRegex
    .allMatches(line)
    .map((match) => match.group(0)!)
    .map((number) => int.tryParse(number) ?? 0)
    .toList();

/// Returns the extrapolated value of the tail of the given [values]
int getExtrapolatedTail(List<int> values) {
  if (values.toSet().length == 1) {
    return values[0];
  }
  final history = List.generate(
    values.length - 1,
    (index) => values[index + 1] - values[index],
  );
  return values[values.length - 1] + getExtrapolatedTail(history);
}

/// Returns the extrapolated value of the head of the given [values]
int getExtrapolatedHead(List<int> values) {
  if (values.toSet().length == 1) {
    return values[0];
  }
  final history = List.generate(
    values.length - 1,
    (index) => values[index + 1] - values[index],
  );
  return values[0] - getExtrapolatedHead(history);
}
