import 'dart:core';
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
  final lines = input.split('\n');
  final directions = lines[0];
  final navigators = parseNavigators(lines.sublist(2));

  print('Part 1: ${solution1(directions, navigators)}');
  print('Part 2: ${solution2(directions, navigators)}');
}

final directionRegex = RegExp(r'([A-Z]){3}');

/// Least common multiple
int lcm(int a, int b) {
  return a * b ~/ a.gcd(b);
}

/// Solution for Puzzle Part 1
///
/// [directions] is the list of directions
///
/// [navigators] is the list of navigators
int solution1(String directions,
    List<MapEntry<String, MapEntry<String, String>>> navigators) {
  return getMoves(
    directions: directions,
    navigators: navigators,
    startPosition: 'AAA',
    isEnd: (position) => position == 'ZZZ',
  );
}

/// Solution for Puzzle Part 2
///
/// [directions] is the list of directions
///
/// [navigators] is the list of navigators
int solution2(String directions,
    List<MapEntry<String, MapEntry<String, String>>> navigators) {
  return navigators
      .where((navigator) => navigator.key.endsWith('A'))
      .map((navigator) {
    return getMoves(
      directions: directions,
      navigators: navigators,
      startPosition: navigator.key,
      isEnd: (position) => position.endsWith('Z'),
    );
  })
      // Get the least common multiple of all move counts
      .reduce((value, moves) => lcm(value, moves));
}

/// Get the number of moves required to reach the end
///
/// [directions] is the list of directions
///
/// [navigators] is the list of navigators
///
/// [startPosition] is the starting position
///
/// [isEnd] is the function to check if the current position is the end
int getMoves({
  required String directions,
  required List<MapEntry<String, MapEntry<String, String>>> navigators,
  required String startPosition,
  required bool Function(String) isEnd,
}) {
  int moves = 0;
  String currentPosition = startPosition;
  while (!isEnd(currentPosition)) {
    final direction = directions[moves % directions.length];
    final nextPositions = navigators
        .firstWhere((navigator) => navigator.key == currentPosition)
        .value;
    if (direction == 'L') {
      // Left
      currentPosition = nextPositions.key;
    } else {
      // Right
      currentPosition = nextPositions.value;
    }
    moves++;
  }
  return moves;
}

/// Parse the list of navigators
///
/// [lines] is the list of lines containing the navigators
List<MapEntry<String, MapEntry<String, String>>> parseNavigators(
  List<String> lines,
) =>
    lines
        .map((line) => directionRegex
            .allMatches(line)
            .map((match) => match.group(0) ?? '')
            .toList())
        .map(
          (parts) => MapEntry(
            // Key is the current position
            parts[0],
            MapEntry(
                // Key of the value is the next position when going left
                parts[1],
                // Value is the next position when going right
                parts[2]),
          ),
        )
        .toList();
