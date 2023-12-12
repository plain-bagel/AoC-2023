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

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  return solution(input, 1);
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  return solution(input, 999999);
}

String solution(String input, int increment) {
  int sum = 0;
  final universe = input.split('\n').map((line) => line.split('')).toList();
  final galaxies = [];

  final expandedRows = <int>[];
  List.generate(universe.length, (index) {
    final line = universe[index];
    if (line.toSet().length == 1) {
      expandedRows.add(index);
    }
    return line;
  });

  final expandedColumns = <int>[];
  for (int i = 0; i < universe.length; i++) {
    final column = universe.map((line) => line[i]).toList();
    if (column.toSet().length == 1) {
      expandedColumns.add(i);
    }
  }

  for (int i = 0; i < universe.length; i++) {
    for (int j = 0; j < universe[i].length; j++) {
      if (universe[j][i] == '#') {
        galaxies.add([i, j]);
      }
    }
  }
  for (int a = 0; a < galaxies.length; a++) {
    final galaxyA = galaxies[a];
    for (int b = a + 1; b < galaxies.length; b++) {
      final galaxyB = galaxies[b];
      sum += getDistance(
        expandedRows,
        expandedColumns,
        galaxyA[0],
        galaxyA[1],
        galaxyB[0],
        galaxyB[1],
        increment,
      );
    }
  }
  return sum.toString();
}

void printSolutions(String input) {
  print('Part 1: ${solution1(input)}');
  print('Part 2: ${solution2(input)}');
}

int getDistance(
  List<int> expandedRows,
  List<int> expandedColumns,
  int x1,
  int y1,
  int x2,
  int y2,
  int increment,
) {
  final minX = x1 < x2 ? x1 : x2;
  final maxX = minX == x1 ? x2 : x1;
  final width = (x1 - x2).abs() +
      expandedColumns.where((column) => minX < column && column < maxX).length *
          increment;
  final minY = y1 < y2 ? y1 : y2;
  final maxY = minY == y1 ? y2 : y1;
  final height = (y1 - y2).abs() +
      expandedRows.where((row) => minY < row && row < maxY).length * increment;
  return width + height;
}
