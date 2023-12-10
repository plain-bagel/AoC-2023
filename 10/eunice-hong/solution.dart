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

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  // TODO Implement solution
  return input;
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  // TODO Implement solution
  return input;
}
