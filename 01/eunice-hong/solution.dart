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

// Map of digit strings to their integer values
const digits = {
  '1': 1,
  '2': 2,
  '3': 3,
  '4': 4,
  '5': 5,
  '6': 6,
  '7': 7,
  '8': 8,
  '9': 9,
  'one': 1,
  'two': 2,
  'three': 3,
  'four': 4,
  'five': 5,
  'six': 6,
  'seven': 7,
  'eight': 8,
  'nine': 9
};

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  int sum = 0;
  final lines = input.split('\n').where((line) => line.isNotEmpty);

  final regex = RegExp(r'([123456789])');
  for (String line in lines) {
    final matches = regex.allMatches(line);
    final tensKey = matches.elementAt(0).group(0);
    final onesKey = matches.elementAt(matches.length - 1).group(0);
    final tens = digits[tensKey] ?? 0;
    final ones = digits[onesKey] ?? 0;
    sum += tens * 10 + ones;
  }

  return sum.toString();
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  return input
      .split('\n') // Split into lines
      .where((line) => line.isNotEmpty) // Remove empty lines
      .map((line) {
        var charStartIndex = 0;
        var charEndIndex = line.length;
        int? tens;
        int? ones;
        String head = line.substring(charStartIndex);
        String tail = line.substring(0, charEndIndex);
        do {
          // If we found a digit at the start or end of the line,
          // and we haven't already found a digit for that position,
          // set the digit for that position.
          for (String digitString in digits.keys) {
            if (head.startsWith(digitString) && tens == null) {
              tens = digits[digitString];
            }
            if (tail.endsWith(digitString) && ones == null) {
              ones = digits[digitString];
            }
          }

          // If we didn't find a digit at the start or end of the line,
          if (tens == null) {
            charStartIndex++;
          }
          if (ones == null) {
            charEndIndex--;
          }
          head = line.substring(charStartIndex);
          tail = line.substring(0, charEndIndex);
        } while (head.isNotEmpty &&
            tail.isNotEmpty &&
            (tens == null || ones == null));
        return (tens ?? 0) * 10 + (ones ?? 0);
      })
      .fold(0, (sum, calibration) => sum + calibration)
      .toString();
}

void printSolutions(String input) {
  print('Part 1: ${solution1(input)}');
  print('Part 2: ${solution2(input)}');
}
