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

// Get the digits in a string
final digits = RegExp(r'\d+');

/// Solution for Puzzle Part 1
///
/// [input] is the contents of the input file
String solution1(String input) {
  final lines = input.split('\n');
  final times = digits
      .allMatches(lines[0])
      .map((m) => int.tryParse(m.group(0) ?? '0') ?? 0)
      .where((element) => 0 < element)
      .toList();

  final distances = digits
      .allMatches(lines[1])
      .map((m) => int.tryParse(m.group(0) ?? '0') ?? 0)
      .where((element) => 0 < element)
      .toList();

  return List.generate(
    times.length,
    (i) => _countChancePerRace(times[i], distances[i]),
  ).fold(1, (total, record) => total * record).toString();
}

/// Solution for Puzzle Part 2
///
/// [input] is the contents of the input file
String solution2(String input) {
  final lines = input.split('\n');
  final timeString =
      digits.allMatches(lines[0]).map((e) => e.group(0)).join('');
  final time = int.tryParse(timeString) ?? 0;
  final distanceString =
      digits.allMatches(lines[1]).map((e) => e.group(0)).join('');
  final distance = int.tryParse(distanceString) ?? 0;
  int chance = _countChancePerRace(time, distance);
  return chance.toString();
}

/// Count chances to win per race.
int _countChancePerRace(int time, int distance) {
  return List.generate(time + 1, (index) => index).where((int velocity) {
    int myDistance = velocity * (time - velocity);
    return distance < myDistance;
  }).length;
}

void printSolutions(String input) {
  print('Part 1: ${solution1(input)}');
  print('Part 2: ${solution2(input)}');
}
