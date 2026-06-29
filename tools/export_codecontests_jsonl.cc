/* """Handoff note: C++ helper for exporting DeepMind CodeContests recordio/protobuf files to JSONL.
It is only needed when the Python loader cannot consume a pre-exported JSONL file; keep it aligned
with tools/BUILD and the contest_problem protobuf dependency.""" */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "contest_problem.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"

namespace {

using ::deepmind::code_contests::ContestProblem;

struct ExportLimits {
  int public_tests = 8;
  int private_tests = 8;
  int generated_tests = 0;
  int solutions = 3;
  int incorrect_solutions = 0;
};

std::string Basename(absl::string_view path) {
  const size_t slash = path.find_last_of('/');
  if (slash == absl::string_view::npos) return std::string(path);
  return std::string(path.substr(slash + 1));
}

std::string InferSplit(absl::string_view path) {
  const std::string text(path);
  if (text.find("valid") != std::string::npos) return "valid";
  if (text.find("test") != std::string::npos) return "test";
  if (text.find("train") != std::string::npos) return "train";
  return "unknown";
}

std::string SourceName(ContestProblem::Source source) {
  switch (source) {
    case ContestProblem::CODECHEF:
      return "CODECHEF";
    case ContestProblem::CODEFORCES:
      return "CODEFORCES";
    case ContestProblem::HACKEREARTH:
      return "HACKEREARTH";
    case ContestProblem::CODEJAM:
      return "CODEJAM";
    case ContestProblem::ATCODER:
      return "ATCODER";
    case ContestProblem::AIZU:
      return "AIZU";
    case ContestProblem::UNKNOWN_SOURCE:
    default:
      return "UNKNOWN_SOURCE";
  }
}

std::string DifficultyName(ContestProblem::Difficulty difficulty) {
  switch (difficulty) {
    case ContestProblem::EASY:
      return "easy";
    case ContestProblem::MEDIUM:
      return "medium";
    case ContestProblem::HARD:
      return "hard";
    case ContestProblem::HARDER:
      return "harder";
    case ContestProblem::HARDEST:
      return "hardest";
    case ContestProblem::EXTERNAL:
      return "external";
    case ContestProblem::A:
      return "A";
    case ContestProblem::B:
      return "B";
    case ContestProblem::C:
      return "C";
    case ContestProblem::D:
      return "D";
    case ContestProblem::E:
      return "E";
    case ContestProblem::F:
      return "F";
    case ContestProblem::G:
      return "G";
    case ContestProblem::H:
      return "H";
    case ContestProblem::I:
      return "I";
    case ContestProblem::J:
      return "J";
    case ContestProblem::K:
      return "K";
    case ContestProblem::L:
      return "L";
    case ContestProblem::M:
      return "M";
    case ContestProblem::N:
      return "N";
    case ContestProblem::O:
      return "O";
    case ContestProblem::P:
      return "P";
    case ContestProblem::Q:
      return "Q";
    case ContestProblem::R:
      return "R";
    case ContestProblem::S:
      return "S";
    case ContestProblem::T:
      return "T";
    case ContestProblem::U:
      return "U";
    case ContestProblem::V:
      return "V";
    case ContestProblem::UNKNOWN_DIFFICULTY:
    default:
      return "unknown";
  }
}

std::string LanguageName(ContestProblem::Solution::Language language) {
  switch (language) {
    case ContestProblem::Solution::PYTHON:
      return "python";
    case ContestProblem::Solution::CPP:
      return "cpp";
    case ContestProblem::Solution::PYTHON3:
      return "python";
    case ContestProblem::Solution::JAVA:
      return "java";
    case ContestProblem::Solution::UNKNOWN_LANGUAGE:
    default:
      return "unknown";
  }
}

void JsonString(std::ostream& out, absl::string_view value) {
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          const char* hex = "0123456789abcdef";
          out << "\\u00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
        } else {
          out << ch;
        }
    }
  }
  out << '"';
}

template <typename RepeatedTests>
void JsonTests(std::ostream& out, const RepeatedTests& tests, int limit) {
  const int count = limit < 0 ? tests.size() : std::min(tests.size(), limit);
  out << "{\"inputs\":[";
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    JsonString(out, tests.Get(i).input());
  }
  out << "],\"outputs\":[";
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    JsonString(out, tests.Get(i).output());
  }
  out << "]}";
}

template <typename RepeatedSolutions>
void JsonSolutions(std::ostream& out, const RepeatedSolutions& solutions,
                   int limit) {
  const int count =
      limit < 0 ? solutions.size() : std::min(solutions.size(), limit);
  out << "{\"language\":[";
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    JsonString(out, LanguageName(solutions.Get(i).language()));
  }
  out << "],\"solution\":[";
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    JsonString(out, solutions.Get(i).solution());
  }
  out << "]}";
}

void JsonStringArray(
    std::ostream& out,
    const google::protobuf::RepeatedPtrField<std::string>& values) {
  out << '[';
  for (int i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    JsonString(out, values.Get(i));
  }
  out << ']';
}

void WriteProblem(std::ostream& out, const ContestProblem& problem,
                  absl::string_view filename, int index,
                  const ExportLimits& limits) {
  out << '{';
  out << "\"id\":";
  JsonString(out, Basename(filename) + "_" + std::to_string(index));
  out << ",\"split\":";
  JsonString(out, InferSplit(filename));
  out << ",\"name\":";
  JsonString(out, problem.name());
  out << ",\"description\":";
  JsonString(out, problem.description());
  out << ",\"source\":";
  JsonString(out, SourceName(problem.source()));
  out << ",\"difficulty\":";
  JsonString(out, DifficultyName(problem.difficulty()));
  out << ",\"public_tests\":";
  JsonTests(out, problem.public_tests(), limits.public_tests);
  out << ",\"private_tests\":";
  JsonTests(out, problem.private_tests(), limits.private_tests);
  out << ",\"generated_tests\":";
  JsonTests(out, problem.generated_tests(), limits.generated_tests);
  out << ",\"public_tests_total\":" << problem.public_tests_size();
  out << ",\"private_tests_total\":" << problem.private_tests_size();
  out << ",\"generated_tests_total\":" << problem.generated_tests_size();
  out << ",\"solutions\":";
  JsonSolutions(out, problem.solutions(), limits.solutions);
  out << ",\"incorrect_solutions\":";
  JsonSolutions(out, problem.incorrect_solutions(), limits.incorrect_solutions);
  out << ",\"solutions_total\":" << problem.solutions_size();
  out << ",\"incorrect_solutions_total\":"
      << problem.incorrect_solutions_size();
  out << ",\"cf_contest_id\":" << problem.cf_contest_id();
  out << ",\"cf_index\":";
  JsonString(out, problem.cf_index());
  out << ",\"cf_points\":" << problem.cf_points();
  out << ",\"cf_rating\":" << problem.cf_rating();
  out << ",\"cf_tags\":";
  JsonStringArray(out, problem.cf_tags());
  out << ",\"is_description_translated\":"
      << (problem.is_description_translated() ? "true" : "false");
  out << ",\"untranslated_description\":";
  JsonString(out, problem.untranslated_description());
  out << ",\"time_limit_seconds\":";
  if (problem.has_time_limit()) {
    out << problem.time_limit().seconds() << "." << problem.time_limit().nanos();
  } else {
    out << "null";
  }
  out << ",\"memory_limit_bytes\":";
  if (problem.has_memory_limit_bytes()) {
    out << problem.memory_limit_bytes();
  } else {
    out << "null";
  }
  out << ",\"input_file\":";
  JsonString(out, problem.input_file());
  out << ",\"output_file\":";
  JsonString(out, problem.output_file());
  out << "}\n";
}

bool ExportFile(std::ostream& out, absl::string_view filename,
                const ExportLimits& limits) {
  riegeli::RecordReader<riegeli::FdReader<>> reader(
      std::forward_as_tuple(filename));
  ContestProblem problem;
  int index = 0;
  while (reader.ReadRecord(problem)) {
    WriteProblem(out, problem, filename, index, limits);
    problem.Clear();
    ++index;
  }
  const bool ok = reader.Close();
  if (!ok) {
    std::cerr << "failed to read " << filename << "\n";
  }
  return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string output_path;
  ExportLimits limits;
  std::vector<absl::string_view> filenames;
  for (int i = 1; i < argc; ++i) {
    const absl::string_view arg(argv[i]);
    if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--max-public-tests" && i + 1 < argc) {
      limits.public_tests = std::stoi(argv[++i]);
    } else if (arg == "--max-private-tests" && i + 1 < argc) {
      limits.private_tests = std::stoi(argv[++i]);
    } else if (arg == "--max-generated-tests" && i + 1 < argc) {
      limits.generated_tests = std::stoi(argv[++i]);
    } else if (arg == "--max-solutions" && i + 1 < argc) {
      limits.solutions = std::stoi(argv[++i]);
    } else if (arg == "--max-incorrect-solutions" && i + 1 < argc) {
      limits.incorrect_solutions = std::stoi(argv[++i]);
    } else {
      filenames.push_back(arg);
    }
  }
  if (output_path.empty() || filenames.empty()) {
    std::cerr << "usage: export_codecontests_jsonl --output OUT.jsonl "
              << "[--max-public-tests N] [--max-private-tests N] "
              << "[--max-generated-tests N] [--max-solutions N] "
              << "[--max-incorrect-solutions N] "
              << "code_contests_*.riegeli...\n";
    return 2;
  }
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "failed to open output: " << output_path << "\n";
    return 2;
  }
  bool ok = true;
  for (const absl::string_view filename : filenames) {
    ok = ExportFile(out, filename, limits) && ok;
  }
  return ok ? 0 : 1;
}
