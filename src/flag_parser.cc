// Copyright 2026 The cuda-batch-canny Authors.

#include "flag_parser.h"

#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace canny {

FlagParser::FlagParser(std::string description)
    : description_(std::move(description)) {}

void FlagParser::AddString(const std::string& name, std::string* value,
                           const std::string& help) {
  Add(name, Type::kString, value, help, "\"" + *value + "\"");
}

void FlagParser::AddInt(const std::string& name, int* value,
                        const std::string& help) {
  Add(name, Type::kInt, value, help, std::to_string(*value));
}

void FlagParser::AddFloat(const std::string& name, float* value,
                          const std::string& help) {
  std::ostringstream text;
  text << *value;
  Add(name, Type::kFloat, value, help, text.str());
}

void FlagParser::AddBool(const std::string& name, bool* value,
                         const std::string& help) {
  Add(name, Type::kBool, value, help, *value ? "true" : "false");
}

void FlagParser::Add(const std::string& name, Type type, void* value,
                     const std::string& help, const std::string& default_text) {
  flags_.push_back(Flag{name, type, value, help, default_text});
}

FlagParser::Flag* FlagParser::Find(const std::string& name) {
  for (Flag& flag : flags_) {
    if (flag.name == name) return &flag;
  }
  return nullptr;
}

bool FlagParser::Assign(Flag* flag, const std::string& text,
                        std::string* error) {
  const char* begin = text.c_str();
  char* end = nullptr;
  errno = 0;
  switch (flag->type) {
    case Type::kString:
      *static_cast<std::string*>(flag->value) = text;
      return true;
    case Type::kInt: {
      const long parsed = std::strtol(begin, &end, 10);  // NOLINT(runtime/int)
      if (text.empty() || *end != '\0' || errno != 0 || parsed < -2147483647 ||
          parsed > 2147483647) {
        *error = "--" + flag->name + " expects an integer, got '" + text + "'";
        return false;
      }
      *static_cast<int*>(flag->value) = static_cast<int>(parsed);
      return true;
    }
    case Type::kFloat: {
      const float parsed = std::strtof(begin, &end);
      if (text.empty() || *end != '\0' || errno != 0) {
        *error = "--" + flag->name + " expects a number, got '" + text + "'";
        return false;
      }
      *static_cast<float*>(flag->value) = parsed;
      return true;
    }
    case Type::kBool:
      if (text == "true" || text == "1" || text == "yes") {
        *static_cast<bool*>(flag->value) = true;
        return true;
      }
      if (text == "false" || text == "0" || text == "no") {
        *static_cast<bool*>(flag->value) = false;
        return true;
      }
      *error = "--" + flag->name + " expects true/false, got '" + text + "'";
      return false;
  }
  return false;
}

bool FlagParser::Parse(int argc, char** argv, std::string* error) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      help_requested_ = true;
      return true;
    }
    if (arg.size() < 3 || arg.compare(0, 2, "--") != 0) {
      *error = "unexpected argument '" + arg + "' (flags start with --)";
      return false;
    }
    arg = arg.substr(2);
    std::string name = arg;
    std::string value;
    bool has_value = false;
    const size_t eq = arg.find('=');
    if (eq != std::string::npos) {
      name = arg.substr(0, eq);
      value = arg.substr(eq + 1);
      has_value = true;
    }
    Flag* flag = Find(name);
    if (flag == nullptr) {
      *error = "unknown flag --" + name + " (see --help)";
      return false;
    }
    if (!has_value) {
      if (flag->type == Type::kBool) {
        value = "true";
      } else if (i + 1 < argc) {
        value = argv[++i];
      } else {
        *error = "flag --" + name + " needs a value";
        return false;
      }
    }
    if (!Assign(flag, value, error)) return false;
  }
  return true;
}

std::string FlagParser::Usage(const std::string& program_name) const {
  std::ostringstream out;
  out << description_ << "\n\nUsage: " << program_name
      << " [flags]\n\nFlags:\n";
  for (const Flag& flag : flags_) {
    out << "  --" << flag.name << "\n      " << flag.help
        << " (default: " << flag.default_text << ")\n";
  }
  out << "  --help\n      Show this message.\n";
  return out.str();
}

}  // namespace canny
