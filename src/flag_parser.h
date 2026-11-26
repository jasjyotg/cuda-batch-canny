// Copyright 2026 The cuda-batch-canny Authors.
//
// A small command-line flag parser in the spirit of gflags/absl::Flags, so
// the project has no third-party dependencies. Supported syntaxes:
//   --name=value   --name value   --bool_flag   --bool_flag=false   --help

#ifndef CUDA_BATCH_CANNY_SRC_FLAG_PARSER_H_
#define CUDA_BATCH_CANNY_SRC_FLAG_PARSER_H_

#include <string>
#include <vector>

namespace canny {

class FlagParser {
 public:
  explicit FlagParser(std::string description);

  FlagParser(const FlagParser&) = delete;
  FlagParser& operator=(const FlagParser&) = delete;

  // Registers a flag bound to `value`. The current contents of `value` are
  // used as the default shown in Usage().
  void AddString(const std::string& name, std::string* value,
                 const std::string& help);
  void AddInt(const std::string& name, int* value, const std::string& help);
  void AddFloat(const std::string& name, float* value, const std::string& help);
  void AddBool(const std::string& name, bool* value, const std::string& help);

  // Parses argv. Returns false and sets `error` on unknown flags, missing
  // values or malformed numbers. --help sets help_requested() and returns
  // true without touching other flags.
  bool Parse(int argc, char** argv, std::string* error);

  bool help_requested() const { return help_requested_; }

  // Returns a formatted help message listing every flag and its default.
  std::string Usage(const std::string& program_name) const;

 private:
  enum class Type { kString, kInt, kFloat, kBool };

  struct Flag {
    std::string name;
    Type type;
    void* value;
    std::string help;
    std::string default_text;
  };

  void Add(const std::string& name, Type type, void* value,
           const std::string& help, const std::string& default_text);
  Flag* Find(const std::string& name);
  static bool Assign(Flag* flag, const std::string& text, std::string* error);

  std::string description_;
  std::vector<Flag> flags_;
  bool help_requested_ = false;
};

}  // namespace canny

#endif  // CUDA_BATCH_CANNY_SRC_FLAG_PARSER_H_
