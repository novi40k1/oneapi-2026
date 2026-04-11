#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
  std::map<std::string, std::vector<std::string>> sorted_strings;
  std::string tmp;
  for (auto& row : dictionary) {
    tmp = row.first;
    std::sort(tmp.begin(), tmp.end());
    sorted_strings[tmp].push_back(row.first);
  }

  for (auto& [str, set] : dictionary) {
    tmp = str;
    std::sort(tmp.begin(), tmp.end());
    for (const auto& str_value : sorted_strings[tmp]) {
      if (str != str_value)
        set.push_back(str_value);
    }
  }

  for (auto& row : dictionary) {
    std::sort(row.second.begin(), row.second.end());
  }
}