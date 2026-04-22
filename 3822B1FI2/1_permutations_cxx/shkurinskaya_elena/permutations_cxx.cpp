#include "permutations_cxx.h"

#include <algorithm>
#include <functional>

void Permutations(dictionary_t& dictionary) {
    // группируем ключи по отсортированному виду:
    // строки-анаграммы после sort дают одинаковую строку-ключ
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& pair : dictionary) {
        std::string sorted_key = pair.first;
        std::sort(sorted_key.begin(), sorted_key.end());
        groups[sorted_key].push_back(pair.first);
    }

    // проходим по словарю и для каждого ключа
    // берем его группу, исключаем саму строку
    for (auto& pair : dictionary) {
        std::string sorted_key = pair.first;
        std::sort(sorted_key.begin(), sorted_key.end());

        const auto& group = groups[sorted_key];
        for (const auto& s : group) {
            if (s != pair.first) {
                pair.second.push_back(s);
            }
        }

        // сортируем в обратном алфавитном порядке (как в примере)
        std::sort(pair.second.begin(), pair.second.end(),
                  std::greater<std::string>());
    }
}