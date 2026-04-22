#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t &dictionary)
{
    using StringPtr = const std::string *;

    std::unordered_map<std::string, std::vector<StringPtr>> groups;
    std::unordered_map<StringPtr, std::string> sorted_cache;

    for (auto &entry : dictionary)
    {
        const std::string &key = entry.first;
        std::string sorted = key;
        std::sort(sorted.begin(), sorted.end());

        sorted_cache[&key] = sorted;
        groups[sorted].push_back(&key);
    }

    for (auto &pair : groups)
    {
        std::vector<StringPtr> &vec = pair.second;
        std::sort(vec.begin(), vec.end(),
                  [](StringPtr a, StringPtr b)
                  { return *a > *b; });
    }

    for (auto &entry : dictionary)
    {
        const std::string &key = entry.first;
        const std::string &sorted = sorted_cache[&key];
        const std::vector<StringPtr> &group = groups[sorted];

        std::vector<std::string> &perms = entry.second;
        perms.clear();
        perms.reserve(group.size() - 1);

        for (StringPtr p : group)
        {
            if (p != &key)
            {
                perms.push_back(*p);
            }
        }
    }
}