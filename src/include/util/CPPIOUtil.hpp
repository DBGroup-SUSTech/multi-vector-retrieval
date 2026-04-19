//
// Created by Administrator on 2025/5/19.
//

#ifndef CPPIOUTIL_HPP
#define CPPIOUTIL_HPP
#include <vector>
#include <string>

namespace VectorSetSearch {
std::vector<uint32_t> readQueryID(const std::string username, const std::string dataset) {
  char location[256];
  sprintf(location, "/home/%s/Dataset/multi-vector-retrieval/RawData/%s/document/queries.dev.tsv",
          username.c_str(), dataset.c_str());

  std::vector<uint32_t> qID_l;
  std::ifstream file(location);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file!" << std::endl;
    exit(1);
  }

  std::string line;
  while (getline(file, line)) {
    if (line.empty()) continue;

    size_t tab_pos = line.find('\t');
    if (tab_pos == std::string::npos) {
      std::cerr << "Warning: Invalid format in line: " << line << std::endl;
      continue;
    }

    std::string id = line.substr(0, tab_pos);
    qID_l.push_back(stoi(id));
  }

  return qID_l;
}
}
#endif //CPPIOUTIL_HPP
