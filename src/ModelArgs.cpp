//
// Created by gaome on 2024/5/11.
//

#include "ModelArgs.h"
#include "BasicTokenizer.h"
#include "json.hpp"
#include "stdafx.h"
#include "utils.h"
#include <Windows.h>
#include <algorithm>
#include <fstream> // Include the necessary header file for ifstream
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bgmner {
ModelArgs::ModelArgs(const std::string &json_path) {
    std::ifstream ifs{json_path};
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open " << json_path << std::endl;
        exit(1);
    }
    nlohmann::json j;
    ifs >> j;
    auto label2id = j["label2id"]; // dict str->int
    for (auto it = label2id.begin(); it != label2id.end(); it++) {
        m_label2id[it.key()] = it.value();
        m_id2label[it.value()] = it.key();
    }
    auto max_len = j["max_seq_len"];
    m_max_seq_len = max_len;
}

std::string ModelArgs::getLabel(int32_t id) { return m_id2label.at(id); }
int64_t ModelArgs::getMaxSeqLen() const { return m_max_seq_len; }

int32_t ModelArgs::getLabelId(const std::string &label) {
        return m_label2id.at(label);
}
} // namespace bgmner.cpp