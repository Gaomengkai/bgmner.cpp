//
// Created by gaome on 2024/5/11.
//

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
[[maybe_unused]] Ort::Value
BasicTokenizer::tokenize(const std::wstring &wtext) {
    auto splitedWChar = splitByOneWChar(wtext);
    unsigned long long int text_len = splitedWChar.size();
    auto token_ids = new int[text_len + 2];
    token_ids[0] = m_word2idx.at(utf8decode("[CLS]"));
    int i = 1;
    for (const auto &token : splitByOneWChar(wtext)) {
        auto it = m_word2idx.find(token);
        if (it != m_word2idx.end()) {
            token_ids[i] = m_word2idx[token];
        } else {
            token_ids[i] = m_word2idx.at(L"[UNK]");
        }
        i++;
    }
    token_ids[i] = m_word2idx.at(utf8decode("[SEP]"));
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    unsigned long long int size = text_len + 2;
    auto *shape = new long long[1];
    shape[0] = size;
    Ort::Value token_ids_tensor =
        Ort::Value::CreateTensor<int>(memory_info, token_ids, size, shape, 1);
    return std::move(token_ids_tensor);
}



std::vector<int> BasicTokenizer::tokenize_vectori32(const std::wstring &wtext) {
    auto splitedWChar = splitByOneWChar(wtext);
    std::vector<int> token_ids;
    token_ids.push_back(m_word2idx.at(utf8decode("[CLS]")));
    for (const auto &ctoken : splitByOneWChar(wtext)) {
        wchar_t p = ctoken[0];
        // lower
        if (p >= 'A' && p <= 'Z') {
            p += 0x20;
        }
        std::wstring token = std::wstring(1, p);
        auto it = m_word2idx.find(token);
        if (it != m_word2idx.end()) {
            token_ids.push_back((*it).second);
        } else {
            token_ids.push_back(m_word2idx.at(L"[UNK]"));
        }
    }
    token_ids.push_back(m_word2idx.at(utf8decode("[SEP]")));
    return token_ids;
}
void BasicTokenizer::loadDict(){
    std::ifstream ifs{m_vocab_file};
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open " << m_vocab_file << std::endl;
        exit(1);
    }
    int idx = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        std::wstring wline = utf8decode(line);
        m_word2idx[wline] = idx;
        m_idx2word[idx] = wline;
        idx++;
    }
}

} // namespace bgmner.cpp