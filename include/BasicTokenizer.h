//
// Created by gaome on 2024/5/11.
//

#ifndef BGMNER_BASICTOKENIZER_H
#define BGMNER_BASICTOKENIZER_H

#include <onnxruntime_cxx_api.h>
#include "utils.h"
#include <fstream>
#include <iostream>
namespace bgmner{
class BasicTokenizer {
  public:
    explicit BasicTokenizer(std::string vocab_file) {
        m_vocab_file = std::move(vocab_file);
        loadDict();
    }
    std::vector<int> tokenize_vectori32(const std::wstring &wtext);
    [[maybe_unused]] Ort::Value tokenize(const std::wstring &wtext);

  private:
    std::string m_vocab_file;
    std::unordered_map<std::wstring, int> m_word2idx;
    std::unordered_map<int, std::wstring> m_idx2word;

    void loadDict();

};
}

#endif // BGMNER_BASICTOKENIZER_H
