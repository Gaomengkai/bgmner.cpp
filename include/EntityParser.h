//
// Created by gaome on 2024/5/11.
//

#ifndef BGMNER_ENTITYPARSER_H
#define BGMNER_ENTITYPARSER_H
#include "ModelArgs.h"
#include <string>
#include <vector>
using std::string;
using std::wstring;
using std::vector;

namespace bgmner {

class EntityParser {
  public:
    explicit EntityParser(const string &json_path) : m_model_args{json_path} {}
    explicit EntityParser(ModelArgs model_args) : m_model_args{std::move(model_args)} {}
    vector<string> seq2label(const int64_t * seq, size_t len);
    [[maybe_unused]] vector<string> seq2label(const vector<int64_t> &seq, size_t len);

    static vector<std::pair<string, std::pair<int32_t, int32_t>>>
    getEntities(vector<string> labelSeq);

  private:
    ModelArgs m_model_args;
};

} // namespace bgmner.cpp

#endif // BGMNER_ENTITYPARSER_H
