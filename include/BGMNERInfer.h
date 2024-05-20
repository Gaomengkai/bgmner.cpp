//
// Created by gaome on 2024/5/11.
//

#ifndef BGMNER_BGMNERINFER_H
#define BGMNER_BGMNERINFER_H

#include "BasicTokenizer.h"
#include "EntityParser.h"
#include "ModelArgs.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace bgmner {

using input_int_type_t = int64_t;
using output_int_type_t = int64_t;

class BGMNERInfer {
  public:
    BGMNERInfer(const string &model_path, const string &ner_args_path,
                const string &vocab_path, bool enable_gpu=false);

    ~BGMNERInfer();

    [[maybe_unused]] std::vector<int64_t> infer(wstring input_text);

    std::vector<std::vector<int64_t>> infer(std::vector<wstring> input_texts);

  private:
    string m_model_path, m_ner_args_path, m_vocab_path;
    Ort::Env *m_env;
    Ort::SessionOptions *m_session_options;
    Ort::Session *m_session;
    BasicTokenizer *m_basic_tokenizer;
    EntityParser *m_entity_parser;

    const char *input_names[2] = {"input_ids", "attention_mask"};
    const char *output_names[1] = {"logits"};
    int64_t m_input_max_len = 192;
    enum device_type {
        CPU, CUDA, DML
    };
    bool enable_gpu = false;

    static Ort::MemoryInfo *getCpuMemoryInfo();

    [[maybe_unused]] static Ort::Value reshape(Ort::Value &&tensor, int64_t *shape, size_t dim);
    [[maybe_unused]] static Ort::Value createMask(size_t length, size_t max_length,
                                           bool invert = false);
    // 将tensor的第dim维度填充到length。如果原有长度大于length，截断；如果小于length，填充
    // 仅处理一维和二维张量
    template <typename T>
    Ort::Value resize(Ort::Value &&tensor, int64_t length, size_t dim,
                      T fill_value);
    device_type checkDeviceType();
    std::tuple<Ort::Value *, Ort::Value *, size_t>
    prebuild(std::vector<wstring> input_texts);
};


} // namespace bgmner.cpp

#endif // BGMNER_BGMNERINFER_H
