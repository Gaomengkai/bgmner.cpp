//
// Created by gaome on 2024/5/11.
//

#include "BGMNERInfer.h"
using Ort::Value;
namespace bgmner {
BGMNERInfer::BGMNERInfer(const string &model_path, const string &ner_args_path,
                         const string &vocab_path, bool enable_gpu)
    : enable_gpu(enable_gpu) {

    m_model_path = model_path;
    m_ner_args_path = ner_args_path;
    m_vocab_path = vocab_path;
    m_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    m_session_options = new Ort::SessionOptions();
    checkDeviceType();
    // auto cuda_options = OrtCUDAProviderOptions{};
    if (enable_gpu) {
        m_session_options->AppendExecutionProvider("DML", {}); // use DirectML
    }
    m_session = new Ort::Session{*m_env, s2ws(m_model_path).c_str(),
                                 *m_session_options};
    m_basic_tokenizer = new BasicTokenizer{m_vocab_path};

    // load model args
    auto args = ModelArgs{m_ner_args_path};
    m_input_max_len = args.getMaxSeqLen();
    m_entity_parser = new EntityParser{std::move(args)};
}
BGMNERInfer::~BGMNERInfer() {
    delete m_env;
    delete m_session_options;
    delete m_session;
    delete m_basic_tokenizer;
    delete m_entity_parser;
}
std::vector<output_int_type_t> BGMNERInfer::infer(wstring input_text) {
    std::vector<wstring> input_texts{std::move(input_text)};
    auto logits = infer(std::move(input_texts));
    return std::move(logits[0]);
}

std::tuple<Value *, Value *, size_t>
BGMNERInfer::prebuild(std::vector<wstring> input_texts) {
    size_t batch_size = input_texts.size();
    auto *data1_bare = new input_int_type_t [batch_size * m_input_max_len];
    auto *data2_bare = new input_int_type_t[batch_size * m_input_max_len];
    for (int batch = 0; batch < batch_size; batch++) {
        auto input_text = input_texts[batch];
        auto input_ids = m_basic_tokenizer->tokenize_vectori32(input_text);
        size_t input_len = input_ids.size();
        // copy
        std::copy(input_ids.data(), input_ids.data() + input_len,
                  data1_bare + batch * m_input_max_len);
        // padding
        for (size_t i = input_len; i < m_input_max_len; i++) {
            data1_bare[batch * m_input_max_len + i] = 0;
        }
        // mask
        auto true_len = min(input_len, m_input_max_len);
        std::fill(data2_bare + batch * m_input_max_len,
                  data2_bare + batch * m_input_max_len + true_len, 1);
        std::fill(data2_bare + batch * m_input_max_len + true_len,
                  data2_bare + (batch + 1) * m_input_max_len, 0);
    }
    auto *pData1 = new Ort::Value(Ort::Value::CreateTensor<input_int_type_t>(
        *getCpuMemoryInfo(), data1_bare, batch_size * m_input_max_len,
        new long long[2]{(long long)batch_size, (long long)m_input_max_len},
        2));
    auto *pData2 = new Ort::Value(Ort::Value::CreateTensor<input_int_type_t>(
        *getCpuMemoryInfo(), data2_bare, batch_size * m_input_max_len,
        new long long[2]{(long long)batch_size, (long long)m_input_max_len},
        2));
    return std::make_tuple(pData1, pData2, batch_size);
}

std::vector<std::vector<output_int_type_t>>
BGMNERInfer::infer(std::vector<wstring> input_texts) {
    auto [data1, data2, batch_size] = prebuild(input_texts);
    auto *input_tensors = new Ort::Value[2]{std::move(*data1), std::move(*data2)};
    auto output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names,
                                         input_tensors, 2, output_names, 1);
    auto output = std::move(output_tensors[0]); // (batchsize * max_seq_len)
    auto logits_data = output.GetTensorMutableData<output_int_type_t>();
    vector<vector<output_int_type_t>> ret;
    for (int i = 0; i < batch_size; i++) {
        vector<output_int_type_t> logits(logits_data + i * m_input_max_len,
                               logits_data + (i + 1) * m_input_max_len);
        ret.push_back(logits);
    }
    return ret;
}
Ort::MemoryInfo *BGMNERInfer::getCpuMemoryInfo() {
    static auto* memory_info = new Ort::MemoryInfo(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault));
    //Ort::MemoryInfo memory_info =
    //    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return memory_info;
}
Ort::Value BGMNERInfer::reshape(Ort::Value &&tensor, int64_t *shape,
                                size_t dim) {
    auto t = Ort::Value::CreateTensor<input_int_type_t>(
        *getCpuMemoryInfo(), tensor.GetTensorMutableData<input_int_type_t>(),
        tensor.GetTensorTypeAndShapeInfo().GetElementCount(), shape, dim);
    return std::move(t);
}
Ort::Value BGMNERInfer::createMask(size_t length, size_t max_length,
                                   bool invert) {
    auto mask = new input_int_type_t[max_length];
    if (length >= max_length) {
        length = max_length;
    }
    if (invert) {
        std::fill(mask, mask + max_length, 1);
        std::fill(mask + length, mask + max_length, 0);
    } else {
        std::fill(mask, mask + length, 1);
        std::fill(mask + length, mask + max_length, 0);
    }
    auto mask_tensor = Ort::Value::CreateTensor<input_int_type_t>(
        *getCpuMemoryInfo(), mask, max_length,
        new long long[1]{(long long)max_length}, 1);
    return std::move(mask_tensor);
}
BGMNERInfer::device_type BGMNERInfer::checkDeviceType() {
    auto aPro = Ort::GetAvailableProviders();
    for (const auto &x : aPro) {
        std::cout << x << " ";
    }
    return CPU;
}
template <typename T>
Ort::Value BGMNERInfer::resize(Ort::Value &&tensor, int64_t length, size_t dim,
                               T fill_value) {
    auto data = tensor.GetTensorMutableData<T>();
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto new_data_length = 1;
    for (long long i : shape) {
        new_data_length *= i;
    }
    new_data_length = new_data_length / shape[dim] * length;
    auto new_data = new T[new_data_length];
    if (shape.size() == 1) {
        std::copy(data, data + shape[0], new_data);
        std::fill(new_data + shape[0], new_data + new_data_length, fill_value);
    } else if (shape.size() == 2) {
        if (dim == 0) {
            for (size_t i = 0; i < shape[1]; i++) {
                std::copy(data + i * shape[0], data + (i + 1) * shape[0],
                          new_data + i * length);
                std::fill(new_data + i * length + shape[0],
                          new_data + (i + 1) * length, fill_value);
            }
        } else {
            for (size_t i = 0; i < shape[0]; i++) {
                std::copy(data + i, data + i + shape[1], new_data + i * length);
                std::fill(new_data + i * length + shape[1],
                          new_data + (i + 1) * length, fill_value);
            }
        }
    }
    auto new_shape = new int64_t[shape.size()];
    std::copy(shape.begin(), shape.end(), new_shape);
    new_shape[dim] = length;
    auto new_tensor = Ort::Value::CreateTensor<T>(
        *getCpuMemoryInfo(), new_data, new_data_length, new_shape, shape.size());
    return std::move(new_tensor);
}
} // namespace bgmner.cpp