#include "BGMNERInfer.h"
#include "BasicTokenizer.h"
#include "EntityParser.h"
#include "ModelArgs.h"
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

using namespace bgmner;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::wstring;
using namespace bgmner;

[[maybe_unused]] void printShape(const Ort::Value &tensor) {
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "(";
    if (!shape.empty()) {
        std::cout << shape[0];
        for (size_t i = 1; i < shape.size(); i++) {
            std::cout << ", " << shape[i];
        }
    }
    std::cout << ")" << std::endl;
}

std::tuple<string, string, string> getConfig() {
    // config.json
    std::ifstream ifs{R"(config.json)"};
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open config.json" << std::endl;
        exit(1);
    }
    nlohmann::json j;
    ifs >> j;
    string onnx = j["onnx"];
    string vocab = j["vocab"];
    string ner_arg = j["ner_arg"];
    return {onnx, vocab, ner_arg};
}
struct RuntimeConfig {
    string onnx,vocab,ner_arg,test_data;
    bool enable_gpu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RuntimeConfig,onnx,vocab,ner_arg,test_data,enable_gpu)

template <class T_ori>
class DataSet {
  public:
    virtual size_t size() = 0;
    virtual const T_ori& operator[](size_t index) = 0;
};
class NerDataSet: public DataSet<wstring> {
  public:
    explicit NerDataSet(const string &file) {
        std::ifstream ifs{file};
        if (!ifs.is_open()) {
            std::cerr << "Error: failed to open " << file << std::endl;
            exit(1);
        }
        std::string s;
        while (std::getline(ifs, s)) {
            wstring text = utf8decode(s);
            document.push_back(text);
        }
    }
    size_t size() override {
        return document.size();
    }
    const wstring& operator[](size_t index) override {
        if (index >= document.size()) {
            throw std::out_of_range("index out of range");
        }
        return document[index];
    }
  private:
        vector<wstring> document;
};


int main(int argc, char **argv, char **envp) {
    std::ifstream ifs1{"config.json"};
    auto j2 = R"({"happy":true,"pi":3.141})"_json;
    nlohmann::json j;
    ifs1 >> j;
    RuntimeConfig rcfg;
    from_json(j, rcfg);
    auto file = rcfg.test_data;
    auto use_gpu = rcfg.enable_gpu;
    auto onnx = rcfg.onnx;
    auto vocab = rcfg.vocab;
    auto ner_arg = rcfg.ner_arg;
    //auto file = j["text_data"].template get<string>();
    //string file = j["test_data"];
    //string use_gpu1 = j["enable_gpu"];
    //bool use_gpu = use_gpu1 == "true";
    //auto [onnx, vocab, ner_arg] = getConfig();
    ModelArgs args{ner_arg};
    EntityParser parser{args};
    BGMNERInfer infer{onnx, ner_arg, vocab, use_gpu};

    std::ifstream ifs{file};
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open " << file << std::endl;
        exit(1);
    }
    auto nowTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    vector<wstring> document;
    while (std::getline(ifs, file)) {
        wstring text = utf8decode(file);
        document.push_back(text);
    }
    auto total_size = document.size();
    auto batch_size = 200;
    vector<wstring> batch;
    int i;
    for (i = 0; i + batch_size < total_size; i += batch_size) {
        for (int j = 0; j < batch_size; j++) {
            batch.push_back(document[i + j]);
        }
        try {
            auto logits = infer.infer(batch);
            cout << logits.size() << endl;
             //for (int j = 0; j < logits.size(); j++) {
             //    auto labelSeq = parser.seq2label(logits[j].data(),
             //                                     min(batch[j].size(),
             //                                     args.getMaxSeqLen()));
             //    auto entities = EntityParser::getEntities(labelSeq);
             //    for (const auto &entity : entities) {
             //        std::cout << entity.first << " " << entity.second.first
             //                  << " " << entity.second.second << " ";
             //        std::cout
             //            << outputChineseToConsole(batch[j].substr(
             //                   entity.second.first,
             //                   entity.second.second - entity.second.first +
             //                   1))
             //            << std::endl;
             //    }
             //}
        } catch (Ort::Exception &e) {
            std::cerr << e.what() << std::endl;
        }
        batch.clear();
    }
    while (i < total_size) {
        string line;
        std::getline(ifs, line);
        wstring text = utf8decode(line);
        batch.push_back(text);
        i++;
    }
    auto logits = infer.infer(batch);
    cout << logits.size() << endl;

    auto endTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    std::cout << "time: " << endTimeMs - nowTimeMs << std::endl;
    std::cout << std::endl;
}
