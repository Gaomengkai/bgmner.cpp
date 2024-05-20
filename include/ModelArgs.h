//
// Created by gaome on 2024/5/11.
//

#ifndef BGMNER_MODELARGS_H
#define BGMNER_MODELARGS_H
#include <string>
#include <unordered_map>

namespace bgmner {
    class ModelArgs {
      public:
        explicit ModelArgs(const std::string &json_path);
        int32_t getLabelId(const std::string &label);
        std::string  getLabel(int32_t id);
        [[nodiscard]] int64_t  getMaxSeqLen() const;
      private:
        std::unordered_map<std::string, int32_t> m_label2id;
        std::unordered_map<int32_t, std::string> m_id2label;
        int64_t m_max_seq_len;
    };
}

#endif // BGMNER_MODELARGS_H
