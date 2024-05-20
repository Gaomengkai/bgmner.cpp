//
// Created by gaome on 2024/5/11.
//

#include "EntityParser.h"

namespace bgmner {
vector<string> EntityParser::seq2label(const int64_t *seq, size_t len) {
    vector<string> ret;
    for (size_t i = 0; i < len; i++) {
        ret.push_back(m_model_args.getLabel(seq[i]));
    }
    return ret;
}
vector<string> EntityParser::seq2label(const vector<int64_t> &seq, size_t len) {
    vector<string> ret;
    for (size_t i = 0; i < len; i++) {
        ret.push_back(m_model_args.getLabel(seq[i]));
    }
    return ret;
}
vector<std::pair<string, std::pair<int32_t, int32_t>>> EntityParser::getEntities(std::vector<string> labelSeq) {
    vector<std::pair<string, std::pair<int32_t, int32_t>>> entities;
    string current_entity_type;
    int32_t current_entity_start = -1;
    int32_t current_entity_end = -1;
    for (int32_t k = 1; k < labelSeq.size(); k++) {
        int i = k - 1;
        string label = labelSeq[k];
        if (label[0] == 'B') {
            if (!current_entity_type.empty()) {
                entities.push_back(
                    {current_entity_type,
                     {current_entity_start, current_entity_end}});
            }
            current_entity_type = label.substr(2);
            current_entity_start = i;
            current_entity_end = i;
        } else if (label[0] == 'I') {
            if (current_entity_type.empty()) {
                current_entity_type = label.substr(2);
                current_entity_start = i;
                current_entity_end = i;
            } else if (label.substr(2) != current_entity_type) {
                entities.push_back(
                    {current_entity_type,
                     {current_entity_start, current_entity_end}});
                current_entity_type = label.substr(2);
                current_entity_start = i;
                current_entity_end = i;
            } else {
                current_entity_end = i;
            }
        } else {
            if (!current_entity_type.empty()) {
                entities.push_back(
                    {current_entity_type,
                     {current_entity_start, current_entity_end}});
                current_entity_type = "";
                current_entity_start = -1;
                current_entity_end = -1;
            }
        }
    }
    if (!current_entity_type.empty()) {
        entities.push_back(
            {current_entity_type, {current_entity_start, current_entity_end}});
    }
    return entities;
}
} // namespace bgmner.cpp