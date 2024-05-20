# BGMNER C++ 推理

[en](README.en.md) | [中文](README.md)


## 摘要

这个项目是 github.com/Gaomengkai/bgmner 的一部分，
该项目是用于中文 Bangumi 文本的命名实体识别（NER）模型。

这个 C++ 推理项目是一个独立程序，可以使用导出的 onnx 格式的 BGMNER 模型
对中文 Bangumi 文本进行命名实体识别。

## 介绍

BGMNER 模型是一个用于中文 Bangumi 文本的命名实体识别（NER）模型。

当我们谈论像这样的句子时
```text
[Lilith-Raws] 偶像大师 闪耀色彩 / The iDOLM@STER Shiny Colors - 07 [Baha][WebDL 1080p AVC AAC][CHT]
```
我们可以从中提取以下实体：
```text
[Lilith-Raws]       -> GR(组名)
偶像大师 闪耀色彩    -> CT(中文标题)
The iDOLM@STER Shiny Colors -> ET(英文标题)
07                  -> EP(集数)
1080p               -> RES(分辨率)
CHT                 -> SUB(字幕)
```

## 配置

### 构建前

在构建项目之前，您需要安装以下依赖项。

- [onnxruntime](https://github.com/microsoft/onnxruntime)

然后配置 CMakeLists.txt 找到 onnxruntime，如下所示：

```cmake
set(ONNXRUNTIME_PATH "D:\\SDK\\onnxruntime-directml")
```

构建脚本仅在 Windows/clang-16 上测试过。Linux 可能需要一些修改，但我不确定。

### 运行前

有些 make 系统不能将 .dll 文件复制到输出目录，因此您可能需要手动复制 .dll 文件。
包括：
- `DirectML.dll`
- `onnxruntime.dll`

如果您想使用 CUDA，还需要复制以下 .dll 文件：
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `onnxruntime_providers_tensorrt.dll`

您需要在可执行文件的同一目录中创建（或修改）一个 `config.json` 文件。

```json
{
  "onnx": "bert_ner.onnx",
  "vocab": "model_hub\\chinese-bert-wwm-ext\\vocab.txt",
  "ner_arg": "ner_args.json",
  "test_data": "1000.txt",
  "enable_gpu": true
}
```

## 模型

onnx 模型有 2 个输入：

- `input_ids: [batch_size, seq_len], dtype=int64`
- `attention_mask: [batch_size, seq_len], dtype=int64`

和 1 个输出：

- `logits: [batch_size, seq_len], dtype=int64`

您需要对输入文本进行分词，并将其转换为 input_ids 和 attention_mask。
但不用担心，`ner_args.json` 文件和 `src/BasicTokenizer.cpp` 将帮助您完成这些工作。
