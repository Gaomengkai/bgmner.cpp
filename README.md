# BGMNER C++ Inference

## Abstract

This project is a part of github.com/Gaomengkai/bgmner,
which is a Named Entity Recognition (NER) model for Chinese Bangumi text.

This C++ inference project is a standalone program that can be used to
perform NER on Chinese Bangumi text using the BGMNER model exported as onnx.

## Introduction

The BGMNER model is a NER model for Chinese Bangumi text.

When we talk about a sentence like 
```text
[Lilith-Raws] 偶像大師 閃耀色彩 / The iDOLM@STER Shiny Colors - 07 [Baha][WebDL 1080p AVC AAC][CHT]
```
We can extract the following entities from it:
```text
[Lilith-Raws]       -> GR(Groups)
偶像大師 閃耀色彩    -> CT(Chinese Title)
The iDOLM@STER Shiny Colors -> ET(English Title)
07                  -> EP(Episode)
1080p               -> RES(Resolution)
CHT                 -> SUB(Sub)
```

## Config

### before building

You'd install the following dependencies before building the project.

- [onnxruntime](https://github.com/microsoft/onnxruntime)

and then config CMakeLists.txt to find the onnxruntime like:

```cmake
set(ONNXRUNTIME_PATH "D:\\SDK\\onnxruntime-directml")
```

The build script was ONLY tested on Windows/clang-16. And Linux 
may need some modifications which I'm not sure about.

### before running

Some make system couldn't copy the .dlls to the output directory, so you may need to copy the .dlls manually.
including:
- `DirectML.dll`
- `onnxruntime.dll`

if you wanna use CUDA, you'd copy the following .dlls too:
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `onnxruntime_providers_tensorrt.dll`

You'd create(or modify) a `config.json` file in the same directory as the executable file.

```json
{
  "onnx": "bert_ner.onnx",
  "vocab": "model_hub\\chinese-bert-wwm-ext\\vocab.txt",
  "ner_arg": "ner_args.json",
  "test_data": "1000.txt",
  "enable_gpu": true
}
```

## model

the onnx model has 2 inputs:

- `input_ids: [batch_size, seq_len], dtype=int64`
- `attention_mask: [batch_size, seq_len], dtype=int64`

and 1 outputs:

- `logits: [batch_size, seq_len], dtype=int64`

You'd tokenize the input text and convert it to input_ids and attention_mask.
But don't worry, the `ner_args.json` file and the `src/BasicTokenizer.cpp` will help you do this.
