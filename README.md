---
license: apache-2.0
tags:
  - onnx
  - mistral
  - instruction-tuned
  - alex-ai
  - fp32
library_name: onnxruntime
model_creator: techAInewb
model_name: mistral-nemo-2407-fp32
---

# Mistral-Nemo Instruct 2407 — ONNX FP32 Export

This repository contains the ONNX-formatted FP32 export of the **Mistral-Nemo Instruct 2407** model, compatible with ONNX Runtime.

## 🧠 Model Summary

This is the **flagship release** of the Alex AI project — and to our knowledge, the **first-ever open ONNX-format export of Mistral-Nemo Instruct 2407** for full-stack experimentation and deployment.

- **Architecture**: Mistral-Transformer hybrid, instruction-tuned for reasoning and alignment
- **Format**: ONNX (graph + external weights)
- **Precision**: FP32 (float32)
- **Exported Using**: PyTorch → ONNX via `torch.onnx.export`

This model forms the foundation for future research in quantization, NPU acceleration, memory-routing, and lightweight agent design. It is being positioned as a clean and transparent baseline for community optimization — with future support for AMD Vitis AI, Olive, and quantized variants.

## 📁 Files Included

| File               | Description                             |
|--------------------|-----------------------------------------|
| `model.onnx`       | The model graph                         |
| `model.onnx.data`  | External tensor weights (~27GB)         |
| `config.json`      | Model configuration metadata            |
| `requirements.txt` | Runtime dependencies                    |
| `LICENSE`          | Apache 2.0 License                      |

## ✅ Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage Example

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
attention_mask = np.ones_like(input_ids)

outputs = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

print(outputs[0].shape)  # (1, 5, vocab_size)
```

## 💡 Project Vision

The Alex AI project was created to explore what’s possible when we combine precision reasoning, self-evolving memory, and strict efficiency — all under real-world constraints.

This model is a public cornerstone for research in ONNX deployment, quantization, agent routing, and modular NPU workflows. It is open, transparent, and designed for practical extension.

We believe high-quality tools shouldn’t be locked behind paywalls.

## 🤝 Get Involved

- GitHub: [github.com/techAInewb](https://github.com/techAInewb)
- Hugging Face: [huggingface.co/techAInewb](https://huggingface.co/techAInewb)
- Project Chat: Coming soon

Contributions, forks, and optimization experiments are welcome!

## 💸 Support the Project

If you’d like to support open-source AI development, please consider donating:

**[🫶 Donate via PayPal](https://www.paypal.com/paypalme/AlexAwakens)**  
_Message: "Thank you for your donation to the Alex AI project!"_

## 📜 License

This model is released under the [Apache 2.0 License](./LICENSE).


## 🧪 Inference Validation

This model has been validated using ONNX Runtime in a local Windows 11 environment:

- **System**: AMD Ryzen 5 7640HS, 16GB RAM, RTX 3050 (6GB), Windows 11 Home
- **Runtime**: `onnxruntime==1.17.0`, Python 3.10, Conda environment `alex-dev`

Test inference was run with:

```python
input_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
attention_mask = np.ones_like(input_ids)
```

**Result:**
- ✅ Model loaded and executed without error
- ✅ Output logits shape: `(1, 5, 131072)`
- ⚠️ Memory usage may exceed 20GB for full batch sizes — ensure pagefile is set appropriately (we used 350GB)
- 🚫 No GPU or CUDA acceleration used for this test — CPU-only validation

This confirms that full ONNX FP32 export is working and stable, even under real-world hardware constraints.
