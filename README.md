# HyC-LoRA: Memory Efficient Fine-tuning with Outlier-aware Hybrid Activation Compression

## News

* [07/05/2024] Release code.

## Usage

### Environment

#### Base

Install the required packages:

```bash
pip install -r requirements.txt
```

#### Transformers

Install modified `transformer` package (some operators are modified to support our method):

```bash
git clone git@github.com:Ther-nullptr/transformers.git
cd transformers
pip install -e .
```

Or replace the original transformer package with following code:

```bash
models/modeling_llama.py -> transformers/src/transformers/models/llama/modeling_llama.py
models/modeling_mistral.py -> transformers/src/transformers/models/mistral/modeling_mistral.py
models/modeling_roberta.py -> transformers/src/transformers/models/roberta/modeling_roberta.py
```

#### BitsandBytes

Install modified `bitsandbytes` package (a memory bug, see [PR](https://github.com/TimDettmers/bitsandbytes/pull/1270)):

```bash
git clone git@github.com:Ther-nullptr/bitsandbytes.git
```

Or replace the original bitsandbytes package with following code:

```bash
bitsandbytes_fix/_function.py -> bitsandbytes/bitsandbytes/autograd/_function.py
```

### Experiments: RoBERTa

Modify the config in the `run_glue.sh` script to run the experiments:

```bash
bash run_glue.sh
```

### Experiments: LLaMA-2-7B/Mistral-7B

(Optional) Use [LoftQ](https://github.com/yxli2123/LoftQ) method to compress the weight:

```bash
bash generate_loftq.sh
```

Modify the config in the `run_gsm8k.sh` script to run the experiments:

```bash
bash run_gsm8k.sh
```

## TODO

* [ ] Add ProSparse-LLaMA modeling code.
* [ ] Add LongLoRA finetuning of RedPajama dataset code.
* [ ] CUDA kernel optimization.
* [ ] Pure C/CUDA implementation on edge devices.

## Acknowledgement

Our code is build upon the following open-source projects:

* [GACT](https://github.com/LiuXiaoxuanPKU/GACT-ICML)
* [LoftQ](https://github.com/yxli2123/LoftQ)
* [LongLoRA](https://github.com/dvlab-research/LongLoRA)

We thank the authors for their open-sourced code.