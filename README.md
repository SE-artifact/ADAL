## 👋 Introduction
This is the repository of the paper _ADAL: An Adversarial Defeasible Reasoning Framework for Automated Patch Correctness Assessment_.

## 📄 Project Structure
```text
├─📁 adal
├─📁 artifacts
│ ├─📁 LLM_as_judge
│ ├─📁 prompts
│ ├─📄 benchmark.tar.bz2
│ └─📄 program_artifacts.tar.bz2
├─📄 config.example.json
└─📄 README.md
```
> **adal** contains deployment and implementation details for our approach, along with the LLM-based automated scoring procedure.<br>
> **artifacts** contains the prompt templates used in our experiments and our benchmark.<br>

## 🛠️ Set Up
### Install dependencies：
- Python 3.12 and `pip`.
- JDK 21, [GumTree](https://github.com/GumTreeDiff/gumtree/releases), and [Joern](https://github.com/joernio/joern#requirements).
- GNU `tar` and `bzip2`.
- A compatible GPU environment for Qwen3-14B deployment. Follow the [vLLM installation instructions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) for the required driver and software environment.

Create a Python environment and install ADAL:
```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```
The installation includes NumPy, scikit-learn, joblib, and requests. Install GumTree and Joern separately and make `gumtree`, `joern-parse`, and `joern-export` available on `PATH`.

### Extract the datasets
```bash
tar -xjf artifacts/benchmark.tar.bz2 -C artifacts/
tar -xjf artifacts/program_artifacts.tar.bz2 -C artifacts/

mkdir -p data results models
```

### Configure the analysis tools and models

Create the configuration file:

```bash
cp config.example.json config.json
```

Set the following entries in the `paths` object of `config.json` to executable names or installation paths:

```json
{
  "gumtree": "gumtree",
  "joern_parse": "joern-parse",
  "joern_export": "joern-export"
}
```

Choose the backbone through `models.provider`:

| Backbone | Provider | Required configuration |
| --- | --- | --- |
| DeepSeek-V3 | `deepseek-v3` | `deepseek_model`, `deepseek_endpoint`, `deepseek_api_key` |
| Qwen3-14B | `qwen3-14b` | `qwen_model`, `vllm_base_url`, optional `vllm_api_key` |
| GPT-4.1 | `gpt-4.1` | `openai_model`, `openai_api_key` |

> **Note:** For the main evaluation and reasoning-quality analysis, configure the DeepSeek-V3 backbone and provide the GPT-4.1 judge credential in `models.openai_api_key`. Use the model identifiers corresponding to the experimental backbones.

## 🚀 Evaluation

Running example:
```bash
python -m adal lopo \
  --input "$DATASET" \
  --config config.json \
  --output results/adal_deepseek_v3.jsonl \
  --models-output models/adal_deepseek_v3.joblib
```
Use the same dataset and backbone configuration for the LLM-based baselines:
```bash
python -m adal baseline \
  --method patch \
  --input "$DATASET" \
  --config config.json \
  --output results/patch_deepseek_v3.jsonl
```

```bash
python -m adal baseline \
  --method specrover \
  --input "$DATASET" \
  --config config.json \
  --output results/specrover_deepseek_v3.jsonl
```
Run automatic scoring:
```bash
python -m adal.score \
  --source results/adal_deepseek_v3.jsonl \
  --output results/reasoning_quality.jsonl \
  --config config.json \
  --source-backbone deepseek-v3 \
  --expected-count "$N_PATCHES"
```

