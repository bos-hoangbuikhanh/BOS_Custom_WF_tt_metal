# Llama-3.2

## Introduction
This codebase includes the Llama-3.2 family of models and currently supports the following variants:
- **Llama-3.2-1B-Instruct:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Llama-3.2-3B-Instruct:** [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Llama-3.1-8B-Instruct:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## Prerequisite: Hugging Face Setup

Before running any Llama model, you must set up your Hugging Face account and access tokens properly.

### 1. Sign Up for Hugging Face
Create an account at [https://huggingface.co](https://huggingface.co) if you don’t already have one.

### 2. Request Access to Llama Models
Access to Meta’s Llama models is **restricted** — you must manually request permission for each model family.

- [Llama-3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Llama-3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

On each model page, click **“Request Access”** and wait for Meta’s approval.
You’ll receive an email once access is granted.

### 3. Generate an Access Token
Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

1. Click **“New token”**
2. Give it a name (e.g. `llama32-access`)
3. Set permission to **Read**
4. Copy the token string — you’ll need it in the next step.

### 4. Set Your Token as an Environment Variable
Store your token so it can be used automatically by scripts:
```bash
export HUGGING_FACE_HUB_TOKEN=<your_token_here>
```

### 5. Log In via CLI (Alternative)
If you prefer to log in interactively instead of setting an environment variable, use the Hugging Face CLI:
```bash
huggingface-cli login
```
When prompted, paste your token and press Enter.
Once authenticated, your credentials will be stored locally and used automatically for future Hugging Face operations.

### 6. Verify Login
To confirm that your login or token setup is working correctly, run:
```bash
huggingface-cli whoami
```
If configured properly, your Hugging Face username will be displayed.
If it shows an authentication error, please repeat Step 5 or check your token’s permission settings.


## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## How to Run
For a single user example:
```
HF_MODEL=<model_name> pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-one'
```

**Notes:**
- `<model_name>` is the HuggingFace model repo string, e.g. `meta-llama/Llama-3.2-1B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`.
- `-k` is the pytest filter; to run a specific test, use `-k <test_name>`; additional test names are listed in `models/bos_model/llama32/demo/demo.py`

For a batch user example:

```
HF_MODEL=<model_name> pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-16'
```

For Live chatting demo example, using Llama-3.2-1B:
```
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python models/bos_model/llama32/run_llama32.py --live
```

**Notes:**
- If you do not use `--live`, Llama takes in queries specified in the file set by the argument `--queries`. Default questions are at `models/bos_model/llama32/tests/queries.txt`
- The number of questions can be limited by `--num_iters` or `-n`
- There are three available `--memory_mode`s, `HIGH_CONTEXT` remembers all questions, and the last answer. `LOW_CONTEXT` remembers only the last 5 questions. `NO_MEMORY` does not remember any questions, and sets no instructions for the model. Default memory mode is `LOW_CONTEXT`
- Use `-g <integer>` to set the maximum generated tokens per answer.
- Use `--output_path <output_file_path.txt>` to print your Llama queries and responses, and performance metrics, into a file.
- Loading the 8B model may take some time. It is recommended to wait 1–2 minutes for the model to fully initialize before starting the chat.

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.

## Tracy profiling

Tracy is a performance profiling tool that provides visual analysis of execution time and memory usage for each operation during model execution.

Just make sure you have built Metal using the `-p` tag, instead of `-b Release` tag, to have enabled profiling

Usage:
```bash
HF_MODEL=<model_name> python -m tracy -r -p -v -m pytest models/bos_model/llama32/test_demo_llama32.py
```

## Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```llama```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/<$EXPERIMENT_NAME>_MMMDD_hhmm/```)
```
HF_MODEL=<model_name> pytest models/bos_model/llama32/test_demo_llama32.py
```

- Third, run ttnn-visualizer
    - ```$REPORT_PATH```: It is the path mentioned in the previous step
    - visit ```http://localhost:8000/``` using your web-browser
```
ttnn-visualizer --profiler-path $REPORT_PATH
```

- If the experiment has finished, please run the following command to clear the environment variables
```
source models/bos_model/unset_l1_vis.sh
```
