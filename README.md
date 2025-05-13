<p align="center">
  <img src="assets/icon.jpg" width="500" alt="MLE-Dojo" />
</p>

<h1 align="center">
  MLE-Dojo: Interactive Environments for Empowering LLM Agents in Machine Learning Engineering
</h1>

<p align="center" style="font-family:'Segoe UI', Roboto, sans-serif; font-weight:bold; text-transform:uppercase;">
  <a href="https://arxiv.org/abs/2505.07782">
    <img src="https://img.shields.io/badge/Arxiv-2505.07782-000000.svg?style=flat-square&logo=arxiv&logoColor=%23FFD700&labelColor=000000" height="28">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://mle-dojo.github.io/MLE-Dojo-page/">
    <img src="https://img.shields.io/badge/Project%20Website-%20-000000.svg?style=flat-square&logo=Google-Chrome&logoColor=%23FFD700&labelColor=000000" height="28">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/MLE-Dojo/Leaderboard">
    <img src="https://img.shields.io/badge/Leaderboard-000000.svg?style=flat-square&logo=huggingface&logoColor=%23FFD700&labelColor=000000" height="28">
  </a>
</p>



**MLE-Dojo** is a Gym-style framework for systematically reinforcement learning, evaluating, and improving autonomous large language model (LLM) agents in iterative machine learning engineering (MLE) workflows. 
Built upon 200+ real-world Kaggle challenges. **MLE-Dojo** covers diverse, open-ended MLE tasks carefully curated to reflect realistic Machine Learning Engineering scenarios such as data processing, architecture search, hyperparameter tuning, and code debugging, etc.
**MLE-Dojo**'s fully executable environment and flexible interface support comprehensive agent training via both supervised fine-tuning and reinforcement learning, facilitating iterative experimentation, realistic data sampling, and real-time outcome verification.

<p align="center">
  <img src="assets/overview.jpg" width="900" alt="MLE-Dojo" />
</p>

## 🛠️ Experiment Setup

### Prerequisites

*   **Docker:** (Optional) Required for building and running the execution environment.
*   **NVIDIA Container Toolkit:** Required if using GPUs within Docker containers.
*   **Conda:** One of the most convenient ways is to build with `conda`.
*   **Git:** Basic git support.

### Installation

**Clone the repository:**
  ```bash
  git clone https://github.com/MLE-Dojo/MLE-Dojo.git
  cd MLE-Dojo
  ```
    
**Build the Docker image:** (Optional)
  ```bash
  docker build -t mle-dojo .
  ```

**Build the Conda Env:**
  ```bash
  pip install --upgrade pip setuptools wheel
  conda create -y -n mle-dojo python=3.11
  conda activate mle-dojo
  pip install -e .
  ```
    
    

## 📇 Data Preparation
MLE-Dojo currently supports **200+** Kaggle competitions.
We aggregate 68 competitions from MLE-Bench (excluding 7 tasks that are unavailable, excessively large, or tightly coupled with specific packages), 74 from DSBench, and 75 additionally scraped and prepared carefully from Kaggle's official website. After removing duplicate entries across sources, we obtain a diverse collection of over 200 unique tasks.

### Install the Kaggle package with pip and setup:
```bash
pip install kaggle
```
Next, go to your Kaggle account settings, generate a new API token, and move the downloaded kaggle.json file to `~/.kaggle/kaggle.json` on Linux/macOS or `C:\Users\<YourUsername>\.kaggle\kaggle.json` on Windows. Refer to [Kaggle API](https://www.kaggle.com/docs/api) for details.

### Accept terms and conditions
Before downloading each competition, you may need to manually accept its terms and conditions on its official Kaggle website. We provide the competition details in `prepare/*.json`, including website urls, in ascending order of competition data sizes. Feel free to prepare and play with any competition you want!

**❗️Note: This action is needed for each competition you want to prepare and utilize.** 

To prepare newly introduced or MLE-Bench competitions, specify the `competitions-file` (a txt file with each line corresponding to a competition), `data-dir` and the `logs-dir`. We actively maintain and update supported competitions in `prepare/mle.json`. 
```bash
PYTHONPATH="." python prepare/mle.py \
  --competitions-file prepare/competitions.txt \
  --data-dir ./data/prepared \
  --logs-dir ./data/prepare_logs
```
Alternatively, you can specify the `competitions` to prepare via args.
```bash
PYTHONPATH="." python prepare/mle.py \
--competitions random-acts-of-pizza \
--data-dir ./data/prepared \
--logs-dir ./data/prepare_logs
```
To prepare the DS-Bench data, specify paths for raw data and prepared data. This will prepare all the competitions in `prepare/dsbench.json`
```
PYTHONPATH="." python prepare/dsbench.py \
--raw-dir ./data/raw \
--prepared-dir ./data/prepared
```
⏰ Reminder: The data preparation is both time- and space-consuming. Please allocate sufficient resources based on the data size and ensure you have accepted the competition's terms and conditions on the official website in advance.

## 🚀 Quick Start in Python
Here's a quick example to show how to interact with MLE-Dojo:

### 1. Import from `mledojo` and register
```python
from mledojo.gym.competition import CompetitionRegistry
from mledojo.competitions import get_metric
from mledojo.gym.env import KaggleEnvironment

competition_name = "random-acts-of-pizza"
data_dir = ...
output_dir = ...

# register the competition
registry.register(
    name=competition_name,
    data_dir=data_dir,  # "random-acts-of-pizza/data"
    comp_info=CompInfo(
        category="General",
        level="beginner",
        output_type="submission.csv",
        higher_is_better=True
    ),
    metric_class=get_metric(competition_name)
)
```
### 2. Initialize the environment and start interacting with it
```python
# initialize the env
env = KaggleEnvironment.make(
    competition_name=competition_name,      
    output_dir=output_dir,         
    competition_registry=registry,                  
    score_mode="position",              
    gpu_device=0,                     
    gpu_memory_limit=32,                   
    execution_timeout=3600             
)

# request_info
env.step("request_info", **{"info_type": "overview"})

# validate_code
env.step("validate_code", **{"code": "import pandas as pd\nprint('Welcome to MLE-Dojo!')"})

# Execute_code
absolute_data_dir = Path(os.path.abspath(data_dir))
absolute_output_dir = Path(os.path.abspath(output_dir))
code_to_execute = f'''
import pandas as pd
submission = pd.read_csv('{absolute_data_dir / "public" / "sample_submission.csv"}')
submission.to_csv('{absolute_output_dir / "submission.csv"}', index=False)
print("Submission created successfully.")
'''
env.step("execute_code", **{"code": code_to_execute})
```

## 💽 Run Experiments
We support running experiments both with and without Docker. We recommend Docker for running experiments with our supported agent scaffolds and models.

### Manage LLM API Keys
Create `.env` under the main directory and save API Keys. "HF_TOKEN" is used for local models.
```
OPENAI_API_KEY=""
GEMINI_API_KEY=""
ANTHROPIC_API_KEY=""
DEEPSEEK_API_KEY=""
XAI_API_KEY=""
HF_TOKEN=""
```
We also support Azure Openai API in `mledojo/chat/utils`, you can leave `OPENAI_API_KEY` blank and fill `AZURE_API_CONFIG` with corresponding information.

### Run with Docker
After getting some competitions prepared, you can get started with running experiments on your prepared data.
Experiment with `o3-mini` and `MLE Agent` scaffold on prepared `random-acts-of-pizza` competition or competitions in `prepare/competitions.txt` (replace `--competitions random-acts-of-pizza` with `--competitions-file prepare/competitions.txt`):
```bash
python run.py \
    --gpu-indices 0 \
    --max-tasks-per-gpu 1 \
    --competitions random-acts-of-pizza \
    --docker-image mle-dojo \
    --task-runtime-seconds 43200 \
    --kaggle-data ./data/prepared \
    --log-dir ./results/logs \
    --output-dir ./results/output
```

### Run without Docker
We also support running experiments without Docker, refer to `main.py` for more args:
```python
python main.py \
    --output-dir ./output \
    --data-dir ./data/prepared \
    --competition-name random-acts-of-pizza \
    --agent-type mle
```
### Configs and Settings
We use separate configs for **Agent** and **Experiment**. Config Path for **Agent** is `mledojo/agent/*/config.yaml` and **Experiment** Config is `config.yaml`. 
**MLE Agent** scaffold supports most mainstream LLMs through API or local serving.

| model_mode | Supported model_name Variants                                                                  |
|------------|-------------------------------------------------------------------------------------------------|
| gpt        | gpt-4o, gpt-4o-mini, o1-mini, o3-mini...                                                        |
| deepseek   | deepseek-reasoner, deepseek-chat                                                                |
| gemini     | gemini-2.0-flash, gemini-2.0-pro, gemini-2.5-pro-preview-03-25, gemini-2.5-pro-exp-03-25...     |
| grok       | grok-3-beta, grok-3-mini-beta...                                                                |
| claude     | claude-3-7-sonnet-latest, claude-3-5-sonnet-latest...                                           |
| local      | LLaMA, Qwen, ...                                                                                |

For running local models, refer to `run_local.sh`. We use [vLLM](https://github.com/vllm-project/vllm) for local model serving.


### Agent Scaffolds and Packages
We currently support:
1. Agents with originally supported action space in `mledojo`, named **MLE Agent**.
   
2. [AIDE](https://github.com/WecoAI/aideml) (AI-Driven Exploration), an LLM agent that starts by generating a set of initial solution drafts and then iteratively refines and improves them based on performance feedback. We modify the source of the feedback from direct code outputs to `mle-dojo` environment feedback.
   
3. [OpenAI Agents](https://openai.github.io/openai-agents-python/), a package that enables building agentic AI apps with abstractions. We implement a version of **MLE Agent** with `openai-agents` package.

## 🌟 Develop with `mledojo`

### `mledojo` Interface and APIs
We provide [quick examples](example/example.ipynb) of APIs.
MLE-Dojo provides flexible Gym-style APIs that allow users to build personalized environments, introduce new datasets and develop/utilize different agent scaffolds.
Specifically, `mledojo` serves a powerful and flexible toolkit for interacting with machine learning competitions, designed for ease of use and extensibility. Its core components offer a seamless development experience:

1. **`Interface`**: Provides the primary way to interact with a competition. It manages distinct sub-interfaces for specific tasks:
    *   `InfoInterface`: Fetches competition details (overview, data structure, etc.).
    *   `CodeValidationInterface`: Performs **safe, preliminary checks** on user code syntax and basic runtime behavior within a sandbox.
    *   `CodeExecutionInterface`: Manages the **full execution of user code** within the sandbox, handles submission generation, and triggers evaluation.
    *   **Flexibility**: The `Interface` is **modular**, allowing developers to easily **register custom interaction components** tailored to specific needs.

2. **`Sandbox`**: Ensures **secure and fair code execution** by running user scripts in an isolated environment with **configurable resource limits** (GPU, memory, time).

3. **`FeedbackManager`**: Processes the results from validation and execution, generating **structured and informative feedback**.
    *   **Extensibility**: Designed to be extensible, allowing the **registration of custom feedback providers** (e.g., LLM-based analysis, specific error pattern detection) alongside the default `BaseFeedback`.

4. **`KaggleEnvironment`**: The main entry point, wrapping all components into a **standardized, Gymnasium-compatible environment**. It orchestrates the entire workflow (setup, action handling via `step`, state tracking, feedback generation) providing a consistent API for users or automated agents.

MLE-Dojo combines well-defined interfaces, secure execution, structured feedback, and centralized management within a familiar Gym-style environment. Its **modular and extensible design** grants developers significant flexibility to adapt and build upon the core framework.

### Collect trajectories for training
MLE-Dojo provides a detailed history management system and a well-defined reward feedback mechanism, including both final outcome rewards and intermediate step rewards. This design enables flexible use for model training via Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL).
We provide both [**Agent History**](trajectories/agent_trajectory.json) and [**Environment History**](trajectories/env_history.json) structures.



## Contributing

Contributions are welcome! We will soon release instructions on Contributing.


## Dataset Usage Terms and Conditions

The dataset is provided solely for educational and academic research purposes, with the goal of supporting scholarly work in relevant fields. Users must comply with the following terms:

- **Data Integrity**:  
  While efforts have been made to curate and organize the dataset, no guarantees are made regarding its accuracy, completeness, or timeliness. Users are responsible for independently verifying the data and for any interpretations or outcomes based on its use.

- **Permitted Use**:  
  The dataset is restricted to non-commercial use only. Any commercial application, product development, or monetization based on the dataset requires prior written permission from the competition organizers.

- **Legal and Privacy Compliance**:  
  Users must comply with applicable laws and regulations, especially those related to data privacy and security. The dataset providers are not liable for any legal issues resulting from improper or unauthorized usage.

- **Intellectual Property**:  
  The dataset may include content derived from external sources and is intended for non-commercial research only. The providers do not claim ownership over such content and acknowledge the rights of the original creators. Users must ensure their usage does not infringe upon any intellectual property rights.

- **Disclaimer**:  
  The dataset is provided *"as is"*, without warranties of any kind. The providers are not responsible for any damages—direct or indirect—that may result from its use, including but not limited to financial loss, misinterpretation, or third-party claims.


## License

This code of project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
We adapt some of the data preparation codes from [DS Bench](https://github.com/liqiangjing/dsbench) and [MLE-Bench](https://github.com/openai/mle-bench).
Different competitions may be governed by different licenses. Users are responsible for reviewing and complying with the specific license associated with each competition.
We provide the all the [URLs](prepare/licenses.json) of specific rules and license for each competition. Please refer to the details.

## Citation

If you find this useful, you are more than welcome to cite:

```bibtex
@misc{qiang2025mledojointeractiveenvironmentsempowering,
  title={MLE-Dojo: Interactive Environments for Empowering LLM Agents in Machine Learning Engineering}, 
  author={Rushi Qiang and Yuchen Zhuang and Yinghao Li and Dingu Sagar V K and Rongzhi Zhang and Changhao Li and Ian Shu-Hei Wong and Sherry Yang and Percy Liang and Chao Zhang and Bo Dai},
  year={2025},
  eprint={2505.07782},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.07782}, 
}

