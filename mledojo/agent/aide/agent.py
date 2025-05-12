import logging
import random
import humanize

import tiktoken
from typing import Any, Callable, cast, Dict, List, Tuple, Union
from dataclasses import dataclass

from mledojo.agent.aide.journal import Journal, Node, ExecutionResult
from mledojo.agent.aide.utils.config import Config
from mledojo.agent.aide.utils import data_preview
from mledojo.agent.aide.utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code
)
from mledojo.agent.aide.utils.util import (
    compile_prompt_to_md,
    opt_messages_to_list,
)
from mledojo.agent.aide.utils.metric import MetricValue, WorstMetricValue
from mledojo.chat import ChatClient, ModelSettings

logger = logging.getLogger("aide")

ExecCallbackType = Callable[[str], Dict]

@dataclass
class LLMConfig:
    """Configuration for LLM API settings"""
    model_mode: str  # "local", "gpt", "gemini", "claude", etc.
    model_name: str  # Model name or engine name
    port: int = 8314  # Port for local model
    max_completion_tokens: int = 8192
    max_prompt_tokens: int = 30000

class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        higher_is_better: bool,
        data_dir: str, 
        output_dir: str,
    ):
        super().__init__()
        self.total_cost = 0.0
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.higher_is_better = higher_is_better
        self.data_preview: str | None = None
        self.data_dir = data_dir
        self.output_dir = output_dir

        # history track
        self.cost_history = []  # Track cost history for each action
        self.conversation_history = []

        # LLM settings
        self.llm_config = self.acfg.code
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')
        self.total_tokens = 0

        # Initialize model client
        self.model_client = ChatClient(
            model_name=self.llm_config.model_name,
            model_category=self.llm_config.model_mode,
            api_idx=self.llm_config.api_idx,
            port=self.llm_config.port,
            api_key=self.llm_config.api_key
        )
        self.model_settings = ModelSettings(
            max_completion_tokens=self.llm_config.max_completion_tokens,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p
        )

    
    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node
    
    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks, use PyTorch rather than TensorFlow. You have access to a GPU and can use CUDA for faster computations if needed."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            # "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the script before finishing the code.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            f'All the provided input data is stored in {self.data_dir} directory.',
            f'**Save test predictions to `submission.csv` in {self.output_dir} directory as specified in the task description.** This file is critical for evaluation and scoring.',
            'Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.',
            'Your goal is to achieve the highest score, so ensure your code produces this file correctly.',
            f'You can also use the "{self.output_dir}/working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
    
    # TODO: AIDE currently **only support** gpt models, like gpt-4o-mini, can enhance later 
    def query_llm(self, system_message: str, user_message: str = None) -> str:
        """Query the LLM model."""
        system_message=compile_prompt_to_md(system_message) if system_message else None
        user_message=compile_prompt_to_md(user_message) if user_message else None
        messages = opt_messages_to_list(system_message, user_message)
        # Check if messages conform to OpenAI standard message format
        # Each message should be a dict with 'role' and 'content' keys
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
            
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role: {msg['role']}. Role must be one of: system, user, assistant")
            
            if not isinstance(msg['content'], str):
                raise ValueError("Message content must be a string")
        return self.model_client.chat_completion(messages, self.model_settings)
    
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text, cost = self.query_llm(
                system_message=prompt,
                user_message=None,
            )
            self.total_cost += cost
            self.cost_history.append({"action": "plan_and_code_query", "cost": cost})
            self.conversation_history.append({"role": "assistant", "content": completion_text})

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code, completion_text

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "","", completion_text  # type: ignore

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric according to the task description.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                f"Your Python code should be able to generate a `submission.csv` file in the `{self.output_dir}` directory to get the metric score."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        self.conversation_history.append({
            "Type": "Draft",
            "Memory": prompt["Memory"],
            "Instructions": prompt["Instructions"],
        })

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        
        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, instruction_prompt=compile_prompt_to_md(prompt), node_type="draft", assistant=assistant)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance, i.e. the position score provided by the competition. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        self.conversation_history.append({
            "Type": "Improve",
            "Memory": prompt["Memory"],
            "Instructions": prompt["Instructions"],
        })

        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            instruction_prompt=compile_prompt_to_md(prompt),
            node_type="improve",
            assistant=assistant
        )

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.feedback, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        self.conversation_history.append({
            "Type": "Debug",
            "Instructions": prompt["Instructions"],
        })

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code, assistant = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node, instruction_prompt=compile_prompt_to_md(prompt), node_type="debug", assistant=assistant)

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)
    
    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        # TODO: need to check if LLM call is working appropriately later
        if parent_node is None:
            print("Drafting...")
            result_node = self._draft()
        elif parent_node.is_buggy:
            print("Debugging...")
            result_node = self._debug(parent_node)
        else:
            print("Improving...")
            result_node = self._improve(parent_node)

        self.parse_exec_result(node=result_node, exec_result=exec_callback(result_node.code))
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: Dict):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        eval_result = ExecutionResult(
            status=exec_result["action_status"],
            feedback=exec_result["feedback"],
            raw_score=exec_result["current_raw_score"],
            position_score=exec_result["current_position_score"],
            # best_raw_score=exec_result["best_raw_score"],
            # best_position_score=exec_result["best_position_score"],
        )
        node.absorb_exec_result(eval_result)
        node.analysis = ""
        node.is_buggy = eval_result.status == "FAILED"

        if node.is_buggy:
            node.metric = WorstMetricValue(value = 0.0)
        else:
            node.metric = MetricValue(eval_result.position_score, maximize=True)