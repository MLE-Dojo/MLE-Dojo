"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict

from dataclasses_json import DataClassJsonMixin
from mledojo.agent.aide.utils.metric import MetricValue

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the status, feedback, scores, and other execution information.
    """

    status: str
    feedback: Dict
    raw_score: float
    position_score: float
    # best_raw_score: float
    # best_position_score: float

@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    instruction_prompt: str = field(default=None, kw_only=True)  # type: ignore
    node_type: str = field(default=None, kw_only=True)  # type: ignore
    assistant: str = field(default=None, kw_only=True)  # type: ignore
    status: str = field(default=None, kw_only=True)  # type: ignore
    feedback: Dict = field(default=None, kw_only=True)  # type: ignore
    raw_score: float = field(default=None, kw_only=True)  # type: ignore
    position_score: float = field(default=None, kw_only=True)  # type: ignore
    # best_raw_score: float = field(default=None, kw_only=True)  # type: ignore
    # best_position_score: float = field(default=None, kw_only=True)  # type: ignore

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self.status = exec_result.status
        self.feedback = exec_result.feedback
        self.raw_score = exec_result.raw_score
        self.position_score = exec_result.position_score
        # self.best_raw_score = exec_result.best_raw_score
        # self.best_position_score = exec_result.best_position_score

    # @property
    # def term_out(self) -> str:
    #     """Get the terminal output of the code execution (after truncating it)."""
    #     return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step+1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step+1}]:\n")
            term_out = n.output if n.output is not None else n.error
            trace.append(term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step+2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest metric value)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric.value)
    
    def get_best_trajectory(self) -> list[Dict]:
        """Return the best trajectory lead to best node found so far."""
        best_node = self.get_best_node()
        trajectory = []
        while best_node is not None:
            best_node_dict = []
            best_node_dict.append({"role":"system", "content":best_node.instruction_prompt})
            best_node_dict.append({"role":"user", "content":best_node.assistant})
            trajectory.extend(best_node_dict)
            best_node = best_node.parent
        return trajectory

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            if n.feedback is not None:
                summary_part += f"Execution and Evaluation Feedback: {n.feedback}\n"
            if n.raw_score is not None:
                summary_part += f"Test Metric Score: {n.raw_score}\n"
            if n.position_score is not None:
                summary_part += f"Test Position Score: {n.position_score}\n"
            # if n.best_raw_score is not None:
            #     summary_part += f"Best Raw Score: {n.best_raw_score}\n"
            # if n.best_position_score is not None:
            #     summary_part += f"Best Position Score: {n.best_position_score}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)