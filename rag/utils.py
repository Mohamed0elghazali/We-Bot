import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'[INFO] Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

import threading
from typing import Any, Dict, List
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


MODEL_COST_PER_1K_TOKENS = {
    "mixtral-8x7b-32768": 0.00024,
    "mixtral-8x7b-32768-completion": 0.00024,

    "llama-3.3-70b-versatile": 0.00059,
    "llama-3.3-70b-versatile-completion": 0.00079,
    "llama-3.1-8b-instant": 0.00005,
    "llama-3.1-8b-instant-completion": 0.00008,

    "deepseek-r1-distill-llama-70b": 0.00075,
    "deepseek-r1-distill-llama-70b-completion": 0.00099,

    "gemma2-9b-it": 0,
    "gemma2-9b-it-completion": 0,

    "us.amazon.nova-lite-v1:0": 0.00006,
    "us.amazon.nova-lite-v1:0-completion": 0.00024,

    "us.amazon.nova-pro-v1:0": 0.0008,
    "us.amazon.nova-pro-v1:0-completion": 0.0032,

    "us.amazon.nova-premier-v1:0": 0.0025,
    "us.amazon.nova-premier-v1:0-completion": 0.0125,

    "us.deepseek.r1-v1:0": 0.00135,
    "us.deepseek.r1-v1:0-completion": 0.0054,

    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 0.003,
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0-completion": 0.015,

    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 0.003,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0-completion": 0.015,

    "us.meta.llama3-3-70b-instruct-v1:0": 0.00072,
    "us.meta.llama3-3-70b-instruct-v1:0-completion": 0.00072,
}

def standardize_model_name(model_name: str, is_completion: bool = False) -> str:
    """
    Standardize the model name to a format that can be used in the calculate price.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if is_completion:
        return model_name + "-completion"
    else:
        return model_name

def get_openai_token_cost_for_model(model_name: str,  num_tokens: int,  is_completion: bool = False) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


class TokensCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks our LLM info."""
    model_stats: Dict = {}

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.current_request_prompt_tokens: int = 0
        self.current_request_completion_tokens: int = 0
        self.current_request_cost: float = 0.0
        self.current_request_stop_reason: str = ""
        self.current_request_excute_time: float = 0.0
        self.current_request_model_name: str = ""
        self.current_request_prompt_name: str = ""

    def __repr__(self) -> str:
        return f"Total Cost: {self.model_stats}"

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        # print("[ON LLM START]", serialized)
        self.current_request_start_time = perf_counter()
        if serialized:
            if serialized.get("kwargs"):
                # self.current_request_model_name = serialized.get("kwargs").get("model_name")
                self.current_request_model_temperature = serialized.get("kwargs").get("temperature")
                self.current_request_model_max_tokens = serialized.get("kwargs").get("max_tokens")


    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # print("[ON LLM END]", response)
        # print("[ON LLM END]", kwargs)
        self.current_request_excute_time = perf_counter() - self.current_request_start_time
        # Check for usage_metadata (langchain-core >= 0.2.2)
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                else:
                    usage_metadata = None
                    response_metadata = None
            except AttributeError:
                usage_metadata = None
                response_metadata = None
        else:
            usage_metadata = None
            response_metadata = None
        if usage_metadata:
            token_usage = {"total_tokens": usage_metadata["total_tokens"]}
            self.current_request_completion_tokens = usage_metadata["output_tokens"]
            self.current_request_prompt_tokens = usage_metadata["input_tokens"]

            # ### THIS PART FOR STREAMING. 
            # self.current_request_stop_reason =  response_metadata.get("finish_reason")

            # if self.current_request_model_name:
            #     self.current_request_model_name = standardize_model_name(self.current_request_model_name)
            # ###

            if response_model_name := (response_metadata or {}).get("model_name") or (response_metadata or {}).get("model_id"):
                self.current_request_model_name = standardize_model_name(response_model_name)
                self.current_request_stop_reason =  response_metadata.get("finish_reason")
            # elif response.llm_output is None:
            #     self.current_request_model_name = ""
            else:
                if response_model_name := (response.llm_output or {}).get("model_name") or (response.llm_output or {}).get("model_id"):
                    self.current_request_model_name = standardize_model_name(response_model_name)
                    self.current_request_stop_reason =  response_metadata.get("finish_reason")

        else:
            if response.llm_output is None:
                return None

            if "token_usage" not in response.llm_output:
                with self._lock:
                    self.successful_requests += 1
                return None

            # compute tokens and cost for this request
            token_usage = response.llm_output["token_usage"]
            self.current_request_completion_tokens = token_usage.get("completion_tokens", 0)
            self.current_request_prompt_tokens = token_usage.get("prompt_tokens", 0)
            if response_model_name := (response.llm_output or {}).get("model_name") or (response.llm_output or {}).get("model_id"):
                self.current_request_model_name = standardize_model_name(response_model_name)
                self.current_request_stop_reason =  response_metadata.get("finish_reason")
            self.current_request_stop_reason =  response.llm_output.get("finish_reason")

        if self.current_request_model_name in MODEL_COST_PER_1K_TOKENS:
            completion_cost = get_openai_token_cost_for_model(
                self.current_request_model_name, self.current_request_completion_tokens, is_completion=True
            )
            prompt_cost = get_openai_token_cost_for_model(self.current_request_model_name, self.current_request_prompt_tokens)
            self.current_request_cost = prompt_cost + completion_cost
        else:
            self.current_request_cost = 0 

        # update shared state behind lock
        with self._lock:
            if self.current_request_model_name not in self.model_stats:
                self.model_stats[self.current_request_model_name] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "n_requests": 0,
                    "total_cost": 0
                }

            self.model_stats[self.current_request_model_name]["input_tokens"] += self.current_request_prompt_tokens
            self.model_stats[self.current_request_model_name]["output_tokens"] += self.current_request_completion_tokens
            self.model_stats[self.current_request_model_name]["n_requests"] += 1
            self.model_stats[self.current_request_model_name]["total_cost"] += round( self.current_request_cost, 6 )

        
        self.print_current_request_stats()

    def __copy__(self) -> "TokensCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "TokensCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        if serialized:
            if serialized.get("kwargs"):
                self.current_request_prompt_name = serialized.get("kwargs", {}).get("metadata", {}).get("prompt_name", "None")
            
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass

    def print_current_request_stats(self):
        print(f"""[INFO] ---> Prompt Name: {self.current_request_prompt_name} - model_id: {self.current_request_model_name} - Input_Tokens: {self.current_request_prompt_tokens} - Output_Tokens: {self.current_request_completion_tokens} - Cost: {self.current_request_cost:.6f} - Time: {self.current_request_excute_time:.3f} seconds""")
    
    def get_request_stats(self):
        return {
            "input_tokens": self.current_request_prompt_tokens,
            "output_tokens": self.current_request_completion_tokens,
            "total_cost": self.current_request_cost
        }

    def get_total_stats(self):
        return self.model_stats

callback_handler = TokensCallbackHandler()