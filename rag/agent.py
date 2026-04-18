import re
from datetime import datetime
from typing_extensions import TypedDict
from typing import Literal, Annotated, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver  

from .prompts import SYSTEM_PROMPT
from .tools import search_kb
from .utils import callback_handler
from .clients import llm
from .parse_files import extract_text_from_file

TOOL_CALL_LIMITS = {
    "search_kb": 3
}

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    answer: str
    chunks: List[Dict[str, str]]
    tool_call: Dict[str, Dict[str, Any]] # Structure: {"tool_name": {"count": 0, "inputs": []}}
    force_search_kb_tool: float

tools = [search_kb]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
tools_description = "\n\n".join([f'{tool.name}\n{tool.description}' for tool in tools])

def check_tool_call_limit(state: AgentState, tool_call_name: str):
    """Check if the tool call limit is exceeded"""
    tool_call_num = state.get("tool_call", {}).get(tool_call_name, {}).get("count", 0)
    tool_call_limit = TOOL_CALL_LIMITS.get(tool_call_name, 3)

    if tool_call_num > tool_call_limit:
        return True
    return False

def update_tool_call(state: AgentState, tool_call_name: str, tool_call_args: dict, tool_call_outupt: dict):
    """Update the tool call count and arguments"""
    if "tool_call" not in state:
        state["tool_call"] = {}

    if tool_call_name not in state["tool_call"]:
        state["tool_call"][tool_call_name] = {"count": 0, "inputs": [], "outputs": []}

    state["tool_call"][tool_call_name]["count"] += 1
    state["tool_call"][tool_call_name]["inputs"].append(tool_call_args)
    state["tool_call"][tool_call_name]["outputs"].append(tool_call_outupt)

def format_chunks(tool_call):
    ordered_chunks = []
    
    search_data = tool_call.get("search_kb", {})
    outputs = search_data.get("outputs", [])
    
    if not outputs or not isinstance(outputs[0], list):
        return []

    for idx, chunk in enumerate(outputs[0], 1):
        chunk_file_name = chunk.metadata.get("file_name")
        
        ordered_chunks.append({
            "order": idx,
            "source_name": chunk_file_name,
            "data": chunk.model_dump() 
        })
        
    return ordered_chunks

def init_state(state: AgentState):
    state["chunks"] = []
    state["tool_call"] = {}
    state["force_search_kb_tool"] = 0.0
    return state

def intent_router(state: AgentState):
    # force search kb based on keyword search + semantic search
    # to mimic the hybrid vector db.
    WE_KEYWORDS = [
        "we ", "وي", "plan", "خطة", "price", "سعر", "offer", "عرض", 
        "internet", "انترنت", "router", "5g", "4g", "prepaid", "postpaid", "وى"
    ]
    force_search_kb = 0
    last_msg = state["messages"][-1].content.lower()
    if any(kw in last_msg for kw in WE_KEYWORDS):
        force_search_kb += 0.25
    results = search_kb.invoke({"query": last_msg, "k_results": 5})
    docs_score = [doc.metadata.get("similarity_score") for doc in results]
    force_search_kb += sum(docs_score) / len(results) if results else 0
    state["force_search_kb_tool"] = force_search_kb
    return state

def llm_call(state: AgentState):
    """LLM decides whether to call a tool or not"""

    system_prompt_1 = [
        SystemMessage(
            content=[
                {"text": SYSTEM_PROMPT.format(TOOLS_PLACEHOLDER=tools_description)},
                {"cachePoint": {"type": "default"}},
            ]
        )
    ]
    system_prompt_2 = [SystemMessage(content=f"Current Time is {datetime.now()}")]

    # check tool call or not.
    already_called = bool(state.get("tool_call", {}).get("search_kb"))

    if state["force_search_kb_tool"] >= 0.5 and not already_called:
        use_llm = llm.bind_tools(tools, tool_choice="search_kb")
        system_prompt_3 = [SystemMessage(content="**MANDATORY**: You MUST call **search_kb** before answering this question.")]
        messages = system_prompt_1 + system_prompt_2 + system_prompt_3 + state["messages"]
    else:
        use_llm = llm_with_tools
        messages = system_prompt_1 + system_prompt_2 + state["messages"]

    state["messages"] = [use_llm.invoke(messages)]
    return state

def tool_node(state: AgentState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # check tool call limit
        if check_tool_call_limit(state, tool_name):
            tool_result = (
                f"SYSTEM ERROR: The tool '{tool_name}' has exceeded its execution limit. "
                "DO NOT try to call this tool again. "
                "If the current context is enough, provide the final answer now. "
                "If you are missing critical info, politely ask the user to provide the missing details."
            )
        else:
            tool = tools_by_name[tool_name]
            tool_result = tool.invoke(tool_args)
            update_tool_call(state, tool_name, tool_args, tool_result)
        
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        state["messages"] = result
    return state

def output_response(state: AgentState):
    final_answer = state["messages"][-1].content
    match = re.search(r'<thinking>(.*?)</thinking>', final_answer, re.DOTALL)
    thinking_variable = match.group(1).strip() if match else None
    state["answer"] = re.sub(r'<thinking>.*?</thinking>', '', final_answer, flags=re.DOTALL).strip()
    state["chunks"] = format_chunks(state.get("tool_call", {}))
    return state

def should_continue(state: AgentState) -> Literal["Action", "END"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return "END"

agent_builder = StateGraph(AgentState)
agent_builder.add_node("init_state", init_state)
agent_builder.add_node("intent_router", intent_router)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_node("output_response", output_response)

agent_builder.add_edge(START, "init_state")
agent_builder.add_edge("init_state", "intent_router")
agent_builder.add_edge("intent_router", "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        "END": "output_response",
    },
)
agent_builder.add_edge("environment", "llm_call")
agent_builder.add_edge("output_response", END)

checkpointer = InMemorySaver()

agent = agent_builder.compile(checkpointer=checkpointer)

from time import perf_counter

def ask_chatbot(session_id: str, question: str) -> tuple[AgentState, Dict[str, int]]:
    start_time = perf_counter()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "callbacks": [callback_handler],
            "configurable": {"thread_id": session_id},
        },
    )
    exec_time = perf_counter() - start_time
    request_stats = callback_handler.get_request_stats()
    request_stats["exec_time"] = exec_time
    print(f"[INFO] Response: {response}")
    print(f"[INFO] Total Stats: {callback_handler.get_total_stats()}")
    return response, request_stats

def ask_chatbot_with_files(session_id: str, question: str, files_paths: List[str]) -> tuple[AgentState, Dict[str, int]]:
    if files_paths:
        print(f"[INFO] Start Processing these files: {files_paths}")
        raw_text = ""
        for file_path in files_paths:
            raw_text += extract_text_from_file(file_path) + "\n---------------------------\n"

        question = (
            "The User Uploaded Documents, need to use it as a reference to answer the question.\n\n"
            f"Attached Document Content:\n\n{raw_text}"
            f"User Question: {question}"
        )

    return ask_chatbot(session_id, question)

if __name__ == "__main__":
    response, stats = ask_chatbot("user1", "ما هى عاصمة مصر")
    print(response)
    print(stats)
    print("\n")
    for msg in response["messages"]:
        msg.pretty_print()
