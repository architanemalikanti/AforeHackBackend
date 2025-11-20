from typing import TypedDict, Annotated
import operator
import os
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system="", fallback_model=None):
        self.system = system
        self.fallback_model = fallback_model
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        if fallback_model:
            self.fallback_model_bound = fallback_model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        try:
            message = self.model.invoke(messages)
            return {"messages": [message]}
        except Exception as e:
            # Check if it's an Anthropic overload error
            error_str = str(e)
            is_overload = "overloaded_error" in error_str or "Overloaded" in error_str

            if is_overload and self.fallback_model:
                print(f"⚠️ Anthropic overloaded, falling back to OpenAI...")
                try:
                    message = self.fallback_model_bound.invoke(messages)
                    return {"messages": [message]}
                except Exception as fallback_error:
                    print(f"❌ FALLBACK ERROR: {type(fallback_error).__name__}: {str(fallback_error)}")
                    raise fallback_error
            else:
                # Log the actual error before it gets masked by httpx
                print(f"❌ API ERROR: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            try:
                result = self.tools[t["name"]].invoke(t["args"])
                results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
            except Exception as e:
                # If tool fails, return error message so conversation can continue
                error_msg = f"Error calling {t['name']}: {str(e)}"
                print(f"⚠️ Tool execution failed: {error_msg}")
                results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=error_msg))
        return {"messages": results}

