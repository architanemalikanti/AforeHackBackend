from typing import TypedDict, Annotated
import operator
import os
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
import anthropic


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
        except anthropic.APIStatusError as e:
            # Catch Anthropic API errors specifically (before they get wrapped by httpx)
            error_body = getattr(e, 'body', {})
            error_type = error_body.get('error', {}).get('type', '') if isinstance(error_body, dict) else ''

            print(f"🔍 DEBUG: Caught APIStatusError - type: {error_type}")

            if error_type == 'overloaded_error' and self.fallback_model:
                print(f"⚠️ Anthropic overloaded, falling back to OpenAI with streaming...")
                try:
                    # Use streaming with OpenAI to maintain the same behavior
                    full_content = ""
                    tool_calls_list = []

                    for chunk in self.fallback_model_bound.stream(messages):
                        # Accumulate content and tool calls from stream
                        if hasattr(chunk, 'content') and chunk.content:
                            full_content += chunk.content
                        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                            tool_calls_list.extend(chunk.tool_calls)

                    # Create the final AIMessage with accumulated content
                    final_message = AIMessage(
                        content=full_content,
                        tool_calls=tool_calls_list if tool_calls_list else []
                    )

                    return {"messages": [final_message]}
                except Exception as fallback_error:
                    print(f"❌ FALLBACK ERROR: {type(fallback_error).__name__}: {str(fallback_error)}")
                    raise fallback_error
            else:
                # Re-raise if not overload error or no fallback available
                print(f"❌ ANTHROPIC API ERROR: {type(e).__name__}: {str(e)}")
                raise
        except Exception as e:
            # Catch any other errors (like httpx.ResponseNotRead)
            error_str = str(e)

            # Check the exception chain for the original Anthropic error
            original_exception = e
            while original_exception is not None:
                if isinstance(original_exception, anthropic.APIStatusError):
                    error_body = getattr(original_exception, 'body', {})
                    error_type = error_body.get('error', {}).get('type', '') if isinstance(error_body, dict) else ''

                    print(f"🔍 DEBUG: Found APIStatusError in exception chain - type: {error_type}")

                    if error_type == 'overloaded_error' and self.fallback_model:
                        print(f"⚠️ Anthropic overloaded (found in exception chain), falling back to OpenAI with streaming...")
                        try:
                            # Use streaming with OpenAI to maintain the same behavior
                            full_content = ""
                            tool_calls_list = []

                            for chunk in self.fallback_model_bound.stream(messages):
                                # Accumulate content and tool calls from stream
                                if hasattr(chunk, 'content') and chunk.content:
                                    full_content += chunk.content
                                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                                    tool_calls_list.extend(chunk.tool_calls)

                            # Create the final AIMessage with accumulated content
                            final_message = AIMessage(
                                content=full_content,
                                tool_calls=tool_calls_list if tool_calls_list else []
                            )

                            return {"messages": [final_message]}
                        except Exception as fallback_error:
                            print(f"❌ FALLBACK ERROR: {type(fallback_error).__name__}: {str(fallback_error)}")
                            raise fallback_error
                    break

                # Walk up the exception chain
                original_exception = getattr(original_exception, '__context__', None)

            # If no Anthropic error found in chain, check string
            is_overload = "overloaded_error" in error_str or "Overloaded" in error_str

            print(f"🔍 DEBUG: Caught {type(e).__name__}: {error_str}")
            print(f"🔍 DEBUG: is_overload (from string) = {is_overload}")

            if is_overload and self.fallback_model:
                print(f"⚠️ Anthropic overloaded (detected in wrapped error string), falling back to OpenAI with streaming...")
                try:
                    # Use streaming with OpenAI to maintain the same behavior
                    full_content = ""
                    tool_calls_list = []

                    for chunk in self.fallback_model_bound.stream(messages):
                        # Accumulate content and tool calls from stream
                        if hasattr(chunk, 'content') and chunk.content:
                            full_content += chunk.content
                        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                            tool_calls_list.extend(chunk.tool_calls)

                    # Create the final AIMessage with accumulated content
                    final_message = AIMessage(
                        content=full_content,
                        tool_calls=tool_calls_list if tool_calls_list else []
                    )

                    return {"messages": [final_message]}
                except Exception as fallback_error:
                    print(f"❌ FALLBACK ERROR: {type(fallback_error).__name__}: {str(fallback_error)}")
                    raise fallback_error
            else:
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

