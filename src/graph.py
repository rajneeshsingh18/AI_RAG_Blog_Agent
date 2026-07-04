from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from functools import partial

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.llm_provider import get_chat_model

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Edges
## Check Relevance
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    provider = st.session_state.get("provider", "ollama")
    model = get_chat_model(
        provider=provider,
        api_key=st.session_state.gemini_api_key
    )

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
# Nodes
## agent node
def agent(state, tools):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    provider = st.session_state.get("provider", "ollama")
    model = get_chat_model(
        provider=provider,
        api_key=st.session_state.gemini_api_key
    )
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    
    # Robust parsing for local Ollama tool calling
    if provider == "ollama" and response.content and not response.tool_calls:
        import json
        import uuid
        content = response.content.strip()
        
        # Sometimes models wrap JSON in markdown block: ```json ... ```
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) >= 3:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                tool_calls_list = parsed
            elif isinstance(parsed, dict):
                tool_calls_list = [parsed]
            else:
                tool_calls_list = []
                
            tool_calls = []
            for tc in tool_calls_list:
                if isinstance(tc, dict) and "name" in tc:
                    args = tc.get("args") or tc.get("arguments") or {}
                    tool_calls.append({
                        "name": tc["name"],
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:12]}"
                    })
            if tool_calls:
                response.tool_calls = tool_calls
                response.content = ""
        except (json.JSONDecodeError, ValueError):
            pass
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

## rewrite node
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                    Here is the initial question:
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    Formulate an improved question: """,
        )
    ]

    # Grader
    provider = st.session_state.get("provider", "ollama")
    model = get_chat_model(
        provider=provider,
        api_key=st.session_state.gemini_api_key
    )
    response = model.invoke(msg)
    return {"messages": [response]}

## generate node
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Initialize a Chat Prompt Template
    prompt_template = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:""",
        input_variables=["question", "context"]
    )

    # Initialize a Generator (i.e. Chat Model)
    provider = st.session_state.get("provider", "ollama")
    chat_model = get_chat_model(
        provider=provider,
        api_key=st.session_state.gemini_api_key
    )

    # Initialize a Output Parser
    output_parser = StrOutputParser()
    
    # RAG Chain
    rag_chain = prompt_template | chat_model | output_parser

    response = rag_chain.invoke({"context": docs, "question": question})
    
    return {"messages": [response]}

# graph function
def get_graph(retriever_tool):
    tools = [retriever_tool]  # Create tools list here
    
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Use partial to pass tools to the agent function
    workflow.add_node("agent", partial(agent, tools=tools))
    
    # Rest of the graph setup remains the same
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    return graph

def generate_message(graph, inputs):
    generated_message = ""

    for output in graph.stream(inputs):
        for key, value in output.items():
            if key in ["generate", "agent"] and isinstance(value, dict):
                msgs = value.get("messages", [])
                if msgs:
                    msg = msgs[0]
                    # Extract string content whether it's an AIMessage object or a raw string
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    
                    # Update message if it has text (ignores empty tool-call-only messages)
                    if content:
                        generated_message = content
    
    return generated_message
