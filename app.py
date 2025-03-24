import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from enum import Enum
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st
import langsmith

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT_NAME'] = os.getenv('LANGCHAIN_PROJECT_NAME')

class Step(Enum):
    INPUT = "input"
    REVIEW = "review"
    IMPROVISATION = "improvisation"
    APPROVAL = "approval"
    APPROVED = "approved"

# Define the state structure
class CodingState(TypedDict):
    code: str
    step: Step

# Define prompts in a configuration dictionary
PROMPTS = {
    "coder": "Create a code as per the {code} provided",
    "peer": "Review the code provided by the coder: {code}. "
            "If correction is needed, return 'improvisation'. "
            "If suggestions are needed, return 'approval'. "
            "If the code is correct, return 'approved'.",
    "manager": "Add necessary docstrings to the following code and approve it: {code}"
}

# Initialize the LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define the coder node
def coder(state):
    """Based on the input message the code is created by the coder."""
    print("Coder Node: Generating code...")
    try:
        prompt = PromptTemplate.from_template(PROMPTS["coder"])
        chain = prompt | llm
        result = chain.invoke({'code': state['code']})
        print(f"Coder Node: Generated code: {result.content}")
        return {'code': result.content, 'step': Step.REVIEW}
    except Exception as e:
        print(f"Coder Node: Error generating code - {e}")
        return {'code': state['code'], 'step': Step.IMPROVISATION} 

# Define the peer node
def peer(state):
    """Reviewing the code provided by the coder and determining the next step."""
    print("Peer Node: Reviewing code...")
    try:
        prompt = PromptTemplate.from_template(PROMPTS["peer"])
        chain = prompt | llm
        result = chain.invoke({'code': state['code']})

        # Extract the decision step from result
        decision = result.content.strip().lower()

        # Validate decision
        valid_decisions = [Step.IMPROVISATION.value, Step.APPROVAL.value, Step.APPROVED.value]
        if decision not in valid_decisions:
            print(f"Peer Node: Invalid decision '{decision}'. Defaulting to 'approval'.")
            decision = Step.APPROVAL.value  # Default fallback

        return {"code": state["code"], "step": Step(decision)}
    except Exception as e:
        print(f"Peer Node: Error reviewing code - {e}")
        return {"code": state["code"], "step": Step.IMPROVISATION} 

def manager(state):
    """Add docstrings to the code and approve it."""
    print("Manager Node: Adding docstrings and approving code...")
    try:
        prompt = PromptTemplate.from_template(PROMPTS["manager"])
        chain = prompt | llm
        result = chain.invoke({'code': state['code']})
        print(f"Manager Node: Approved code: {result.content}")
        return {'code': result.content, 'step': Step.APPROVED}
    except Exception as e:
        print(f"Manager Node: Error approving code - {e}")
        return {'code': state['code'], 'step': Step.APPROVAL} 

# Define the code validity function
def code_validity(state):
    """Determine the next step based on the current state."""
    print(f"Code Validity: Current step: {state['step'].value}")
    if state['step'] == Step.IMPROVISATION:
        return "coder"
    elif state['step'] == Step.APPROVAL:
        return "manager"
    elif state['step'] == Step.APPROVED:
        return END

# Build the workflow
builder = StateGraph(CodingState)

# Add nodes
builder.add_node("coder", coder)
builder.add_node("peer", peer)
builder.add_node("manager", manager)

# Add edges
builder.add_edge(START, "coder")
builder.add_edge("coder", "peer")
builder.add_conditional_edges("peer", code_validity, {"coder": "coder", "manager": "manager", END: END})
builder.add_edge("manager", END)

# Compile the workflow
workflow = builder.compile()

# Streamlit frontend
st.title("Automated Code Peer Review")
st.write("Submit your code for an automated peer review using an open-source LLM.")

# Text area for code input
code = st.text_area("Paste your code here:", height=300)

if st.button("Generate Code"):
    if code.strip() == "":
        st.error("Please paste some code to review.")
    else:
        with st.spinner("Generating review..."):
            try:
                # Invoke the workflow
                result = workflow.invoke({"code": code, 'step': Step.INPUT})
                st.success("Review Generated!")
                st.write("### Code Review Feedback")
                st.write(result['code'])

            except Exception as e:
                st.error(f"An error occurred: {e}")