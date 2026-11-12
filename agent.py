import operator
from typing import Annotated, List, Dict, Any, TypedDict
import os
from pydantic import BaseModel, Field
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

from utils import get_deepseek_llm


# 1. Define Structured Output Schemas for DeepSeek
class JobMatch(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name if available, else 'Unknown'")
    url: str = Field(description="Direct URL to the job posting")
    match_score: int = Field(description="Match score from 0 to 100 based on CV alignment")
    rationale: str = Field(description="Brief explanation of why this job matches the CV")

class JobListResponse(BaseModel):
    jobs: List[JobMatch] = Field(description="Filtered list of top matching remote jobs")

# 2. Define the LangGraph State Schema
class JobSearchState(TypedDict):
    resume_text: str
    target_role_query: str
    searched_jobs: List[Dict[str, Any]]         # Found by Agent 1
    selected_jobs: List[Dict[str, Any]]         # Selected at Checkpoint 1 (Human-in-the-loop 1)
    applications: Dict[str, str]                # Drafted by Agent 2 at Checkpoint 2 (Human-in-the-loop 2)
    messages: Annotated[List[Any], operator.add]

# 3. Agent 1: The Job Searcher
def job_searcher_node(state: JobSearchState) -> Dict[str, Any]:
    """
    Agent 1: Searches remote job platforms using Tavily's native domain filtering, evaluates matches against CV using DeepSeek with manual JSON parsing and saves them to state.
    """
    resume = state["resume_text"]
    query = state.get(
        "target_role_query",
        "remote AI engineer python jobs"
    )

    # Initialize Tavily
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from environment variables."
        )

    tavily = TavilyClient(api_key=tavily_api_key)

    # Using Tavily native domain filtering
    search_results = tavily.search(
        query=f"{query} remote",
        include_domains=[
            # "linkedin.com/jobs",
            "boards.greenhouse.io",
            "lever.co",
            "jobs.ashbyhq.com"
        ],
        search_depth="advanced",
        max_results=10,
        time_range="day"            # Restrict to last 24 hours
    )

    # Format search results for DeepSeek evaluation
    context_listings = ""
    for idx, res in enumerate(search_results.get("results", [])):
        context_listings += (
            f"\n[{idx}] Title/Snippet: {res.get('title')}\n"
            f"URL: {res.get('url')}\n"
            f"Content: {res.get('content')}\n"
        )

    # Prompt DeepSeek to filter and score matches against the user's CV
    llm = get_deepseek_llm(temperature=0.1)

    system_prompt = """
    You are an expert technical recruiter and career strategist.
    Analyze the provided candidate CV and evaluate the scraped job listings.
    Select remote jobs matching the candidate's core skill set.
    You MUST return your response strictly as a JSON object matching this exact schema:
    {
        "jobs": [
            {
                "title": "string",
                "company": "string",
                "url": "string",
                "match_score": 0-100 (integer),
                "rationale": "string"
            }
        ]
    }
Do not include any markdown code blocks (like ```json) or extra text. Output raw 
JSON only.
"""

    user_prompt = f"""
    CANDIDATE CV:
    {resume}

    SCRAPPED JOB LISTINGS:
    {context_listings}

    Evaluate these listings and return the JSON response.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Clean up response content in case the model wrapped it in markdown code blocks
    raw_content = response.content.strip()
    
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]

    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    raw_content = raw_content.strip()

    # Parse and validate with Pydantic
    try:
        parsed_response = JobListResponse.model_validate_json(raw_content)
        formatted_jobs = [job.model_dump() for job in parsed_response.jobs]
    except Exception as e:
        # Fallback or empty list of JSON parsing fails
        print(f"JSON parsing error: {e}, Raw content was: {raw_content}")
        formatted_jobs = []

    return {
        "searched_jobs": formatted_jobs,
        "messages": [HumanMessage(
            content=f"Agent 1 identified {len(formatted_jobs)} potential matching jobs."
        )]
    }

# 4. Agent 2: The Application Writer Node
def application_writer_node(state: JobSearchState) -> Dict[str, Any]:
    """
    Agent 2: Takes human-selected jobs from Checkpoint 1 and crafts both tailored cover letters and optimized resume bullet points based on the candidate's CV.
    """
    resume = state["resume_text"]
    selected_jobs = state.get("selected_jobs", [])

    if not selected_jobs:
        return {
            "applications": {},
            "messages": [HumanMessage(
                content="No jobs were selected by the user for application drafting."
            )],
        }

    llm = get_deepseek_llm(temperature=0.3)
    generated_applications: Dict[str, str] = {}

    system_prompt = """
    You are an elite executive career coach, technical resume writer and career strategist.
    For each selected job listing, you must generate two deliverables based on the candidate's CV:
    1. Tailored resume bullet points and key skill highlights optimized specifically to match the job description's requirements.
    2. A compelling, tailored cover letter.
    
    Format your output clearly using Markdown headings:
    ### Tailored Resume Highlights & Bullet Points
    [Optimized resume bullet points and keywords here]
    ### Cover Letter
    [Cover letter text here]
    """

    for job in selected_jobs:
        job_key = f"{job.get('company', 'Company')} - {job.get('title', 'Role')}"
        user_prompt = f"""
        CANDIDATE CV:
        {resume}

        TARGET JOB DETAILS:
        Title: {job.get('title')}
        Company: {job.get('company')}
        URL: {job.get('url')}
        Match Rationale: {job.get('rationale')}

        Generate the optimized resume bullet points and tailored cover letter for this position.
        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        generated_applications[job_key] = response.content

    return {
        "applications": generated_applications,
        "messages": [HumanMessage(
            content=f"Agent 2 drafted application packages (Cover Letter + Resume Bullets) for {len(generated_applications)} selected roles."
        )],
    }

# 5. LangGraph Workflow Construction
def create_job_search_graph():
    """
    Assembles and compiles the two-agent LangGraph system with checkpointing and 
    a native interrupt before Agent 2 (Checkpoint 1).
    """
    workflow = StateGraph(JobSearchState)

    # Add Nodes
    workflow.add_node("job_searcher", job_searcher_node)
    workflow.add_node("application_writer", application_writer_node)

    # Add Edges
    workflow.add_edge(START, "job_searcher")
    workflow.add_edge("job_searcher", "application_writer")
    workflow.add_edge("application_writer", END)

    # Memory Checkpointer enables state saving across human-in-the-loop interactions
    memory = MemorySaver()

    # interrupt_before pauses the graph right before running application_writer
    app = workflow.compile(
        checkpointer=memory, interrupt_before=["application_writer"]
    )

    return app
