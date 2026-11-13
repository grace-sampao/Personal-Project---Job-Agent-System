import os
import tempfile

import streamlit as st

from agent import create_job_search_graph
from utils import extract_resume_text


st.set_page_config(
    page_title="AI Job Search & Application Assistant",
    page_icon="💼",
    layout="wide"
)

# Initialize Session State variables for tracking workflow stage
if "graph" not in st.session_state:
    st.session_state.graph = create_job_search_graph("job_sessions.sqlite")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_session"

st.title("💼 Autonomous Multi-Agent Job Search & Application Assistant")
st.markdown(
    "Powered by **DeepSeek**, **Tavily** and **LangGraph**. "
    "This system searches remote roles, lets you review and select them (**Checkpoint 1**), "
    "and drafts tailored application responses for your manual submission (**Checkpoint 2**)."
)

# Sidebar for configuration and inputs
with st.sidebar:
    st.header("1. Session & Input Configuration")

    # Allow switching or resuming specific saved sessions by ID
    session_input = st.text_input(
        "Session ID (Use same ID to resume)",
        value=st.session_state.thread_id
    )

    if session_input != st.session_state.thread_id:
        st.session_state.thread_id = session_input
        st.rerun()

    st.session_state.thread_config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    # PDF Resume Uploader
    uploaded_file = st.file_uploader("Upload your CV / Resume (PDF)", type=["pdf"])

    # Target role search query
    target_query = st.text_input(
        "Search Query / Role Focus",
        value="remote AI engineer python jobs"
    )

    run_search_btn = st.button("🚀 Run Agent 1: Search & Evaluate", type="primary")

    # Main execution logic
    config = st.session_state.thread_config
    graph = st.session_state.graph

    # Handle Agent 1 execution trigger
    if run_search_btn:
        if not uploaded_file:
            st.error("Please upload your PDF resume first.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Extracting resume text and running Agent 1 (Tavily search & DeepSeek scoring)..."):
                    resume_text = extract_resume_text(tmp_path)

                    # Initial state payload
                    initial_state = {
                        "resume_text": resume_text,
                        "target_role_query": target_query,
                        "searched_jobs": [],
                        "selected_jobs": [],
                        "applications": {},
                        "messages": []
                    }

                    # Run graph until it hits the interrupt_before=['application_writer'] breakpoint
                    events = graph.stream(initial_state, config, stream_mode="values")
                    for event in events:
                        pass            # Process runs until interrupt

                st.success("Agent 1 search completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"An error occured during execution: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Retrieve current state from LangGraph checkpointer
current_state_snapshot = graph.get_state(config)
state_values = current_state_snapshot.values if current_state_snapshot else {}
searched_jobs = state_values.get("searched_jobs", [])
applications = state_values.get("applications", {})

# Checkpoint 1: Job Review & Selection UI
if searched_jobs and not applications:
    st.markdown("---")
    st.header("🛑 Human-in-the-Loop Checkpoint #1: Review & Select Jobs")
    st.markdown("Agent 1 found the following matching remote roles posted in the last 24 hours. Select the ones you want to apply for:")

    selected_indices = []

    with st.form("job_selection_form"):
        for idx, job in enumerate(searched_jobs):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.subheader(f"{job.get('title')} @ {job.get('company', 'Unknown Company')}")
                st.markdown(f"**Match Score:** `{job.get('match_score')}/100`")
                st.markdown(f"**Rationale:** {job.get('rationale')}")
                st.markdown(f"[🔗 Direct Link to Job Posting]({job.get('url')})")
            with col2:
                is_checked = st.checkbox("Select Role", key=f"job_select_{idx}")
                if is_checked:
                    selected_indices.append(idx)
            st.markdown("---")

        generate_apps_btn = st.form_submit_button("✍🏽 Approve Selected & Run Agent 2 (Draft Applications)", type="primary")

    if generate_apps_btn:
        if not selected_indices:
            st.warning("Please select at least one job before proceeding.")
        else:
            # Collect user-approved jobs
            approved_jobs = [searched_jobs[i] for i in selected_indices]

            # Update the graph state with the selected jobs
            graph.update_state(config, {"selected_jobs": approved_jobs})

            with st.spinner("Agent 2 is drafting tailored cover letters and application answers..."):
                # Resume execution from the checkpoint (runs application_write node)
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    pass

            st.success("Applications drafted successfully!")
            st.rerun()

# Checkpoint 2: Application Review & Manual Submission UI
if applications:
    st.markdown("---")
    st.header("🛑 Human-in-the-Loop Checkpoint #2: Review Drafted Application Packages")
    st.markdown("Here are the custom cover letters and optimized resume bullets drafted by Agent 2. Review them, copy the text into your application materials and submit manually.")

    for job_title, app_content in applications.items():
        with st.expander(f"📝 Application Package: {job_title}", expanded=True):
            st.markdown(app_content)
            st.markdown("---")
            st.markdown("*(Copy the relevant sections above, open the job posting link and submit your application manually).*")

    if st.button("🔄 Reset & Start New Search"):
        # Clear thread state or implement thread ID for a fresh session
        st.session_state.thread_config = {"configurable": {
            "thread_id": f"user_job_session_{os.urandom(4).hex()}"
        }}
        st.rerun()
