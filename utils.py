import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from pypdf import PdfReader


# Load environment variables from .env file
load_dotenv()

def extract_resume_text(pdf_path: str) -> str:
    """
    Extracts and concatenates text from a PDF resume.

    Args:
        pdf_path (str): The local file path to the PDF resume.

    Returns:
        str: The clean textual content extracted from the resume.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume file not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    resume_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            resume_text += f"\n--- Page {i + 1} ---\n" + text

    return resume_text.strip()

def get_deepseek_llm(temperature: float = 0.1) -> ChatOpenAI:
    """
    Initializes and returns the DeepSeek LLM client via LangChain.

    Architectural Rationale:
        DeepSeek uses an OpenAI-compatible schema. By utilizing LangChain's 
        ChatOpenAI class with a custom base_url, we gain immediate access to 
        LangChain's robust tool-calling, structured output parsing and agent 
        frameworks without needing a specialized wrapper.

    Args:
        temperature (float): Low temperature (e.g., 0.1) ensures deterministic, 
        factual outputs when matching CV requirements against jobs.

    Returns:
        ChatOpenAI: The configured DeepSeek chat model instance.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is missing from environment variables in .env"
        )

    return ChatOpenAI(
        model="deepseek-chat",          # DeepSeek-V3 fast and powerful chat model
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=temperature,
        max_tokens=2000
    )
