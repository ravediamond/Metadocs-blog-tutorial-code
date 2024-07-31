import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Metadocs Langserve Tutorial",
    version="0.1",
    description="Spin up a LangServe API for learning purpose",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


template = """"Answer the question by first detailing the steps or considerations you think are important for solving it. After explaining each part, provide the final answer.
If there is any aspect you are unsure about, describe what you know and what remains unclear
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
    openai_api_key=os.environ["OPENAI_KEY"],
)
chain = prompt | model | StrOutputParser()


add_routes(app, chain, path="/chat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
