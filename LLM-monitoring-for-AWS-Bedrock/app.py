aws_region_name = "us-east-1"
credentials_profile_name = "perso"
claude_3_sonnet = "anthropic.claude-3-sonnet-20240229-v1:0"
mistral_large = "mistral.mistral-large-2402-v1:0"

from langchain_aws.chat_models import ChatBedrock
from langchain_core.runnables import ConfigurableField

mistral_large_bedrock_chat = ChatBedrock(
    model_id=mistral_large,
    credentials_profile_name=credentials_profile_name,
    region_name=aws_region_name,
)

claude_3_sonnet = ChatBedrock(
    model_id=claude_3_sonnet,
    credentials_profile_name=credentials_profile_name,
    region_name=aws_region_name,
)

_model_alternatives = {"mistral_large": mistral_large_bedrock_chat}

bedrock_llm = claude_3_sonnet.configurable_alternatives(
    which=ConfigurableField(
        id="model", name="Model", description="The model that will be used"
    ),
    default_key="claude_3_sonnet",
    **_model_alternatives,
)

from langchain_core.prompts import PromptTemplate

_MISTRAL_PROMPT = PromptTemplate.from_template(
    """
<s>[INST] You are a conversational AI designed to answer in a friendly way to a question.
You should always answer in rhymes.

Human:
<human_reply>
{input}
</human_reply>

Generate the AI's response.[/INST]</s>
"""
)

_CLAUDE_PROMPT = PromptTemplate.from_template(
    """
You are a conversational AI designed to answer in a friendly way to a question.
You should always answer in jokes.

Human:
<human_reply>
{input}
</human_reply>

Assistant:
"""
)

_CHAT_PROMPT_ALTERNATIVES = {"mistral_large": _MISTRAL_PROMPT}

CONFIGURABLE_CHAT_PROMPT = _CLAUDE_PROMPT.configurable_alternatives(
    which=ConfigurableField(
        id="model",
        name="Model",
        description="The model that will be used",
    ),
    default_key="claude_3_sonnet",
    **_CHAT_PROMPT_ALTERNATIVES,
)

from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-lf-06ed5b67-378a-49d6-b562-96e5b500a066",
    secret_key="sk-lf-94107cc4-6984-4cef-8d7e-a8e024e6fce4",
    host="https://cloud.langfuse.com",
)

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableConfig

# chain = CONFIGURABLE_CHAT_PROMPT | bedrock_llm | StrOutputParser()
chain = (CONFIGURABLE_CHAT_PROMPT | bedrock_llm | StrOutputParser()).with_config(
    RunnableConfig(callbacks=[langfuse_handler])
)

chain.with_config(configurable={"model": "mistral_large"}).invoke(
    "What is a large language model ?"
)
