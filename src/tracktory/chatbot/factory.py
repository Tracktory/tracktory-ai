"""챗봇 그래프 기본 의존성 조립

콘솔(chatbot/main.py)·FastAPI(api/main.py lifespan) 양쪽이 공유하는 단일 진입점
checkpointer 만 주입받고 LLM·retriever·체인 2종은 기본값으로 묶어 컴파일
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from tracktory.chatbot.graph import build_chatbot_graph
from tracktory.chatbot.rag.ragflow import RagFlowChatbotRetriever
from tracktory.common.config import config as common_config
from tracktory.prompts.chatbot.general_advice import GENERAL_ADVICE_PROMPT
from tracktory.prompts.chatbot.intent import (
    INTENT_CLASSIFIER_PROMPT,
    IntentClassification,
)
from tracktory.prompts.chatbot.rag_response import (
    RAG_RESPONSE_PROMPT,
    ChatbotResponse,
)

CHECKPOINT_DB_PATH = common_config.PROJECT_ROOT / "data" / "checkpoints" / "api" / "chatbot.sqlite"


def build_default_chatbot_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """기본 LLM·retriever·체인 2종으로 챗봇 그래프 컴파일

    LLM 은 gpt-4o-mini temperature=0 고정
    retriever 는 RAGFlow 기본 설정 (settings 에서 인증·dataset 읽음)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # with_structured_output 반환 타입이 mypy 에서 generic 으로 안 좁혀짐 (runtime 은 정확)
    return build_chatbot_graph(
        checkpointer=checkpointer,
        classifier=INTENT_CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification),  # type: ignore[arg-type]
        retriever=RagFlowChatbotRetriever(),
        rag_response_chain=RAG_RESPONSE_PROMPT | llm.with_structured_output(ChatbotResponse),  # type: ignore[arg-type]
        general_advice_chain=GENERAL_ADVICE_PROMPT | llm.with_structured_output(ChatbotResponse),  # type: ignore[arg-type]
    )
