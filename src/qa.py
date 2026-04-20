from __future__ import annotations

from functools import lru_cache

import numpy as np

from .llm_client import LLMClient
from .schemas import QAResult, Transcript


SYSTEM_PROMPT = (
    "You answer questions about a meeting using ONLY the provided excerpts. "
    "Cite your evidence by referencing utterance indices like [3]. "
    "If the excerpts do not contain the answer, say so explicitly."
)


EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(EMBED_MODEL)


class MeetingQA:
    def __init__(self, transcript: Transcript, client: LLMClient):
        self.transcript = transcript
        self.client = client
        texts = [f"[{u.speaker}] {u.text}" for u in transcript.utterances]
        model = _embedder()
        mat = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        self.matrix = mat

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        model = _embedder()
        q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        scores = self.matrix @ q_vec
        return np.argsort(scores)[::-1][:k].tolist()

    def ask(self, question: str, k: int = 5) -> QAResult:
        idxs = self.retrieve(question, k=k)
        excerpts = "\n".join(
            f"[{i}] [{self.transcript.utterances[i].speaker}] {self.transcript.utterances[i].text}"
            for i in idxs
        )
        user = f"EXCERPTS:\n{excerpts}\n\nQUESTION: {question}"
        answer = self.client.chat_text(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return QAResult(question=question, answer=answer, cited_utterance_indices=idxs)
