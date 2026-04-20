from __future__ import annotations

import numpy as np

from .llm_client import LLMClient
from .schemas import QAResult, Transcript


SYSTEM_PROMPT = (
    "You answer questions about a meeting using ONLY the provided excerpts. "
    "Cite your evidence by referencing utterance indices like [3]. "
    "If the excerpts do not contain the answer, say so explicitly."
)


class MeetingQA:
    def __init__(self, transcript: Transcript, client: LLMClient):
        self.transcript = transcript
        self.client = client
        texts = [f"[{u.speaker}] {u.text}" for u in transcript.utterances]
        embeddings = client.embed(texts)
        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        self.matrix = mat / norms

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        q_vec = np.array(self.client.embed([query])[0], dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) + 1e-9
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
