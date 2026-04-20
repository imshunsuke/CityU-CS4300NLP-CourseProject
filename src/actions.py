from __future__ import annotations

from .llm_client import LLMClient
from .schemas import ActionItem, ActionItemList, Transcript


SYSTEM_PROMPT = (
    "You analyze meeting transcripts and extract commitments (action items). "
    "An action item is a concrete task someone agreed to do, with a clear owner. "
    "Do not invent tasks. If a task has no clear owner, choose the speaker who proposed it. "
    "Use the evidence_span field to quote the transcript line that supports each item."
)


def _action_tool(speakers: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_action_items",
            "description": "Return a list of action items extracted from the meeting transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Short imperative description of the task.",
                                },
                                "assignee": {
                                    "type": "string",
                                    "enum": speakers or ["UNKNOWN"],
                                    "description": "Speaker responsible for completing the task.",
                                },
                                "due": {
                                    "type": "string",
                                    "description": "Due date or deadline as stated in the meeting (free text). Empty if none.",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["H", "M", "L"],
                                },
                                "evidence_span": {
                                    "type": "string",
                                    "description": "Direct quote from transcript supporting this action item.",
                                },
                            },
                            "required": ["task", "assignee", "priority", "evidence_span"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def extract_action_items(transcript: Transcript, client: LLMClient) -> ActionItemList:
    tool = _action_tool(transcript.speakers)
    user = (
        f"Speakers in this meeting: {', '.join(transcript.speakers)}.\n\n"
        f"TRANSCRIPT:\n{transcript.as_plain_text(with_speaker=True)}\n\n"
        "Extract every action item and submit via the function call."
    )
    args = client.chat_tool(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tool=tool,
        temperature=0.1,
    )
    items = [ActionItem.model_validate(it) for it in args.get("items", [])]
    return ActionItemList(items=items)
