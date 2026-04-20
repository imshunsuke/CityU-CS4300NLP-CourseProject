from __future__ import annotations

from .llm_client import LLMClient
from .schemas import Language, TranslatedTranscript, TranslationPair, Transcript, Utterance


SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter for business meetings. "
    "Translate the user's utterance into the requested target language. "
    "Preserve speaker tone, idiom, and meeting context from the provided prior turns. "
    "Return ONLY the translated text, no prefix, no quotes."
)


def _guess_lang(text: str) -> Language:
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if chinese >= max(3, int(0.15 * len(text))) else "en"


def translate_utterance_llm(
    utterance: Utterance,
    context: list[Utterance],
    target_lang: Language,
    client: LLMClient,
) -> str:
    context_text = "\n".join(f"[{u.speaker}] {u.text}" for u in context[-3:])
    user = (
        f"Target language: {target_lang}.\n"
        f"Prior turns (for context, do not translate):\n{context_text}\n\n"
        f"Translate this turn:\n[{utterance.speaker}] {utterance.text}"
    )
    return client.chat_text(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    ).strip()


def translate_transcript(
    transcript: Transcript,
    target_lang: Language,
    client: LLMClient,
) -> TranslatedTranscript:
    pairs: list[TranslationPair] = []
    for i, u in enumerate(transcript.utterances):
        src_lang: Language = u.lang if u.lang != "auto" else _guess_lang(u.text)
        if src_lang == target_lang:
            pairs.append(
                TranslationPair(
                    source=u.text, source_lang=src_lang, target=u.text, target_lang=target_lang
                )
            )
            continue
        tgt = translate_utterance_llm(
            u, transcript.utterances[max(0, i - 3):i], target_lang, client
        )
        pairs.append(
            TranslationPair(
                source=u.text, source_lang=src_lang, target=tgt, target_lang=target_lang
            )
        )
    return TranslatedTranscript(pairs=pairs, target_lang=target_lang)


def translate_with_nllb(
    transcript: Transcript,
    target_lang: Language,
    *,
    model_name: str = "facebook/nllb-200-distilled-600M",
) -> TranslatedTranscript:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    NLLB_CODES = {"en": "eng_Latn", "zh": "zho_Hans"}
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tgt_code = NLLB_CODES[target_lang]

    pairs: list[TranslationPair] = []
    for u in transcript.utterances:
        src_lang: Language = u.lang if u.lang != "auto" else _guess_lang(u.text)
        tok.src_lang = NLLB_CODES[src_lang]
        inputs = tok(u.text, return_tensors="pt", truncation=True, max_length=512)
        out = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
            max_length=512,
        )
        tgt = tok.batch_decode(out, skip_special_tokens=True)[0]
        pairs.append(
            TranslationPair(source=u.text, source_lang=src_lang, target=tgt, target_lang=target_lang)
        )
    return TranslatedTranscript(pairs=pairs, target_lang=target_lang)
