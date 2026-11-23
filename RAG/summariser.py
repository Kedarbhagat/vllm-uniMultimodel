from transformers import pipeline

# Load ONCE at startup (IMPORTANT)
_title_summarizer = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU only
)

async def generate_summary(messages):
    """
    messages: list of message dicts from DB
    [
        {"role": "human", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    Returns: short title string
    """

    # ---- Minimal context selection ----
    # First user message + second user message
    user_msgs = messages

    if len(user_msgs) < 2:
        return None

    selected_msgs = user_msgs[:2]

    conversation = ""
    for msg in selected_msgs:
        conversation += f"User: {msg['content']}\n"

    prompt = (
        "Summarize the topic of this conversation in 4 to 6 words. "
        "Do not use punctuation. Do not add extra words.\n\n"
        f"Conversation:\n{conversation}"
    )

    result = _title_summarizer(
        prompt,
        max_new_tokens=12,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
    )

    return result[0]["generated_text"].strip()
