conversation_memory = []


def update_memory(question, answer):

    conversation_memory.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_memory) > 5:
        conversation_memory.pop(0)


def get_memory():

    text = ""

    for m in conversation_memory:
        text += f"""
User: {m['question']}
Assistant: {m['answer']}
"""

    return text