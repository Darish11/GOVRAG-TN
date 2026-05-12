import ollama

# ===================================
# GENERATION FUNCTION (CHATBOT USE)
# ===================================

def build_prompt(context_chunks, question):

    structured_context = []

    for chunk in context_chunks:

        meta = chunk.get("metadata", {})

        go_no = meta.get("go_number", "Unknown GO")
        dept = meta.get("department", "Unknown Department")
        year = meta.get("year", "Unknown Year")
        clause = (
            meta.get("section_id")
            or meta.get("clause_no")
            or "-"
        )

        structured_context.append(
            f"""
GO Number: {go_no}
Department: {dept}
Year: {year}
Clause: {clause}

Content:
{chunk['text']}
"""
        )

    context = "\n\n".join(structured_context)

    return f"""
You are GOVRAG — Tamil Nadu Government Order Assistant.

STRICT RULES:
1. ALWAYS mention:
   - GO Number
   - Department Name
   - Year
   - Clause/Section
2. Use ONLY provided context.
3. If multiple GOs are relevant, summarize them.
4. If answer not present say EXACTLY:
"I do not have information about this Government Order."

---------------------
CONTEXT
{context}
---------------------

QUESTION:
{question}

FINAL ANSWER:
"""


def generate_answer(query, retrieved_chunks):

    prompt = build_prompt(
        retrieved_chunks,
        query
    )

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    citations = []

    for chunk in retrieved_chunks:

        meta = chunk.get("metadata", {})

        citations.append({
            "go_number": meta.get("go_number", ""),
            "department": meta.get("department", ""),
            "year": meta.get("year", ""),
            "clause": meta.get("section_id")
                    or meta.get("clause_no")
        })

    return answer, citations