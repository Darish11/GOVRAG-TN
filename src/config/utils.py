def build_context(chunks):

    context=""

    for c in chunks:

        go=c["metadata"].get("go_number","")
        clause=c["metadata"].get("section_id","")

        context+=f"""
[GO:{go} | Clause:{clause}]
{c.get("text","")}
"""

    return context