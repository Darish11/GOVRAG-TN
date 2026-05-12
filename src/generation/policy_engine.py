from collections import defaultdict


def group_by_go(chunks):

    grouped = defaultdict(list)

    for c in chunks:
        meta = c.get("metadata", {})
        go = meta.get("go_number", "UNKNOWN")

        grouped[go].append(c)

    return grouped


def temporal_sort(grouped):

    def year_of(chunks):
        for c in chunks:
            y = c["metadata"].get("year")
            if y and str(y).isdigit():
                return int(y)
        return 0

    return dict(
        sorted(
            grouped.items(),
            key=lambda x: year_of(x[1]),
            reverse=True
        )
    )


def build_multi_go_context(grouped, max_go=5):

    context = ""

    count = 0

    for go, chunks in grouped.items():

        if go in ["", "UNKNOWN"]:
            continue

        context += f"\n===== {go} =====\n"

        for c in chunks[:3]:
            context += c["text"][:800] + "\n"

        count += 1
        if count >= max_go:
            break

    return context