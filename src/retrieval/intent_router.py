def detect_intent(question):

    q = question.lower()

    if any(k in q for k in [
        "how many","number of","count","total"
    ]):
        return "metadata"

    if any(k in q for k in [
        "latest","recent","new","year"
    ]):
        return "temporal"

    if any(k in q for k in [
        "all go","which go","find go",
        "related go","scheme"
    ]):
        return "discovery"

    if any(k in q for k in [
        "summarize","summary","overall policy"
    ]):
        return "multi_summary"

    return "semantic"