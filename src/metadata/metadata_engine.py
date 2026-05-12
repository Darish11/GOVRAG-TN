from collections import defaultdict

#################################################
# BUILD FAST GOVERNMENT KNOWLEDGE INDEXES
#################################################

def build_indexes(metadata):

    dept_index = defaultdict(set)
    year_index = defaultdict(set)

    for chunk in metadata:

        meta = chunk.get("metadata", {})

        go = meta.get("go_number", "")
        dept = meta.get("department", "").lower()
        year = meta.get("year", "")

        if go:

            if dept:
                dept_index[dept].add(go)

            if year:
                year_index[year].add(go)

    return dept_index, year_index


#################################################
# DEPARTMENT ANALYTICS
#################################################

def department_query(question, dept_index):

    q = question.lower()

    for dept in dept_index:

        if dept in q:

            gos = list(dept_index[dept])

            return f"""
There are {len(gos)} Government Orders issued in
{dept.title()} Department.

Sample GOs:
{', '.join(gos[:5])}
"""

    return "Department not found."


#################################################
# TEMPORAL REASONING
#################################################

def temporal_query(question, year_index):

    q = question.lower()

    for year in year_index:

        if year in q:

            gos = list(year_index[year])

            return f"""
Total Government Orders issued in {year}: {len(gos)}

Example:
{', '.join(gos[:5])}
"""

    return None


#################################################
# AUTOMATIC GO DISCOVERY
#################################################

def discover_go(question, retrieved_chunks):

    gos = set()

    for c in retrieved_chunks:

        meta = c.get("metadata", {})
        go = meta.get("go_number", "")

        if go:
            gos.add(go)

    if not gos:
        return "No related Government Orders identified."

    return (
        "Relevant Government Orders discovered:\n\n"
        + "\n".join(list(gos)[:10])
    )