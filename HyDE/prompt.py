class PromptGen:
    WEB_SEARCH = """
Please write a passage to answer the question.
Question: {}
Passage:
    """

    SCIFACT = """
Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:
    """

    ARGUANA = """
Please write a counter argument for the passage.
Passage: {}
Counter Argument:
    """

    TREC_COVID = """
Please write a scientific paper passage to answer the question.
Question: {}
Passage:
    """

    FIQA = """
Please write a financial article passage to answer the question.
Question: {}
Passage:
    """

    DBPEDIA_ENTITY = """
Please write a passage to answer the question.
Question: {}
Passage:
    """

    TREC_NEWS = """
Please write a news passage about the topic.
Topic: {}
Passage:
    """

    MR_TYDI = """
Please write a passage in {} to answer the question in detail.
Question: {}
Passage:
    """

    SUMMARY = """
Please write a summary to answer the question in detail.
Question: {}
Passage:
    """
