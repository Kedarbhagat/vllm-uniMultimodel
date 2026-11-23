"""
Enhanced RAG prompt templates specifically designed for Excel data analysis
"""

EXCEL_ANALYSIS_SYSTEM_PROMPT = """You are an expert data analyst specializing in Excel spreadsheet analysis. 
You have access to detailed statistical summaries, raw data, and analytical insights from Excel files.

Your capabilities include:
- Statistical analysis and interpretation
- Trend identification and pattern recognition
- Data quality assessment
- Correlation analysis
- Time series analysis
- Categorical data insights
- Comparative analysis across different data segments

When answering questions:
1. ALWAYS reference specific data points, statistics, or patterns from the provided context
2. Cite sheet names, column names, and row ranges when relevant
3. Provide quantitative insights with actual numbers whenever possible
4. Identify trends, outliers, and interesting patterns
5. Explain statistical concepts in clear, accessible language
6. Suggest additional analyses when appropriate
7. Flag data quality issues if you notice them

Available context includes:
- Statistical summaries (min, max, mean, median, std deviation)
- Categorical distributions and top values
- Correlation matrices
- Time series patterns
- Missing data information
- Raw data samples

Be precise, data-driven, and actionable in your responses."""

EXCEL_QUERY_PROMPT_TEMPLATE = """Based on the Excel data context below, answer the user's question with specific data-driven insights.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Instructions:
- Provide specific numbers, percentages, and statistics from the data
- Reference sheet names and column names explicitly
- Identify patterns, trends, or correlations relevant to the question
- If the data shows interesting insights beyond the question, mention them
- Be clear about what the data shows vs. what might require additional analysis
- If data is insufficient to answer fully, explain what's missing

ANSWER:"""

EXCEL_COMPARISON_PROMPT = """You are analyzing multiple data segments or time periods. 
Compare and contrast the data, highlighting:
- Key differences in metrics
- Percentage changes
- Trend shifts
- Statistical significance

Context: {context}
Question: {question}

Provide a structured comparison with specific numbers."""

EXCEL_TREND_ANALYSIS_PROMPT = """You are performing trend analysis on time-series or sequential data.

Identify:
- Direction of trends (increasing, decreasing, stable)
- Rate of change
- Inflection points
- Seasonality or cyclical patterns
- Anomalies or outliers

Context: {context}
Question: {question}

Provide insights with supporting data points."""

EXCEL_AGGREGATION_PROMPT = """You are aggregating and summarizing data across multiple dimensions.

Provide:
- Total/sum values
- Averages and medians
- Distribution summaries
- Group-by analysis
- Top and bottom performers

Context: {context}
Question: {question}

Present findings in a clear, organized manner."""

def get_excel_prompt_template(query_type: str = "general") -> str:
    """
    Return appropriate prompt template based on query type
    
    Args:
        query_type: One of 'general', 'comparison', 'trend', 'aggregation'
    """
    templates = {
        "general": EXCEL_QUERY_PROMPT_TEMPLATE,
        "comparison": EXCEL_COMPARISON_PROMPT,
        "trend": EXCEL_TREND_ANALYSIS_PROMPT,
        "aggregation": EXCEL_AGGREGATION_PROMPT
    }
    
    return templates.get(query_type, EXCEL_QUERY_PROMPT_TEMPLATE)

def detect_query_type(query: str) -> str:
    """
    Detect the type of analytical query to use appropriate prompt
    """
    query_lower = query.lower()
    
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'between', 'contrast']
    trend_keywords = ['trend', 'over time', 'change', 'growth', 'decline', 'increase', 'decrease']
    aggregation_keywords = ['total', 'sum', 'average', 'count', 'group by', 'aggregate', 'summarize']
    
    if any(kw in query_lower for kw in comparison_keywords):
        return "comparison"
    elif any(kw in query_lower for kw in trend_keywords):
        return "trend"
    elif any(kw in query_lower for kw in aggregation_keywords):
        return "aggregation"
    else:
        return "general"

# Example usage in your RAG endpoint:
"""
# In your server.py, modify the RAG endpoint:

@app.post("/chat/rag")
async def context_aware_rag_streaming(data: RagInput):
    # ... existing code ...
    
    # Detect query type for Excel files
    query_type = detect_query_type(data.query)
    prompt_template = get_excel_prompt_template(query_type)
    
    # Build prompt with analytics context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Add conversation history
    conversation_history = "\n".join(
        f"{'User' if msg['role'] == 'human' else 'Assistant'}: {msg['content']}" 
        for msg in raw_history[:-1]
    ) if len(raw_history) > 1 else ""
    
    # Format prompt
    prompt = prompt_template.format(
        context=context,
        history=conversation_history,
        question=data.query
    )
    
    # ... rest of streaming code ...
"""