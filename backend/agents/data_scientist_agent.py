"""Data Scientist Agent with tools for data analysis, ML, and visualization."""
import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base_agent import BaseAgent, AgentResponse


class AnalysisPlan(BaseModel):
    """Structured output for analysis planning."""
    understanding: str = Field(description="Brief understanding of what the user wants")
    steps: List[str] = Field(description="List of analysis steps to perform")
    tools_needed: List[str] = Field(description="Tools to use: profile, sentiment, keywords, classify, cluster, visualize")
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID if mentioned or available in context")
    columns: List[str] = Field(default_factory=list, description="Specific columns mentioned by user")
    visualization_type: Optional[str] = Field(default=None, description="Type of visualization if requested")


DATA_SCIENTIST_SYSTEM_PROMPT = """You are a Data Scientist Agent that helps users analyze datasets.

You have access to these tools:
1. **profile** - Get dataset statistics, column types, missing values, correlations
2. **sentiment** - Analyze sentiment in text columns (returns polarity and subjectivity)
3. **keywords** - Extract top keywords from text columns using TF-IDF
4. **classify** - Train classification models (Random Forest, Logistic Regression, SVM)
5. **cluster** - Perform clustering (K-Means, DBSCAN, Hierarchical)
6. **visualize** - Generate charts (histogram, scatter, correlation, timeseries, pairplot)

Based on the user's query, determine:
1. What they want to accomplish
2. Which tools to use and in what order
3. Which columns to analyze (if specified)
4. What visualizations to generate

Available datasets context:
{datasets_context}

{format_instructions}"""


class DataScientistAgent(BaseAgent):
    """
    Data Scientist Agent that performs data analysis using various tools.

    Capabilities:
    - Dataset profiling and statistics
    - NLP analysis (sentiment, keywords)
    - Machine learning (classification, clustering)
    - Data visualization
    """

    def __init__(self):
        super().__init__(
            name="Data Scientist",
            description="Analyzes datasets using profiling, NLP, machine learning, and visualization tools.",
            model="gpt-4o-mini",
            temperature=0.3
        )
        self.parser = PydanticOutputParser(pydantic_object=AnalysisPlan)

    async def _get_datasets_context(self, db: AsyncSession) -> str:
        """Get available datasets for context."""
        from services.data_service import get_data_service

        data_service = get_data_service()
        datasets = await data_service.get_datasets(db)

        if not datasets:
            return "No datasets uploaded yet."

        context_parts = []
        for ds in datasets[:5]:  # Limit to 5 most recent
            cols = ds.schema.get("column_names", []) if ds.schema else []
            context_parts.append(
                f"- **{ds.filename}** (ID: {ds.id}): {ds.row_count} rows, "
                f"columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}"
            )

        return "\n".join(context_parts)

    async def _plan_analysis(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisPlan:
        """Create an analysis plan based on the user's query."""
        datasets_context = await self._get_datasets_context(db)

        # Add dataset_id from context if available
        extra_context = ""
        if context and context.get("dataset_id"):
            extra_context = f"\n\nCurrent dataset ID: {context['dataset_id']}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", DATA_SCIENTIST_SYSTEM_PROMPT),
            ("user", "{query}{extra_context}")
        ]).partial(
            format_instructions=self.parser.get_format_instructions(),
            datasets_context=datasets_context
        )

        chain = prompt | self.llm | self.parser

        plan = await chain.ainvoke({
            "query": query,
            "extra_context": extra_context
        })

        # If no dataset_id in plan but one in context, use that
        if not plan.dataset_id and context and context.get("dataset_id"):
            plan.dataset_id = context["dataset_id"]

        return plan

    async def _execute_profile(
        self,
        dataset_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute dataset profiling."""
        from services.data_profiling_service import get_profiling_service

        profiling_service = get_profiling_service()
        profile = await profiling_service.get_profile(uuid.UUID(dataset_id), db)

        if not profile:
            return {"error": "Could not generate profile"}

        # Return summarized profile
        return {
            "basic_stats": profile.get("basic_stats", {}),
            "column_count": len(profile.get("column_profiles", [])),
            "data_quality": profile.get("data_quality", {}),
            "top_correlations": self._get_top_correlations(profile.get("correlations", {}))
        }

    def _get_top_correlations(self, correlations: Dict, top_n: int = 5) -> List[Dict]:
        """Extract top correlations from correlation matrix."""
        if not correlations:
            return []

        pairs = []
        seen = set()

        for col1, corrs in correlations.items():
            if not isinstance(corrs, dict):
                continue
            for col2, value in corrs.items():
                # Skip non-numeric values (could be nested dicts or other metadata)
                if not isinstance(value, (int, float)):
                    continue
                if col1 != col2 and (col2, col1) not in seen:
                    seen.add((col1, col2))
                    pairs.append({
                        "columns": [col1, col2],
                        "correlation": round(float(value), 3)
                    })

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs[:top_n]

    async def _execute_sentiment(
        self,
        dataset_id: str,
        column: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute sentiment analysis."""
        from services.nlp_service import get_nlp_service

        nlp_service = get_nlp_service()
        result = await nlp_service.analyze_sentiment(uuid.UUID(dataset_id), column, db)

        if not result:
            return {"error": f"Could not analyze sentiment for column '{column}'"}

        return {
            "column": column,
            "total_analyzed": result.results.get("analyzed_rows", 0),
            "aggregate": result.results.get("aggregate", {}),
            "insights": result.results.get("insights", [])
        }

    async def _execute_keywords(
        self,
        dataset_id: str,
        column: str,
        db: AsyncSession,
        top_n: int = 15
    ) -> Dict[str, Any]:
        """Execute keyword extraction."""
        from services.nlp_service import get_nlp_service

        nlp_service = get_nlp_service()
        result = await nlp_service.extract_keywords(
            uuid.UUID(dataset_id), column, db, top_n=top_n
        )

        if not result:
            return {"error": f"Could not extract keywords from column '{column}'"}

        return {
            "column": column,
            "top_keywords": result.results.get("top_keywords", [])[:10],
            "insights": result.results.get("insights", [])
        }

    async def _execute_classify(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: List[str],
        db: AsyncSession,
        algorithm: str = "random_forest"
    ) -> Dict[str, Any]:
        """Execute classification training."""
        from services.ml_service import get_ml_service

        ml_service = get_ml_service()

        try:
            model = await ml_service.train_classifier(
                dataset_id=uuid.UUID(dataset_id),
                target_column=target_column,
                feature_columns=feature_columns,
                algorithm=algorithm,
                db=db
            )

            return {
                "model_id": str(model.id),
                "algorithm": model.algorithm,
                "metrics": model.metrics,
                "feature_importance": model.feature_importance[:5] if model.feature_importance else [],
                "status": model.status
            }
        except Exception as e:
            return {"error": str(e)}

    async def _execute_cluster(
        self,
        dataset_id: str,
        feature_columns: List[str],
        db: AsyncSession,
        algorithm: str = "kmeans",
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute clustering analysis."""
        from services.ml_service import get_ml_service

        ml_service = get_ml_service()

        try:
            result = await ml_service.cluster_data(
                dataset_id=uuid.UUID(dataset_id),
                feature_columns=feature_columns,
                algorithm=algorithm,
                n_clusters=n_clusters,
                db=db
            )

            return {
                "algorithm": result.get("algorithm"),
                "n_clusters": result.get("n_clusters"),
                "metrics": result.get("metrics"),
                "cluster_profiles": result.get("cluster_profiles", [])[:3]  # Top 3 clusters
            }
        except Exception as e:
            return {"error": str(e)}

    async def _execute_visualize(
        self,
        dataset_id: str,
        viz_type: str,
        db: AsyncSession,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute visualization generation."""
        from services.visualization_service import get_visualization_service

        viz_service = get_visualization_service()

        try:
            df = await viz_service._load_dataset(uuid.UUID(dataset_id), db)

            if viz_type == "auto":
                charts = await viz_service.auto_generate_visualizations(
                    uuid.UUID(dataset_id), db, max_charts=4
                )
                return {
                    "type": "auto",
                    "charts_generated": len(charts),
                    "chart_types": [c.get("type") for c in charts]
                }

            elif viz_type == "distribution":
                column = kwargs.get("column")
                if not column:
                    return {"error": "Column required for distribution plot"}
                figure = viz_service.generate_distribution_plot(df, column)
                return {"type": "distribution", "column": column, "figure": figure}

            elif viz_type == "scatter":
                x_col = kwargs.get("x_column")
                y_col = kwargs.get("y_column")
                if not x_col or not y_col:
                    return {"error": "x_column and y_column required for scatter plot"}
                figure = viz_service.generate_scatter_plot(
                    df, x_col, y_col, kwargs.get("color_by")
                )
                return {"type": "scatter", "figure": figure}

            elif viz_type == "correlation":
                figure = viz_service.generate_correlation_matrix(df, kwargs.get("columns"))
                return {"type": "correlation", "figure": figure}

            elif viz_type == "timeseries":
                date_col = kwargs.get("date_column")
                value_col = kwargs.get("value_column")
                if not date_col or not value_col:
                    return {"error": "date_column and value_column required"}
                figure = viz_service.generate_time_series_plot(df, date_col, value_col)
                return {"type": "timeseries", "figure": figure}

            else:
                return {"error": f"Unknown visualization type: {viz_type}"}

        except Exception as e:
            return {"error": str(e)}

    async def _generate_insights(
        self,
        query: str,
        results: Dict[str, Any],
        plan: AnalysisPlan
    ) -> str:
        """Generate natural language insights from analysis results."""
        # Build context from results
        results_summary = json.dumps(results, indent=2, default=str)

        insight_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data scientist explaining analysis results to a user.

Based on the analysis results, provide clear, actionable insights in natural language.

Guidelines:
- Be concise but informative
- Highlight key findings
- Mention any patterns or anomalies
- Suggest next steps if appropriate
- Use numbers and percentages where relevant
- Format nicely with bullet points if multiple insights"""),
            ("user", """Original question: {query}

Analysis plan: {plan}

Results:
{results}

Provide insights based on these results:""")
        ])

        chain = insight_prompt | self.llm

        response = await chain.ainvoke({
            "query": query,
            "plan": f"Steps: {', '.join(plan.steps)}",
            "results": results_summary
        })

        return response.content

    async def invoke(
        self,
        query: str,
        db: AsyncSession,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a data analysis query.

        1. Plan the analysis based on user query
        2. Execute relevant tools
        3. Generate insights from results
        """
        try:
            # Step 1: Plan the analysis
            plan = await self._plan_analysis(query, db, context)

            # Check if we have a dataset to work with
            if not plan.dataset_id:
                return AgentResponse(
                    content="I need a dataset to analyze. Please upload a dataset first using the /datasets/upload endpoint, or specify which dataset you'd like me to analyze.",
                    metadata={"agent": self.name, "plan": plan.model_dump()}
                )

            # Step 2: Execute tools based on plan
            results = {
                "dataset_id": plan.dataset_id,
                "analyses": {}
            }

            visualizations = []

            for tool in plan.tools_needed:
                if tool == "profile":
                    results["analyses"]["profile"] = await self._execute_profile(
                        plan.dataset_id, db
                    )

                elif tool == "sentiment":
                    if plan.columns:
                        for col in plan.columns:
                            results["analyses"][f"sentiment_{col}"] = await self._execute_sentiment(
                                plan.dataset_id, col, db
                            )
                    else:
                        results["analyses"]["sentiment"] = {
                            "error": "Please specify which text column to analyze for sentiment"
                        }

                elif tool == "keywords":
                    if plan.columns:
                        for col in plan.columns:
                            results["analyses"][f"keywords_{col}"] = await self._execute_keywords(
                                plan.dataset_id, col, db
                            )
                    else:
                        results["analyses"]["keywords"] = {
                            "error": "Please specify which text column to extract keywords from"
                        }

                elif tool == "cluster":
                    if plan.columns:
                        results["analyses"]["clustering"] = await self._execute_cluster(
                            plan.dataset_id, plan.columns, db
                        )
                    else:
                        results["analyses"]["clustering"] = {
                            "note": "No columns specified, will use all numeric columns"
                        }
                        # Get profile to find numeric columns
                        profile = await self._execute_profile(plan.dataset_id, db)
                        results["analyses"]["clustering"] = await self._execute_cluster(
                            plan.dataset_id, [], db  # Empty list = use all numeric
                        )

                elif tool == "visualize":
                    viz_type = plan.visualization_type or "auto"
                    viz_result = await self._execute_visualize(
                        plan.dataset_id, viz_type, db,
                        column=plan.columns[0] if plan.columns else None,
                        columns=plan.columns if plan.columns else None
                    )
                    results["analyses"]["visualization"] = viz_result
                    if "figure" in viz_result:
                        visualizations.append(viz_result)

            # Step 3: Generate insights
            insights = await self._generate_insights(query, results, plan)

            # Build response metadata
            metadata = {
                "agent": self.name,
                "plan": plan.model_dump(),
                "results": results,
                "tools_executed": plan.tools_needed
            }

            if visualizations:
                metadata["visualizations"] = visualizations

            return AgentResponse(
                content=insights,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            import traceback
            print(f"DataScientistAgent error: {e}")
            traceback.print_exc()
            return await self.handle_error(e)


# Singleton instance
_data_scientist_agent: Optional[DataScientistAgent] = None


def get_data_scientist_agent() -> DataScientistAgent:
    """Get the data scientist agent singleton."""
    global _data_scientist_agent
    if _data_scientist_agent is None:
        _data_scientist_agent = DataScientistAgent()
    return _data_scientist_agent
