"""NLP service for text analysis on dataset columns."""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Dataset, AnalysisResult
from services.data_service import get_data_service


class NLPService:
    """Service for NLP analysis on text columns."""

    def __init__(self):
        self.data_service = get_data_service()

    async def analyze_sentiment(
        self,
        dataset_id: uuid.UUID,
        column: str,
        db: AsyncSession
    ) -> Optional[AnalysisResult]:
        """
        Analyze sentiment of text in a column.

        Uses TextBlob for lexicon-based sentiment analysis.
        Returns polarity (-1 to 1) and subjectivity (0 to 1) for each row.

        :param dataset_id: Dataset UUID
        :param column: Name of the text column
        :param db: Database session
        :return: AnalysisResult with sentiment scores
        """
        df = await self.data_service.get_dataframe(dataset_id, db)
        if df is None:
            return None

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")

        # Get text values, handle missing
        texts = df[column].fillna("").astype(str).tolist()

        # Analyze sentiment for each text
        sentiments = []
        for text in texts:
            if text.strip():
                blob = TextBlob(text)
                sentiments.append({
                    "polarity": round(blob.sentiment.polarity, 4),
                    "subjectivity": round(blob.sentiment.subjectivity, 4),
                    "label": self._polarity_to_label(blob.sentiment.polarity)
                })
            else:
                sentiments.append({
                    "polarity": None,
                    "subjectivity": None,
                    "label": "neutral"
                })

        # Compute aggregate stats
        valid_polarities = [s["polarity"] for s in sentiments if s["polarity"] is not None]

        if valid_polarities:
            avg_polarity = round(np.mean(valid_polarities), 4)
            avg_subjectivity = round(np.mean([s["subjectivity"] for s in sentiments if s["subjectivity"] is not None]), 4)
        else:
            avg_polarity = 0
            avg_subjectivity = 0

        # Count labels
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in sentiments:
            label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

        results = {
            "column": column,
            "total_rows": len(texts),
            "analyzed_rows": len(valid_polarities),
            "aggregate": {
                "average_polarity": avg_polarity,
                "average_subjectivity": avg_subjectivity,
                "label_distribution": label_counts,
                "overall_sentiment": self._polarity_to_label(avg_polarity)
            },
            "row_results": sentiments,
            "insights": self._generate_sentiment_insights(avg_polarity, avg_subjectivity, label_counts)
        }

        # Store result
        analysis_result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type="sentiment",
            parameters={"column": column},
            results=results,
            status="completed",
            completed_at=datetime.utcnow()
        )

        db.add(analysis_result)
        await db.flush()

        return analysis_result

    def _polarity_to_label(self, polarity: float) -> str:
        """Convert polarity score to label."""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def _generate_sentiment_insights(
        self,
        avg_polarity: float,
        avg_subjectivity: float,
        label_counts: Dict[str, int]
    ) -> List[str]:
        """Generate human-readable insights from sentiment analysis."""
        insights = []
        total = sum(label_counts.values())

        if total == 0:
            return ["No text data available for analysis."]

        # Overall sentiment insight
        if avg_polarity > 0.3:
            insights.append("Overall sentiment is strongly positive.")
        elif avg_polarity > 0.1:
            insights.append("Overall sentiment is moderately positive.")
        elif avg_polarity < -0.3:
            insights.append("Overall sentiment is strongly negative.")
        elif avg_polarity < -0.1:
            insights.append("Overall sentiment is moderately negative.")
        else:
            insights.append("Overall sentiment is neutral.")

        # Distribution insight
        pos_pct = round(label_counts["positive"] / total * 100, 1)
        neg_pct = round(label_counts["negative"] / total * 100, 1)
        neu_pct = round(label_counts["neutral"] / total * 100, 1)

        insights.append(f"Sentiment breakdown: {pos_pct}% positive, {neg_pct}% negative, {neu_pct}% neutral.")

        # Subjectivity insight
        if avg_subjectivity > 0.6:
            insights.append("Text is highly subjective/opinionated.")
        elif avg_subjectivity < 0.4:
            insights.append("Text is relatively objective/factual.")

        return insights

    async def extract_keywords(
        self,
        dataset_id: uuid.UUID,
        column: str,
        db: AsyncSession,
        top_n: int = 20,
        min_df: int = 2,
        max_df: float = 0.8
    ) -> Optional[AnalysisResult]:
        """
        Extract keywords from a text column using TF-IDF.

        :param dataset_id: Dataset UUID
        :param column: Name of the text column
        :param db: Database session
        :param top_n: Number of top keywords to return
        :param min_df: Minimum document frequency (ignore rare terms)
        :param max_df: Maximum document frequency (ignore common terms)
        :return: AnalysisResult with keywords and scores
        """
        df = await self.data_service.get_dataframe(dataset_id, db)
        if df is None:
            return None

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")

        # Get text values, filter empty
        texts = df[column].fillna("").astype(str).tolist()
        valid_texts = [t for t in texts if t.strip()]

        if len(valid_texts) < 2:
            raise ValueError("Need at least 2 non-empty text rows for keyword extraction")

        # Adjust min_df if we have few documents
        actual_min_df = min(min_df, max(1, len(valid_texts) // 3))

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            min_df=actual_min_df,
            max_df=max_df,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
        except ValueError as e:
            raise ValueError(f"Could not extract keywords: {str(e)}")

        # Get feature names and average TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Sort by score and get top N
        top_indices = avg_scores.argsort()[::-1][:top_n]
        keywords = [
            {
                "keyword": feature_names[i],
                "score": round(float(avg_scores[i]), 4)
            }
            for i in top_indices
            if avg_scores[i] > 0
        ]

        # Per-document top keywords (for each row)
        row_keywords = []
        dense_matrix = tfidf_matrix.toarray()
        for i, row in enumerate(dense_matrix):
            top_k_idx = row.argsort()[::-1][:5]
            row_keywords.append([
                {"keyword": feature_names[idx], "score": round(float(row[idx]), 4)}
                for idx in top_k_idx if row[idx] > 0
            ])

        results = {
            "column": column,
            "total_rows": len(texts),
            "analyzed_rows": len(valid_texts),
            "parameters": {
                "top_n": top_n,
                "min_df": actual_min_df,
                "max_df": max_df
            },
            "top_keywords": keywords,
            "row_keywords": row_keywords,
            "insights": self._generate_keyword_insights(keywords)
        }

        # Store result
        analysis_result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type="keywords",
            parameters={"column": column, "top_n": top_n},
            results=results,
            status="completed",
            completed_at=datetime.utcnow()
        )

        db.add(analysis_result)
        await db.flush()

        return analysis_result

    def _generate_keyword_insights(self, keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from keyword extraction."""
        insights = []

        if not keywords:
            return ["No significant keywords found in the text."]

        # Top keywords
        top_5 = [kw["keyword"] for kw in keywords[:5]]
        insights.append(f"Top keywords: {', '.join(top_5)}")

        # Check for bigrams
        bigrams = [kw["keyword"] for kw in keywords if " " in kw["keyword"]]
        if bigrams:
            insights.append(f"Key phrases: {', '.join(bigrams[:3])}")

        return insights


# Singleton instance
_nlp_service: Optional[NLPService] = None


def get_nlp_service() -> NLPService:
    """Get the singleton NLPService instance."""
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service
