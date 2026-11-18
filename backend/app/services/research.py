from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

import httpx


EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class ResearchProviderError(RuntimeError):
    """Raised when the external scholarly search provider is unavailable."""


def _quoted_term(value: str) -> str:
    normalized = " ".join(
        "".join(character if character.isalnum() or character in " -" else " " for character in value).split()
    )
    if not normalized:
        raise ValueError("Search terms must contain letters or numbers.")
    return f'"{normalized[:100]}"'


def _article_url(article: dict[str, Any]) -> str | None:
    doi = article.get("doi")
    if isinstance(doi, str) and doi.strip():
        return f"https://doi.org/{doi.strip()}"
    pmid = article.get("pmid")
    if isinstance(pmid, str) and pmid.strip():
        return f"https://europepmc.org/article/MED/{pmid.strip()}"
    return None


async def search_publications(crop: str, disease: str, max_results: int = 5) -> dict[str, Any]:
    crop_term = _quoted_term(crop)
    disease_term = _quoted_term(disease)
    query = f"SRC:MED AND TITLE_ABS:({crop_term} AND {disease_term})"
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": max_results}

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(EUROPE_PMC_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ResearchProviderError("Europe PMC could not complete the scholarly search.") from exc

    result_list = payload.get("resultList", {}) if isinstance(payload, dict) else {}
    raw_articles = result_list.get("result", []) if isinstance(result_list, dict) else []
    articles = []
    seen = set()
    for article in raw_articles:
        if not isinstance(article, dict):
            continue
        title = article.get("title")
        url = _article_url(article)
        if not isinstance(title, str) or not title.strip() or not url:
            continue
        identifier = article.get("doi") or article.get("pmid") or title
        normalized_identifier = re.sub(r"\s+", " ", str(identifier)).lower()
        if normalized_identifier in seen:
            continue
        seen.add(normalized_identifier)
        abstract = article.get("abstractText")
        articles.append(
            {
                "title": title.strip(),
                "authors": str(article.get("authorString") or "Not provided"),
                "journal": str(article.get("journalTitle") or "Not provided"),
                "year": str(article.get("pubYear") or "Not provided"),
                "doi": str(article.get("doi") or "") or None,
                "url": url,
                "abstract": abstract.strip()[:5000] if isinstance(abstract, str) and abstract.strip() else None,
            }
        )

    return {
        "provider": "Europe PMC / PubMed-indexed records",
        "query": f"{crop.strip()} / {disease.strip()}",
        "status": "results" if articles else "no_results",
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "results": articles[:max_results],
        "message": (
            "Retrieved publication metadata and abstracts where available. These records are evidence to assess, "
            "not confirmation that the image shows this disease."
            if articles
            else "No matching PubMed-indexed records were found for these terms."
        ),
    }