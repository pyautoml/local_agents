# Author: https://github.com/pyautoml/
# License: CC BY-NC

import os
import httpx
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Union


class WebsiteData(BaseModel):
    """Represents structured data extracted from a website."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    language: Optional[str] = None
    engine: Optional[str] = None
    engines: Optional[List[str]] = None
    score: Optional[float] = None
    category: Optional[str] = None
    published_date: Optional[str] = None
    extracted_at: Optional[str] = None
    success: bool = True


class WebsiteError(BaseModel):
    """Represents an error encountered during website data extraction."""

    query: str
    error: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    success: bool = False


class WebsiteSearch(BaseModel):
    """Performs website search and result extraction using a SearxNG instance."""

    class Config:
        arbitrary_types_allowed = True

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        query: str,
        page: int,
        timeout: int,
    ):
        """
        Fetch a single page of search results from the SearxNG instance.

        :param client: httpx.AsyncClient instance for making HTTP requests.
        :param base_url: Base URL of the SearxNG search endpoint.
        :param query: Search query string.
        :param page: Page number to fetch.
        :param timeout: Timeout in seconds for the HTTP request.
        :return: A list of search result items or an error message string.
        :rtype: Union[List[dict], str]
        """
        try:
            params = {"q": query, "format": "json", "pageno": page}
            response = await client.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            page_json = response.json()
            return page_json.get("results", [])
        except Exception as e:
            return f"Page {page} failed: {e}"

    async def _async_search(
        self,
        query: str,
        max_page_number: int,
        timeout: int,
        max_results: Optional[int],
        output_for_llm: bool,
    ) -> Union[List[WebsiteData] | str, WebsiteError | str]:
        """
        Perform an asynchronous search across multiple pages and return structured results.

        :param query: Search query string.
        :param max_page_number: Number of result pages to fetch.
        :param timeout: Timeout in seconds for each HTTP request.
        :param max_results: Maximum number of results to return (after sorting by score).
        :param output_for_llm: Whether to format results as newline-separated JSON strings.
        :return: A list of WebsiteData or a WebsiteError, or formatted string if output_for_llm is True.
        :rtype: Union[List[WebsiteData], str, WebsiteError]
        :raises ValueError if SEARXNG_BASE_URL is not set.
        """
        base_url: str = os.getenv("SEARXNG_BASE_URL")
        if not base_url:
            raise ValueError("SEARXNG_BASE_URL environment variable is not set.")

        all_data: List[WebsiteData] = []

        async with httpx.AsyncClient() as client:
            tasks = [
                self._fetch_page(client, base_url, query, page, timeout)
                for page in range(1, max_page_number + 1)
            ]
            pages_results = await asyncio.gather(*tasks)

        for result in pages_results:
            if isinstance(result, str):
                return (
                    result
                    if output_for_llm
                    else WebsiteError(query=query, error=result)
                )

            for item in result:
                all_data.append(
                    WebsiteData(
                        url=item.get("url"),
                        title=item.get("title"),
                        snippet=item.get("snippet"),
                        content=item.get("content"),
                        author=item.get("author"),
                        date=f"{item.get('date')}",
                        language=item.get("language"),
                        engine=item.get("engine"),
                        engines=item.get("engines"),
                        score=item.get("score"),
                        category=item.get("category"),
                        published_date=f"{item.get('published_date')}",
                        extracted_at=datetime.now().isoformat(),
                        success=True,
                    )
                )

        if max_results:
            all_data.sort(
                key=lambda d: d.score if d.score is not None else 0, reverse=True
            )
            all_data = all_data[:max_results]

        if output_for_llm:
            return "\n".join(d.model_dump_json() for d in all_data)

        return all_data

    def search(
        self,
        query: str,
        timeout: int = 10,
        max_page_number: int = 5,
        output_for_llm: bool = False,
        max_results: Optional[int] = None
    ) -> Union[List[WebsiteData] | str, WebsiteError | str]:
        """
        Synchronously performs a search query using asyncio and returns results.

        :param query: Search query string.
        :param max_page_number: Number of result pages to retrieve (default is 5).
        :param timeout: Timeout in seconds for each request (default is 10).
        :param max_results: Maximum number of results to return (optional).
        :param output_for_llm: If True, returns newline-separated JSON strings for LLM input.
        :return: A list of WebsiteData, or WebsiteError on failure; or formatted string if output_for_llm is True.
        :rtype: Union[List[WebsiteData], str, WebsiteError]
        """
        return asyncio.run(
            self._async_search(
                query, max_page_number, timeout, max_results, output_for_llm
            )
        )
