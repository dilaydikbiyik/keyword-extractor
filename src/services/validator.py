"""
LLM Validator Service (Optional)

Validates extracted keywords using OpenAI API for quality assurance.
This is an optional component for high-confidence extraction.
"""

from typing import List, Dict, Tuple, Optional
import json
import time


class LLMValidator:
    """
    Validates keywords using LLM (e.g., GPT-3.5).

    Useful for:
    - Validating low-confidence predictions
    - Domain-specific keyword verification
    - Final quality assurance

    Note: Requires OpenAI API key. Can be expensive for large batches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout_seconds: int = 10
    ):
        """
        Initialize the LLM validator.

        Args:
            api_key: OpenAI API key (from environment if not provided)
            model: Model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            max_retries: Maximum retries for API calls
            timeout_seconds: Timeout for API calls
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Import OpenAI lazily
        try:
            from openai import OpenAI

            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI()

            self.available = True
        except ImportError:
            print("Warning: OpenAI library not installed. Validator disabled.")
            self.available = False
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
            self.available = False

    def validate_keywords(
        self,
        text: str,
        keywords: List[Tuple[str, float]],
        sector: Optional[str] = None,
        confidence_range: Tuple[float, float] = (0.3, 0.6)
    ) -> List[Dict]:
        """
        Validate keywords using LLM for keywords in confidence range.

        Args:
            text: Original text
            keywords: List of (keyword, score) tuples
            sector: Optional sector context
            confidence_range: Only validate keywords in this confidence range

        Returns:
            List of validation results
        """
        if not self.available:
            return self._format_validation_results(keywords, validated=False)

        # Filter keywords by confidence range
        to_validate = [
            (kw, score) for kw, score in keywords
            if confidence_range[0] <= score <= confidence_range[1]
        ]

        if not to_validate:
            # All keywords are either high or low confidence, no validation needed
            return self._format_validation_results(keywords, validated=True, approved_all=True)

        # Prepare validation prompt
        prompt = self._build_validation_prompt(text, to_validate, sector)

        # Get LLM validation
        try:
            response = self._call_llm(prompt)
            validation_result = self._parse_validation_response(response, keywords)
            return validation_result
        except Exception as e:
            print(f"LLM validation failed: {e}. Returning original scores.")
            return self._format_validation_results(keywords, validated=False)

    def _build_validation_prompt(
        self,
        text: str,
        keywords: List[Tuple[str, float]],
        sector: Optional[str] = None
    ) -> str:
        """Build validation prompt for LLM."""
        sector_context = f"Sector: {sector}\n" if sector else ""

        keywords_str = ", ".join([f'"{kw}"' for kw, _ in keywords])

        prompt = f"""Validate the following extracted keywords for a business description.

{sector_context}
Business Description: {text[:500]}...

Keywords to validate: {keywords_str}

For each keyword, respond with:
1. Whether it's relevant to the business description (yes/no)
2. A confidence score (0-1)
3. Brief reason

Format your response as JSON:
{{"validated_keywords": [{{"keyword": "...", "relevant": true/false, "confidence": 0.0-1.0, "reason": "..."}}]}}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at validating business keywords. Always respond in JSON format."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=self.timeout_seconds
                )
                return response.choices[0].message.content
            except Exception:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def _parse_validation_response(
        self,
        response: str,
        original_keywords: List[Tuple[str, float]]
    ) -> List[Dict]:
        """Parse LLM validation response."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return self._format_validation_results(original_keywords, validated=False)

            json_str = response[json_start:json_end]
            validation_data = json.loads(json_str)

            validated_kws = validation_data.get('validated_keywords', [])

            # Merge with original scores
            results = []
            for kw, orig_score in original_keywords:
                # Find matching validation
                matching = [
                    v for v in validated_kws
                    if v.get('keyword', '').lower() == kw.lower()
                ]

                if matching:
                    val = matching[0]
                    final_score = orig_score * val.get('confidence', 0.5)
                    results.append({
                        'keyword': kw,
                        'original_score': orig_score,
                        'llm_confidence': val.get('confidence', 0.5),
                        'final_score': final_score,
                        'relevant': val.get('relevant', True),
                        'reason': val.get('reason', ''),
                        'validated': True
                    })
                else:
                    results.append({
                        'keyword': kw,
                        'original_score': orig_score,
                        'final_score': orig_score,
                        'validated': False
                    })

            return results

        except Exception:
            print("Failed to parse validation response. Returning original scores.")
            return self._format_validation_results(original_keywords, validated=False)

    def _format_validation_results(
        self,
        keywords: List[Tuple[str, float]],
        validated: bool = False,
        approved_all: bool = False
    ) -> List[Dict]:
        """Format validation results."""
        results = []

        for kw, score in keywords:
            results.append({
                'keyword': kw,
                'original_score': score,
                'final_score': score,
                'validated': validated,
                'approved_all': approved_all
            })

        return results

    def batch_validate(
        self,
        items: List[Dict],
        confidence_range: Tuple[float, float] = (0.3, 0.6)
    ) -> List[Dict]:
        """
        Validate keywords for multiple items.

        Args:
            items: List of items with 'text' and 'keywords' fields
            confidence_range: Confidence range for validation

        Returns:
            List of items with validated keywords
        """
        results = []

        for i, item in enumerate(items):
            print(f"Validating item {i + 1}/{len(items)}...")

            validated_keywords = self.validate_keywords(
                item.get('text', ''),
                item.get('keywords', []),
                item.get('sector'),
                confidence_range
            )

            item['validated_keywords'] = validated_keywords
            results.append(item)

            # Rate limiting
            if i < len(items) - 1:
                time.sleep(1)

        return results

    def is_available(self) -> bool:
        """Check if validator is available."""
        return self.available