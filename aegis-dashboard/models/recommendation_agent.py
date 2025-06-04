"""
Recommendation Agent for Disease Early Warning System
===================================================

Uses Anthropic Claude API to generate context-aware recommendations
for different stakeholders based on disease data and forecasts.
"""

import anthropic
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
import re
import os

logger = logging.getLogger(__name__)

class RecommendationAgent:
    def __init__(self, api_key: str = None):
        """Initialize the recommendation agent with API key from argument or environment variable"""
        if api_key is None:
            api_key = os.getenv("CLAUDE_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = """You are an expert public health advisor with deep knowledge of disease surveillance, 
        healthcare systems, and public policy. Your role is to analyze disease data and generate actionable 
        recommendations for different stakeholders. Be specific, data-driven, and consider both short-term 
        and long-term implications. Focus on practical, implementable advice. Always format your response as valid JSON."""

    def _prepare_data_context(self, 
                            master_data: pd.DataFrame,
                            forecasts: Dict,
                            stakeholder: str) -> str:
        """Prepare data context for the LLM"""
        
        # Get latest data points
        latest_data = master_data.groupby('Disease').last().reset_index()
        
        # Calculate trends
        trends = {}
        for disease in master_data['Disease'].unique():
            disease_data = master_data[master_data['Disease'] == disease]
            if len(disease_data) >= 2:
                current = disease_data['Cases'].iloc[-1]
                previous = disease_data['Cases'].iloc[-2]
                trend = ((current - previous) / previous) * 100
                trends[disease] = trend
        
        # Format data context
        context = f"""
        Current Disease Status (as of {latest_data['date'].iloc[0].strftime('%Y-%m-%d')}):
        """
        
        for _, row in latest_data.iterrows():
            disease = row['Disease']
            context += f"\n{disease}:"
            context += f"\n- Current Cases: {row['Cases']:,.0f}"
            context += f"\n- Week-over-Week Change: {trends.get(disease, 0):.1f}%"
            if disease in forecasts:
                context += f"\n- 30-Day Forecast: {forecasts[disease]['prediction']:,.0f} cases"
        
        return context

    def _get_stakeholder_prompt(self, stakeholder: str) -> str:
        """Get stakeholder-specific prompt"""
        prompts = {
            'federal': """As a Federal Policy Maker, focus on:
            1. National-level resource allocation
            2. Interstate coordination
            3. Federal policy recommendations
            4. Long-term strategic planning
            5. Cross-agency coordination""",
            
            'healthcare': """As a Healthcare Professional, focus on:
            1. Clinical response protocols
            2. Resource management
            3. Staff training needs
            4. Patient care strategies
            5. Healthcare system capacity""",
            
            'state': """As a State Government official, focus on:
            1. State-level resource allocation
            2. Local public health measures
            3. State policy recommendations
            4. Regional coordination
            5. State-specific interventions"""
        }
        return prompts.get(stakeholder.lower(), "")

    def _extract_json_from_response(self, text: str) -> Dict:
        """Extract and parse JSON from LLM response"""
        # --- Strip code block markers if present ---
        text = text.strip()
        if text.startswith('```'):
            text = text.lstrip('`').strip()
            # Remove 'json' or 'html' after code block if present
            if text.startswith('json') or text.startswith('html'):
                text = text.split('\n', 1)[1] if '\n' in text else text
        try:
            # First try direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Try to find JSON-like structure in the text
                json_pattern = r'\{[\s\S]*\}'
                json_match = re.search(json_pattern, text)
                if json_match:
                    json_str = json_match.group()
                    # Clean up potential formatting issues
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
                    json_str = re.sub(r':\s*([^",\{\}\[\]\s][^,\{\}\[\]]*?)([,\}\]])', r':"\1"\2', json_str)  # Add quotes to unquoted string values
                    return json.loads(json_str)
            except Exception as e:
                logger.error(f"Failed to extract JSON from response: {str(e)}")
                logger.debug(f"Raw response text: {text}")
        
        # If all parsing attempts fail, return a structured error response
        return {
            "error": "Failed to parse recommendations",
            "raw_response": text,
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "risk_factors": [],
            "data_gaps": []
        }

    def generate_recommendations(self,
                               master_data: pd.DataFrame,
                               forecasts: Dict,
                               stakeholder: str,
                               additional_context: Optional[Dict] = None) -> Dict:
        """Generate recommendations for a specific stakeholder"""
        
        try:
            # Prepare data context
            data_context = self._prepare_data_context(master_data, forecasts, stakeholder)
            stakeholder_prompt = self._get_stakeholder_prompt(stakeholder)
            
            # Prepare the full prompt
            prompt = f"""
            {self.system_prompt}
            
            {stakeholder_prompt}
            
            Current Data Analysis:
            {data_context}
            
            Additional Context:
            {json.dumps(additional_context) if additional_context else 'None'}
            
            Please provide recommendations in the following JSON format:
            {{
                "short_term": [
                    {{
                        "recommendation": "string",
                        "rationale": "string",
                        "priority": "high|medium|low",
                        "success_metrics": ["string"]
                    }}
                ],
                "medium_term": [...],
                "long_term": [...],
                "risk_factors": ["string"],
                "data_gaps": ["string"]
            }}
            
            Ensure the response is valid JSON. Do not include any text outside the JSON structure.
            """
            
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.7,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse JSON from response
            recommendations = self._extract_json_from_response(response.content[0].text)
            
            # Validate the structure
            required_keys = ["short_term", "medium_term", "long_term", "risk_factors", "data_gaps"]
            if not all(key in recommendations for key in required_keys):
                logger.error("Response missing required keys")
                return {
                    "error": "Invalid response structure",
                    "raw_response": response.content[0].text,
                    "short_term": [],
                    "medium_term": [],
                    "long_term": [],
                    "risk_factors": [],
                    "data_gaps": []
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "error": str(e),
                "stakeholder": stakeholder,
                "short_term": [],
                "medium_term": [],
                "long_term": [],
                "risk_factors": [],
                "data_gaps": []
            }

    def format_recommendations_for_display(self, recommendations: Dict) -> str:
        """Format recommendations for display in the dashboard"""
        if "error" in recommendations:
            return f"Error generating recommendations: {recommendations['error']}"
        
        html = "<div class='recommendations-container'>"
        
        # Short-term recommendations
        html += "<h3>Short-term Recommendations (Next 7 Days)</h3>"
        html += "<div class='recommendation-section'>"
        for rec in recommendations.get("short_term", []):
            html += f"""
            <div class='recommendation-card priority-{rec["priority"]}'>
                <h4>{rec["recommendation"]}</h4>
                <p><strong>Rationale:</strong> {rec["rationale"]}</p>
                <p><strong>Success Metrics:</strong></p>
                <ul>
                    {''.join(f'<li>{metric}</li>' for metric in rec["success_metrics"])}
                </ul>
            </div>
            """
        html += "</div>"
        
        # Medium-term recommendations
        html += "<h3>Medium-term Recommendations (Next 30 Days)</h3>"
        html += "<div class='recommendation-section'>"
        for rec in recommendations.get("medium_term", []):
            html += f"""
            <div class='recommendation-card priority-{rec["priority"]}'>
                <h4>{rec["recommendation"]}</h4>
                <p><strong>Rationale:</strong> {rec["rationale"]}</p>
                <p><strong>Success Metrics:</strong></p>
                <ul>
                    {''.join(f'<li>{metric}</li>' for metric in rec["success_metrics"])}
                </ul>
            </div>
            """
        html += "</div>"
        
        # Long-term recommendations
        html += "<h3>Long-term Recommendations (Next 90 Days)</h3>"
        html += "<div class='recommendation-section'>"
        for rec in recommendations.get("long_term", []):
            html += f"""
            <div class='recommendation-card priority-{rec["priority"]}'>
                <h4>{rec["recommendation"]}</h4>
                <p><strong>Rationale:</strong> {rec["rationale"]}</p>
                <p><strong>Success Metrics:</strong></p>
                <ul>
                    {''.join(f'<li>{metric}</li>' for metric in rec["success_metrics"])}
                </ul>
            </div>
            """
        html += "</div>"
        
        # Risk factors
        if recommendations.get("risk_factors"):
            html += "<h3>Key Risk Factors to Monitor</h3>"
            html += "<ul class='risk-factors'>"
            for risk in recommendations["risk_factors"]:
                html += f"<li>{risk}</li>"
            html += "</ul>"
        
        # Data gaps
        if recommendations.get("data_gaps"):
            html += "<h3>Identified Data Gaps</h3>"
            html += "<ul class='data-gaps'>"
            for gap in recommendations["data_gaps"]:
                html += f"<li>{gap}</li>"
            html += "</ul>"
        
        html += "</div>"
        # --- Log the HTML output for debugging ---
        logger.info("Formatted recommendations HTML output:\n%s", html)
        return html 