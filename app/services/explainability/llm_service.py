"""
llm_service.py
LLM-powered explanation generator using Google Gemini 1.5 Flash
Ultra-fast, low-latency explanations for real-time AI governance
Alternative to SAP Joule's natural language processing
"""

import logging
import json
import requests
from typing import Dict, List
from app.services.explainability.shap_service import FeatureImportance

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyAwvh-467cM86qWIMuzHUitTcot7w49AdM"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for generating natural language explanations using Google Gemini 1.5 Flash.
    
    This implements the 'Explainability Layer' from the Ethical AI Auditor proposal,
    providing dynamic, role-based model insights as a true alternative to SAP Joule.
    
    Key Features:
    - Uses gemini-1.5-flash for low-latency responses
    - Role-specific explanations (executive, compliance, technical)
    - Context-aware bias detection
    - SAP BTP-compatible microservice design
    """
    
    @staticmethod
    def _call_gemini_api(prompt: str) -> str:
        """
        Call Gemini 1.5 Flash API using requests.
        
        Args:
            prompt: Input text for the model
            
        Returns:
            Generated text response
        """
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        try:
            logger.info("Calling Gemini 1.5 Flash API...")
            response = requests.post(GEMINI_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result = response.json()
            generated_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
            
            if not generated_text:
                raise ValueError("Empty response from Gemini API")
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to call Gemini API: {str(e)}")
            raise

    @staticmethod
    def generate_explanation(
        shap_values: Dict[str, float],
        feature_importance: List[FeatureImportance],
        target_variable: str,
        sensitive_attribute: str,
        role: str
    ) -> str:
        """
        Generate a natural language explanation using Gemini 1.5 Flash.
        """
        try:
            # Prepare context
            top_features = [f.feature for f in feature_importance[:5]]
            top_importances = [round(float(f.importance), 3) for f in feature_importance[:5]]
            directions = [f.direction for f in feature_importance[:5]]
            
            context = {
                "target_variable": target_variable,
                "sensitive_attribute": sensitive_attribute,
                "top_features": top_features,
                "top_importances": top_importances,
                "directions": directions,
                "shap_values": {k: round(float(v), 3) for k, v in shap_values.items()}
            }

            # Define role-specific prompts
            prompts = {
                "executive": f"""
                You are an AI ethics advisor to a CTO. Explain in 3-4 sentences how this model makes decisions.
                Focus on business risk, brand reputation, and regulatory exposure.
                
                IMPORTANT BIAS ASSESSMENT RULES:
                - If '{sensitive_attribute}' has impact > 0.15: Flag as HIGH compliance risk
                - If '{sensitive_attribute}' has impact 0.05-0.15: Flag as MODERATE compliance risk  
                - If '{sensitive_attribute}' has impact < 0.05: Note as LOW/MINIMAL bias risk
                - Consider both the ranking AND the actual magnitude of impact
                
                Use executive tone: concise, strategic, board-ready.
                Context: {json.dumps(context, indent=2)}
                """,
                
                "compliance": f"""
                You are an AI compliance officer. Analyze whether this model complies with GDPR, EU AI Act, and EEOC.
                
                BIAS RISK ASSESSMENT:
                - '{sensitive_attribute}' impact > 0.15: HIGH RISK - Immediate action required
                - '{sensitive_attribute}' impact 0.05-0.15: MODERATE RISK - Monitor and investigate
                - '{sensitive_attribute}' impact < 0.05: LOW RISK - Acceptable under most regulations
                
                Recommend specific actions based on the actual magnitude, not just ranking.
                Use formal, audit-ready language.
                Context: {json.dumps(context, indent=2)}
                """,
                
                "technical": f"""
                You are a senior ML engineer. Explain how each of the top 3 features influences the prediction.
                Describe the direction and magnitude of impact.
                
                TECHNICAL BIAS THRESHOLDS:
                - '{sensitive_attribute}' SHAP impact > 0.15: Significant bias requiring model retraining
                - '{sensitive_attribute}' SHAP impact 0.05-0.15: Moderate bias, consider fairness constraints
                - '{sensitive_attribute}' SHAP impact < 0.05: Minimal bias, acceptable for most use cases
                
                Focus on actual numerical impact, not just feature ranking.
                Suggest model improvements based on magnitude of bias.
                Use technical but clear language.
                Context: {json.dumps(context, indent=2)}
                """
            }
            
            # Select prompt
            selected_prompt = prompts.get(role, prompts["executive"])
            
            # Call Gemini 1.5 Flash
            return LLMService._call_gemini_api(selected_prompt)
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            raise ValueError(f"Explanation generation failed: {str(e)}")

    @staticmethod
    def generate_recommendations(
        shap_values: Dict[str, float],
        sensitive_attribute: str
    ) -> List[str]:
        """
        Generate actionable recommendations using Gemini 1.5 Flash.
        """
        try:
            context = {
                "sensitive_attribute": sensitive_attribute,
                "shap_values": {k: round(float(v), 3) for k, v in shap_values.items()},
                "sensitive_impact": abs(shap_values.get(sensitive_attribute, 0.0))
            }
            
            prompt = f"""
            You are an AI governance consultant. Based on the SHAP values below,
            generate 2-3 actionable recommendations to improve model fairness and compliance.
            
            If '{sensitive_attribute}' has high impact, recommend:
            - Fairness-aware retraining
            - Synthetic data balancing
            - Threshold adjustment
            
            If no bias detected, recommend monitoring.
            
            Return only the recommendations, one per line, starting with '• '.
            
            Context: {json.dumps(context, indent=2)}
            """
            
            # Call Gemini 1.5 Flash
            response_text = LLMService._call_gemini_api(prompt)
            
            # Parse bullet points
            recommendations = [
                line.strip()[2:] for line in response_text.split('\n') 
                if line.strip().startswith('•')
            ]
            
            # Fallback if parsing fails
            if not recommendations:
                recommendations = [
                    f"Review the influence of '{sensitive_attribute}' on model outcomes.",
                    "Ensure fairness constraints are applied during retraining.",
                    "Monitor model performance quarterly for bias drift."
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            raise ValueError(f"Recommendation generation failed: {str(e)}")