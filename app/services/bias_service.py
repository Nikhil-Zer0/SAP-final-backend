"""
bias_service.py
Bias Detection Engine for Ethical AI Auditor
Implements FairML metrics to detect demographic disparities in AI models
Alternative to SAP HANA FairML + SAP AI Core integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
import re

# AIF360 for fairness metrics (SAP HANA FairML alternative)
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

# Models and utilities
from app.models.bias_models import BiasMetric
from app.utils.data_processor import validate_dataset_structure

# Initialize logger
logger = logging.getLogger(__name__)


class BiasDetectionService:
    """
    Service for detecting bias in AI models using FairML metrics.
    
    This implements the 'Bias Detection Engine' from the Ethical AI Auditor proposal,
    providing automated, scalable bias detection to achieve 80% faster audits.
    
    Key Features:
    - Disparate Impact, Statistical Parity, Equal Opportunity Difference
    - EEOC 80% Rule compliance checking
    - Synthetic data-ready (prepares for fairness-aware retraining)
    - SAP BTP-compatible microservice design
    """

    # EEOC 80% Rule threshold for disparate impact
    FAIRNESS_THRESHOLD = 0.8

    @staticmethod
    def detect_bias(
        dataset: List[Dict[str, Any]],
        target_variable: str,
        sensitive_attribute: str,
        privileged_group: Any,
        unprivileged_group: Any
    ) -> Tuple[Dict[str, float], str, List[str], int]:
        """
        Detect bias using FairML metrics (AIF360).
        
        Args:
            dataset: Input dataset as list of dicts
            target_variable: Outcome variable (e.g., 'hired')
            sensitive_attribute: Sensitive attribute (e.g., 'gender')
            privileged_group: Privileged value (e.g., 'Male')
            unprivileged_group: Unprivileged value (e.g., 'Female')

        Returns:
            metrics, audit_status, recommendations, record_count
        """
        try:
            # Validate dataset structure
            validate_dataset_structure(dataset, target_variable, sensitive_attribute)
            logger.info(f"Dataset validation passed: {len(dataset)} records")

            # Store record count before any conversion
            record_count = len(dataset)

            # Convert to DataFrame
            df = pd.DataFrame(dataset)

            #STEP 1: AGGRESSIVE NA CLEANING - ULTIMATE VERSION
            logger.info("Starting aggressive NA cleaning process...")
            
            # Replace all possible NA indicators with pd.NA
            na_indicators = [
                '', ' ', 'N/A', 'NULL', None, 'nan', 'NaN', 'none', 'null', 'na', 
                'N/a', 'n/a', 'NA', '\t', '\n', '  ', '   ', '    ', 'NULL', 'NUL',
                'nil', 'NIL', '.', '..', '...', 'N/A ', ' N/A', 'N/ A', 'N / A',
                'N.A.', 'N. A.', 'N . A .', 'N.A', 'NA.', 'N A', 'N  A', 'N   A'
            ]
            
            # Apply to all string columns
            for col in df.select_dtypes(include=['object']).columns:
                # Convert to string first to handle all types
                df[col] = df[col].astype(str)
                
                # Clean whitespace
                df[col] = df[col].str.strip()
                
                # Replace NA indicators with pd.NA
                for indicator in na_indicators:
                    if indicator is None:
                        continue
                    # Use exact string replacement
                    df[col] = df[col].replace(str(indicator), pd.NA)
            
            # 🔥 STEP 2: CRITICAL COLUMN VALIDATION
            critical_columns = [target_variable, sensitive_attribute]
            
            # Check for NA in critical columns
            na_mask = df[critical_columns].isna().any(axis=1)
            na_count = na_mask.sum()
            
            if na_count > 0:
                logger.warning(f"Found {na_count} rows with NA in critical columns. Removing them.")
                initial_count = len(df)
                df = df[~na_mask]
                logger.info(f"Removed {na_count} rows with NA in critical columns. Remaining: {len(df)}")
            
            if len(df) == 0:
                raise ValueError("No valid data remaining after removing NA values from critical columns.")
            
            # 🔥 STEP 3: TARGET VARIABLE VALIDATION
            logger.info(f"Validating target variable '{target_variable}'...")
            
            # Convert to numeric
            df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
            
            # Check for NA after conversion
            na_after_conv = df[target_variable].isna().sum()
            if na_after_conv > 0:
                logger.warning(f"Found {na_after_conv} NA values after numeric conversion of '{target_variable}'. Removing them.")
                initial_count = len(df)
                df = df.dropna(subset=[target_variable])
                logger.info(f"Removed {na_after_conv} rows. Remaining: {len(df)}")
            
            # Ensure binary values (0 or 1)
            unique_values = df[target_variable].unique()
            logger.info(f"Unique values in '{target_variable}': {unique_values}")
            
            # Map non-binary values to 0/1 if needed
            if not set(unique_values).issubset({0, 1}):
                logger.warning(f"Non-binary values found in '{target_variable}'. Mapping to binary.")
                # Try common mappings
                if set(unique_values) == {0, 1, 2}:
                    df[target_variable] = df[target_variable].map({0: 0, 1: 1, 2: 1})
                elif set(unique_values) == {'0', '1'}:
                    df[target_variable] = df[target_variable].astype(int)
                elif set(unique_values) == {'No', 'Yes'}:
                    df[target_variable] = df[target_variable].map({'No': 0, 'Yes': 1})
                elif set(unique_values) == {'False', 'True'}:
                    df[target_variable] = df[target_variable].map({'False': 0, 'True': 1})
                else:
                    # Fallback: anything not 0 or 1 becomes 0
                    df[target_variable] = df[target_variable].apply(lambda x: 1 if x == 1 else 0)
            
            # Final check - must be integer
            df[target_variable] = df[target_variable].astype(int)
            
            # 🔥 STEP 4: SENSITIVE ATTRIBUTE VALIDATION
            logger.info(f"Validating sensitive attribute '{sensitive_attribute}'...")
            
            # Convert to string and clean
            df[sensitive_attribute] = df[sensitive_attribute].astype(str)
            df[sensitive_attribute] = df[sensitive_attribute].str.strip()
            
            # Check for NA
            na_count = df[sensitive_attribute].isna().sum()
            if na_count > 0:
                logger.warning(f"Found {na_count} NA values in '{sensitive_attribute}'. Removing them.")
                initial_count = len(df)
                df = df.dropna(subset=[sensitive_attribute])
                logger.info(f"Removed {na_count} rows. Remaining: {len(df)}")
            
            # 🔥 STEP 5: GROUP VALIDATION
            logger.info("Validating privileged and unprivileged groups...")
            
            # Normalize group values
            priv_str = str(privileged_group).strip()
            unpriv_str = str(unprivileged_group).strip()
            
            logger.info(f"Looking for privileged group: '{priv_str}'")
            logger.info(f"Looking for unprivileged group: '{unpriv_str}'")
            
            # Check group presence
            priv_count = (df[sensitive_attribute] == priv_str).sum()
            unpriv_count = (df[sensitive_attribute] == unpriv_str).sum()
            
            logger.info(f"Privileged group count: {priv_count}")
            logger.info(f"Unprivileged group count: {unpriv_count}")
            
            if priv_count == 0:
                # Try case-insensitive match
                priv_match = df[sensitive_attribute].str.lower() == priv_str.lower()
                priv_count = priv_match.sum()
                if priv_count > 0:
                    logger.info(f"Found {priv_count} case-insensitive matches for privileged group")
                    # Update the column to use consistent casing
                    df.loc[priv_match, sensitive_attribute] = priv_str
                else:
                    raise ValueError(f"Privileged group '{privileged_group}' not found in dataset")
            
            if unpriv_count == 0:
                # Try case-insensitive match
                unpriv_match = df[sensitive_attribute].str.lower() == unpriv_str.lower()
                unpriv_count = unpriv_match.sum()
                if unpriv_count > 0:
                    logger.info(f"Found {unpriv_count} case-insensitive matches for unprivileged group")
                    # Update the column to use consistent casing
                    df.loc[unpriv_match, sensitive_attribute] = unpriv_str
                else:
                    raise ValueError(f"Unprivileged group '{unprivileged_group}' not found in dataset")
            
            if priv_count < 5 or unpriv_count < 5:
                logger.warning(
                    f"One group has very few samples: "
                    f"Privileged='{priv_str}'({priv_count}), Unprivileged='{unpriv_str}'({unpriv_count}). "
                    "Results may be unreliable."
                )
            
            # 🔥 STEP 6: FINAL NA SCORCHED EARTH POLICY
            logger.info("Performing FINAL NA sweep before AIF360...")
            
            # Double-check for any remaining NA
            na_count = df.isna().sum().sum()
            if na_count > 0:
                logger.error(f"CRITICAL: Found {na_count} NA values before AIF360 processing!")
                
                # Log which columns have NA
                na_cols = df.columns[df.isna().any()].tolist()
                for col in na_cols:
                    count = df[col].isna().sum()
                    logger.error(f"  Column '{col}' has {count} NA values")
                
                # Impute with most frequent value (mode) for categorical, median for numeric
                for col in df.columns:
                    if df[col].isna().any():
                        if df[col].dtype == 'object':
                            mode_val = df[col].mode()
                            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                            logger.info(f"  Filling NA in '{col}' with mode value: '{fill_val}'")
                        else:
                            fill_val = df[col].median() if not pd.isna(df[col].median()) else 0
                            logger.info(f"  Filling NA in '{col}' with median value: {fill_val}")
                        
                        df[col] = df[col].fillna(fill_val)
            
            # 🔥 STEP 7: ULTIMATE NA CHECK - MIMIC AIF360'S STRICT CHECKS
            logger.info("Performing ULTIMATE NA CHECK - mimicking AIF360's internal checks...")
            
            # Convert all object columns to strings and check for empty/whitespace
            for col in df.select_dtypes(include=['object']).columns:
                # Check for empty strings or whitespace
                empty_mask = df[col].str.strip() == ''
                if empty_mask.any():
                    logger.warning(f"Found {empty_mask.sum()} empty/whitespace values in '{col}'. Replacing with 'Unknown'.")
                    df.loc[empty_mask, col] = 'Unknown'
            
            # Final check - absolutely no NA allowed
            if df.isna().any().any():
                logger.critical("CRITICAL ERROR: NA values still exist after all cleaning steps!")
                # Nuclear option - drop ALL rows with ANY NA
                initial_count = len(df)
                df = df.dropna()
                logger.critical(f"Dropped {initial_count - len(df)} rows to ensure NA-free data for AIF360")
                
                if len(df) == 0:
                    raise ValueError("No valid data remaining after aggressive NA cleaning.")
            
            # 🔥 STEP 8: ENCODE CATEGORICAL DATA FOR AIF360
            logger.info("Encoding categorical data for AIF360...")
            
            # Verify no NA in the final dataset
            assert not df.isna().any().any(), "DataFrame contains NA values before AIF360"
            
            # AIF360 requires all data to be numerical - encode categorical columns
            from sklearn.preprocessing import LabelEncoder
            
            label_encoders = {}
            encoded_df = df.copy()
            
            # Identify categorical columns (object type)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            logger.info(f"Encoding categorical columns: {categorical_cols}")
            
            # Encode each categorical column
            for col in categorical_cols:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                logger.info(f"Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
            # Get encoded values for privileged/unprivileged groups
            if sensitive_attribute in label_encoders:
                sensitive_le = label_encoders[sensitive_attribute]
                # Find encoded values for the groups
                priv_encoded = None
                unpriv_encoded = None
                
                for original_val, encoded_val in zip(sensitive_le.classes_, sensitive_le.transform(sensitive_le.classes_)):
                    if str(original_val).strip() == priv_str:
                        priv_encoded = int(encoded_val)
                        logger.info(f"Privileged group '{priv_str}' encoded as {priv_encoded}")
                    elif str(original_val).strip() == unpriv_str:
                        unpriv_encoded = int(encoded_val)
                        logger.info(f"Unprivileged group '{unpriv_str}' encoded as {unpriv_encoded}")
                
                if priv_encoded is None:
                    raise ValueError(f"Could not find encoded value for privileged group '{priv_str}'")
                if unpriv_encoded is None:
                    raise ValueError(f"Could not find encoded value for unprivileged group '{unpriv_str}'")
            else:
                # If sensitive attribute is already numeric, use original values
                priv_encoded = priv_str
                unpriv_encoded = unpriv_str
            
            # Prepare group definitions with encoded values
            # StandardDataset expects: privileged_classes=[[value]]
            # ClassificationMetric expects: privileged_groups=[{attr: value}]
            privileged_classes_for_dataset = [[priv_encoded]]
            privileged_groups_for_metric = [{sensitive_attribute: priv_encoded}]
            unprivileged_groups_for_metric = [{sensitive_attribute: unpriv_encoded}]
            
            logger.info(f"Using encoded privileged classes for dataset: {privileged_classes_for_dataset}")
            logger.info(f"Using encoded privileged groups for metric: {privileged_groups_for_metric}")
            logger.info(f"Using encoded unprivileged groups for metric: {unprivileged_groups_for_metric}")
            
            # 🔥 STEP 9: CONVERT TO STANDARD DATASET
            logger.info("Converting to StandardDataset for AIF360...")
            
            # Create StandardDataset with encoded data
            standard_dataset = StandardDataset(
                encoded_df,
                label_name=target_variable,
                favorable_classes=[1],
                protected_attribute_names=[sensitive_attribute],
                privileged_classes=privileged_classes_for_dataset
            )
            
            # 🔥 STEP 9: COMPUTE FAIRNESS METRICS
            logger.info("Computing fairness metrics...")
            
            metric = ClassificationMetric(
                standard_dataset,
                standard_dataset,
                privileged_groups=privileged_groups_for_metric,
                unprivileged_groups=unprivileged_groups_for_metric
            )
            
            # Compile metrics
            raw_metrics = {
                "disparate_impact": metric.disparate_impact(),
                "statistical_parity_difference": metric.statistical_parity_difference(),
                "equal_opportunity_difference": metric.equal_opportunity_difference(),
                "average_odds_difference": metric.average_odds_difference()
            }
            
            # Sanitize metrics
            metrics = {}
            for key, value in raw_metrics.items():
                if pd.isna(value) or value == float('inf') or value == float('-inf'):
                    logger.warning(f"Metric '{key}' was {value}, replacing with 0.0")
                    metrics[key] = 0.0
                else:
                    metrics[key] = float(value)
            
            # 🔥 STEP 10: DETERMINE AUDIT STATUS
            logger.info("Determining compliance status...")
            
            if metrics["disparate_impact"] < BiasDetectionService.FAIRNESS_THRESHOLD:
                audit_status = "NON-COMPLIANT"
                recommendations = [
                    f"Bias detected: Disparate impact ratio ({metrics['disparate_impact']:.2f}) below 0.8 threshold (EEOC 80% rule)",
                    "Recommendation: Retrain model with fairness constraints",
                    "Recommendation: Use synthetic data balancing for underrepresented groups",
                    "Recommendation: Adjust decision threshold for fairness"
                ]
            else:
                audit_status = "COMPLIANT"
                recommendations = [
                    "No significant bias detected based on disparate impact ratio",
                    "Continue monitoring for potential bias drift"
                ]
            
            # Calculate ground truth accuracy for validation
            actual_hiring_rates = df.groupby(sensitive_attribute)[target_variable].mean()
            if len(actual_hiring_rates) >= 2:
                rates = list(actual_hiring_rates.values())
                ground_truth_di = min(rates) / max(rates) if max(rates) > 0 else 0.0
                accuracy_note = f"Ground truth DI: {ground_truth_di:.3f}"
            else:
                accuracy_note = "Cannot calculate ground truth - insufficient groups"
            
            # Log final results with accuracy info
            logger.info(
                f"Bias detection COMPLETED | Status: {audit_status} | "
                f"Detected DI: {metrics['disparate_impact']:.3f} | "
                f"{accuracy_note} | Records: {len(df)}"
            )
            
            return metrics, audit_status, recommendations, len(df)

        except Exception as e:
            logger.exception(f"Critical error in bias detection: {str(e)}")
            raise ValueError(f"Bias detection failed: {str(e)}")

    @staticmethod
    def format_metrics(metrics: Dict[str, float], sensitive_attribute: str) -> List[BiasMetric]:
        """Format raw metrics into standardized BiasMetric objects."""
        formatted_metrics = []

        def safe_float(value: float) -> float:
            if pd.isna(value) or value == float('inf') or value == float('-inf'):
                return 0.0
            return float(value)

        attr = str(sensitive_attribute)

        # Disparate Impact
        formatted_metrics.append(BiasMetric(
            name="Disparate Impact Ratio",
            value=safe_float(metrics["disparate_impact"]),
            description=f"Ratio of favorable outcomes between privileged and unprivileged groups ({attr})",
            threshold=0.8,
            compliant=safe_float(metrics["disparate_impact"]) >= 0.8
        ))

        # Statistical Parity
        formatted_metrics.append(BiasMetric(
            name="Statistical Parity Difference",
            value=safe_float(metrics["statistical_parity_difference"]),
            description=f"Difference in selection rates between privileged and unprivileged groups ({attr})",
            threshold=0.1,
            compliant=abs(safe_float(metrics["statistical_parity_difference"])) <= 0.1
        ))

        # Equal Opportunity
        formatted_metrics.append(BiasMetric(
            name="Equal Opportunity Difference",
            value=safe_float(metrics["equal_opportunity_difference"]),
            description=f"Difference in true positive rates between groups ({attr})",
            threshold=0.1,
            compliant=abs(safe_float(metrics["equal_opportunity_difference"])) <= 0.1
        ))

        # Average Odds
        formatted_metrics.append(BiasMetric(
            name="Average Odds Difference",
            value=safe_float(metrics["average_odds_difference"]),
            description=f"Average of false positive and true positive rate differences ({attr})",
            threshold=0.1,
            compliant=abs(safe_float(metrics["average_odds_difference"])) <= 0.1
        ))

        return formatted_metrics

    @staticmethod
    def generate_recommendations(metrics: Dict[str, float], sensitive_attribute: str) -> List[str]:
        """Generate actionable recommendations."""
        di = metrics.get("disparate_impact", 0.0)
        spd = metrics.get("statistical_parity_difference", 0.0)
        eod = metrics.get("equal_opportunity_difference", 0.0)
        attr = str(sensitive_attribute)

        recommendations = []

        if di < 0.8:
            recommendations.append(
                f"Disparate impact ratio ({di:.2f}) below 0.8 threshold for {attr}. "
                "Consider retraining with fairness constraints or using synthetic data balancing."
            )
        else:
            recommendations.append("Disparate impact ratio is within acceptable limits.")

        if abs(spd) > 0.1:
            direction = "higher" if spd > 0 else "lower"
            recommendations.append(
                f"Statistical parity difference ({spd:.3f}) indicates "
                f"privileged group has {direction} selection rate for {attr}. Evaluate bias vs. real-world differences."
            )

        if abs(eod) > 0.1:
            direction = "higher" if eod > 0 else "lower"
            recommendations.append(
                f"Equal opportunity difference ({eod:.3f}) suggests "
                f"privileged group has {direction} true positive rate for {attr}. May indicate outcome bias."
            )

        if di >= 0.8 and abs(spd) <= 0.1 and abs(eod) <= 0.1:
            recommendations = [
                "No significant bias detected across all metrics.",
                "Continue regular monitoring for potential bias drift in production."
            ]

        return recommendations