import pandas as pd
import io
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def process_csv_file(file_content: bytes) -> List[Dict[str, Any]]:
    """
    Process uploaded CSV file and convert to list of dictionaries
    
    Args:
        file_content: Raw bytes content of the uploaded CSV file
        
    Returns:
        List of dictionaries representing the dataset rows
    """
    try:
        # Decode the bytes to string
        content_str = file_content.decode('utf-8')
        
        # Read CSV into DataFrame
        df = pd.read_csv(io.StringIO(content_str))
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        logger.info(f"Successfully processed CSV with {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise ValueError(f"Failed to process CSV file: {str(e)}")

def validate_dataset_structure(data: List[Dict[str, Any]], target_variable: str, sensitive_attribute: str) -> None:
    """
    Validate that the dataset has required structure for bias detection
    
    Args:
        data: Dataset as list of dictionaries
        target_variable: Name of the target/output variable
        sensitive_attribute: Name of the sensitive attribute
        
    Raises:
        ValueError: If validation fails
    """
    if not data:
        raise ValueError("Dataset is empty")
    
    # Check if we have at least 2 records
    if len(data) < 2:
        raise ValueError("Dataset must contain at least 2 records for bias detection")
    
    # Check if target variable exists
    if target_variable not in data[0]:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset")
    
    # Check if sensitive attribute exists
    if sensitive_attribute not in data[0]:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in dataset")
    
    # Check if target variable has at least 2 unique values
    target_values = set(item[target_variable] for item in data)
    if len(target_values) < 2:
        raise ValueError(f"Target variable '{target_variable}' must have at least 2 unique values")
    
    logger.info(f"Dataset validation successful: {len(data)} records, target='{target_variable}', sensitive_attr='{sensitive_attribute}'")