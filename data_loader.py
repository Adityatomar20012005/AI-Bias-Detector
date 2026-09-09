"""
Data Loader Module
Handles loading UCI Adult dataset and CSV files
with validation and preprocessing
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import io
import requests


class DataLoader:
    """Load and validate datasets for fairness analysis"""
    
    def __init__(self):
        self.uci_adult_data = None
    
    def load_uci_adult(self, use_cache=True):
        """
        Load UCI Adult Income dataset
        
        Dataset: Census income (classification task)
        - 48,842 samples
        - 14 features
        - Sensitive: age, sex, race
        - Target: income (>50K or <=50K)
        
        Reference: Lichman, M. (2013). UCI Machine Learning Repository
        
        Args:
            use_cache: Use cached data if available
        
        Returns:
            pandas DataFrame with dataset
        """
        
        try:
            # Try fetching from UCI ML Repository
            adult = fetch_ucirepo(id=2)
            
            X = adult.data
            y = adult.targets
            
            # Combine X and y
            df = pd.concat([X, y], axis=1)
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            return df
        
        except Exception as e:
            print(f"Error loading from UCI ML Repository: {e}")
            print("Attempting alternative loading method...")
            
            # Fallback: Try loading from direct CSV URL
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
                column_names = [
                    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                    'income'
                ]
                
                df = pd.read_csv(url, names=column_names, header=None, skipinitialspace=True)
                
                # Clean column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                return df
            
            except Exception as e2:
                print(f"Error loading from URL: {e2}")
                print("Creating synthetic UCI Adult-like dataset...")
                
                # Create synthetic dataset with UCI Adult-like structure
                return self._create_synthetic_adult()
    
    def _create_synthetic_adult(self):
        """Create synthetic UCI Adult-like dataset for demo purposes"""
        
        np.random.seed(42)
        n_samples = 1000  # Smaller for demo
        
        df = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'workclass': np.random.choice(['Private', 'Self-emp', 'Federal-gov', 'Local-gov', 'State-gov'], n_samples),
            'education': np.random.choice(['Preschool', '9th', 'HS-grad', 'Bachelors', 'Masters', 'Doctorate'], n_samples),
            'marital_status': np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], n_samples),
            'occupation': np.random.choice(['Tech', 'Sales', 'Executive', 'Craft', 'Service'], n_samples),
            'relationship': np.random.choice(['Spouse', 'Not-in-family', 'Own-child', 'Unmarried'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'capital_gain': np.random.randint(0, 100000, n_samples),
            'capital_loss': np.random.randint(0, 50000, n_samples),
            'hours_per_week': np.random.randint(0, 100, n_samples),
            'native_country': np.random.choice(['United-States', 'Mexico', 'Philippines', 'Canada'], n_samples),
        })
        
        # Create target: income based on age and hours worked (simulating bias)
        df['income'] = ((df['age'] > 40) & (df['hours_per_week'] > 35)).astype(int)
        
        # Add bias: males more likely to have high income
        male_mask = df['sex'] == 'Male'
        df.loc[male_mask, 'income'] = np.where(
            np.random.random(male_mask.sum()) < 0.4,
            1,
            df.loc[male_mask, 'income']
        )
        
        return df
    
    def load_csv(self, file_path):
        """
        Load CSV file
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            pandas DataFrame
        """
        
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    def validate_data(self, df, target_column, sensitive_columns):
        """
        Validate dataset for fairness analysis
        
        Args:
            df: DataFrame to validate
            target_column: Target variable column
            sensitive_columns: Sensitive attributes
        
        Returns:
            Tuple (is_valid, error_messages)
        """
        
        errors = []
        warnings = []
        
        # Check minimum size
        if len(df) < 50:
            errors.append("Dataset too small (minimum 50 samples)")
        
        # Check target column exists
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found")
        
        # Check sensitive columns exist
        for col in sensitive_columns:
            if col not in df.columns:
                errors.append(f"Sensitive column '{col}' not found")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 50:
                errors.append(f"Too many missing values ({missing_pct:.1f}%)")
            else:
                warnings.append(f"Dataset has {missing_pct:.1f}% missing values (will be dropped)")
        
        # Check target is binary or can be made binary
        if target_column in df.columns:
            unique_targets = df[target_column].nunique()
            if unique_targets > 10:
                warnings.append(f"Target has {unique_targets} unique values (may need binary conversion)")
        
        # Check sensitive columns have reasonable cardinality
        for col in sensitive_columns:
            if col in df.columns:
                unique_vals = df[col].nunique()
                if unique_vals > 50:
                    warnings.append(f"Sensitive column '{col}' has {unique_vals} unique values (very high cardinality)")
                elif unique_vals < 2:
                    errors.append(f"Sensitive column '{col}' has less than 2 unique values")
        
        return len(errors) == 0, errors, warnings
    
    def preprocess_data(self, df, target_column, sensitive_columns):
        """
        Preprocess data for fairness analysis
        
        Args:
            df: Raw DataFrame
            target_column: Target variable
            sensitive_columns: Sensitive attributes
        
        Returns:
            Cleaned DataFrame
        """
        
        df_clean = df.copy()
        
        # Drop rows with missing values in critical columns
        critical_cols = [target_column] + sensitive_columns
        df_clean = df_clean.dropna(subset=critical_cols)
        
        # Drop duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Remove rows with missing values anywhere
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def get_dataset_info(self, df, target_column, sensitive_columns):
        """
        Get information about dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        
        info = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'n_features': len([col for col in df.columns if col not in [target_column] + sensitive_columns]),
            'target_column': target_column,
            'target_unique_values': df[target_column].nunique(),
            'sensitive_columns': sensitive_columns,
            'missing_values_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'column_types': df.dtypes.to_dict()
        }
        
        # Add sensitive column info
        for col in sensitive_columns:
            if col in df.columns:
                info[f'{col}_unique_values'] = df[col].nunique()
                info[f'{col}_value_counts'] = df[col].value_counts().to_dict()
        
        return info
