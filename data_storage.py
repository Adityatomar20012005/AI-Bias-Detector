"""
Data Storage Module
SQLite-based persistent storage for fairness analyses
Enables audit trails, reproducibility, and comparative analysis
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class BiasAnalysisStorage:
    """
    SQLite database for storing fairness analyses
    
    Tables:
    - analyses: Main results
    - chat_history: Chatbot Q&A
    - mitigation_tests: Before/after comparisons
    - intersectional_metrics: Intersectional fairness data
    """
    
    def __init__(self, db_path='bias_analyses.db'):
        self.db_path = db_path
        self.connection = None
    
    def init_db(self):
        """Initialize database and create tables"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                target_column TEXT NOT NULL,
                sensitive_columns TEXT NOT NULL,
                num_samples INTEGER,
                accuracy REAL,
                dp_gap REAL,
                di_ratio REAL,
                model_type TEXT DEFAULT 'RandomForest',
                results_json TEXT NOT NULL
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
        ''')
        
        # Mitigation tests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mitigation_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                sensitive_column TEXT NOT NULL,
                strategy TEXT NOT NULL,
                accuracy_before REAL,
                accuracy_after REAL,
                di_before REAL,
                di_after REAL,
                dp_gap_before REAL,
                dp_gap_after REAL,
                timestamp TEXT NOT NULL,
                results_json TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
        ''')
        
        # Intersectional metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intersectional_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                attribute_pair TEXT NOT NULL,
                di_ratio REAL,
                dp_gap REAL,
                verdict TEXT,
                timestamp TEXT NOT NULL,
                details_json TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, dataset_name: str, target_column: str, 
                     sensitive_columns: List[str], analysis_results: Dict,
                     accuracy: float = 0, dp_gap: float = 0, di_ratio: float = 1.0) -> int:
        """
        Save analysis to database
        
        Args:
            dataset_name: Name of dataset
            target_column: Target column name
            sensitive_columns: List of sensitive columns
            analysis_results: Full analysis results dictionary
            accuracy: Model accuracy
            dp_gap: Demographic parity gap
            di_ratio: Disparate impact ratio
        
        Returns:
            analysis_id of saved analysis
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        sensitive_cols_json = json.dumps(sensitive_columns)
        results_json = json.dumps(analysis_results, default=str)
        
        try:
            cursor.execute('''
                INSERT INTO analyses 
                (timestamp, dataset_name, target_column, sensitive_columns,
                 accuracy, dp_gap, di_ratio, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, dataset_name, target_column, sensitive_cols_json,
                  accuracy, dp_gap, di_ratio, results_json))
            
            conn.commit()
            analysis_id = cursor.lastrowid
            
            return analysis_id
        
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return -1
        
        finally:
            conn.close()
    
    def load_analysis(self, analysis_id: int) -> Optional[Dict]:
        """
        Load analysis from database
        
        Args:
            analysis_id: ID of analysis to load
        
        Returns:
            Dictionary with analysis data or None
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
            row = cursor.fetchone()
            
            if row:
                analysis = {
                    'id': row[0],
                    'timestamp': row[1],
                    'dataset_name': row[2],
                    'target_column': row[3],
                    'sensitive_columns': json.loads(row[4]),
                    'num_samples': row[5],
                    'accuracy': row[6],
                    'dp_gap': row[7],
                    'di_ratio': row[8],
                    'model_type': row[9],
                    'results': json.loads(row[10])
                }
                return analysis
            else:
                return None
        
        except Exception as e:
            print(f"Error loading analysis: {e}")
            return None
        
        finally:
            conn.close()
    
    def load_analyses_summary(self, limit: int = 20) -> List[Dict]:
        """
        Load summary of recent analyses
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of analysis summaries
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, timestamp, dataset_name, target_column, 
                       sensitive_columns, accuracy, dp_gap, di_ratio, results_json
                FROM analyses
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            summaries = []
            for row in rows:
                summary = {
                    'id': row[0],
                    'timestamp': row[1],
                    'dataset_name': row[2],
                    'target_column': row[3],
                    'sensitive_columns': row[4],
                    'accuracy': row[5],
                    'dp_gap': row[6],
                    'di_ratio': row[7],
                    'results_json': row[8]
                }
                summaries.append(summary)
            
            return summaries
        
        except Exception as e:
            print(f"Error loading analyses summary: {e}")
            return []
        
        finally:
            conn.close()
    
    def save_chat_message(self, analysis_id: int, role: str, message: str):
        """
        Save chatbot message
        
        Args:
            analysis_id: Associated analysis ID
            role: 'user' or 'assistant'
            message: Message text
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO chat_history (analysis_id, role, message, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (analysis_id, role, message, timestamp))
            
            conn.commit()
        
        except Exception as e:
            print(f"Error saving chat message: {e}")
        
        finally:
            conn.close()
    
    def load_chat_history(self, analysis_id: int, limit: int = 50) -> List[Dict]:
        """
        Load chat history for analysis
        
        Args:
            analysis_id: Analysis ID
            limit: Maximum messages to return
        
        Returns:
            List of messages
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT role, message, timestamp
                FROM chat_history
                WHERE analysis_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (analysis_id, limit))
            
            rows = cursor.fetchall()
            
            messages = []
            for row in reversed(rows):
                messages.append({
                    'role': row[0],
                    'message': row[1],
                    'timestamp': row[2]
                })
            
            return messages
        
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return []
        
        finally:
            conn.close()
    
    def save_mitigation_test(self, analysis_id: int, sensitive_column: str,
                            strategy: str, accuracy_before: float, accuracy_after: float,
                            di_before: float, di_after: float,
                            dp_gap_before: float, dp_gap_after: float):
        """
        Save mitigation test results
        
        Args:
            analysis_id: Associated analysis ID
            sensitive_column: Sensitive column tested
            strategy: Mitigation strategy used
            accuracy_before/after: Accuracy before and after
            di_before/after: Disparate impact before and after
            dp_gap_before/after: DP gap before and after
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO mitigation_tests
                (analysis_id, sensitive_column, strategy,
                 accuracy_before, accuracy_after,
                 di_before, di_after,
                 dp_gap_before, dp_gap_after, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_id, sensitive_column, strategy,
                  accuracy_before, accuracy_after,
                  di_before, di_after,
                  dp_gap_before, dp_gap_after, timestamp))
            
            conn.commit()
        
        except Exception as e:
            print(f"Error saving mitigation test: {e}")
        
        finally:
            conn.close()
    
    def save_intersectional_metrics(self, analysis_id: int, attribute_pair: str,
                                   di_ratio: float, dp_gap: float, verdict: str,
                                   details: Dict = None):
        """
        Save intersectional fairness metrics
        
        Args:
            analysis_id: Associated analysis ID
            attribute_pair: e.g., "gender × race"
            di_ratio: Disparate impact ratio
            dp_gap: Demographic parity gap
            verdict: 'Fair', 'Mild', or 'Significant'
            details: Additional details dictionary
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        details_json = json.dumps(details or {})
        
        try:
            cursor.execute('''
                INSERT INTO intersectional_metrics
                (analysis_id, attribute_pair, di_ratio, dp_gap, verdict,
                 timestamp, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_id, attribute_pair, di_ratio, dp_gap, verdict,
                  timestamp, details_json))
            
            conn.commit()
        
        except Exception as e:
            print(f"Error saving intersectional metrics: {e}")
        
        finally:
            conn.close()
    
    def delete_analysis(self, analysis_id: int):
        """
        Delete analysis (cascades to chat history and test results)
        
        Args:
            analysis_id: ID of analysis to delete
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
            conn.commit()
        
        except Exception as e:
            print(f"Error deleting analysis: {e}")
        
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM analyses')
            n_analyses = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM chat_history')
            n_messages = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM mitigation_tests')
            n_tests = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(accuracy) FROM analyses')
            avg_accuracy = cursor.fetchone()[0] or 0
            
            return {
                'total_analyses': n_analyses,
                'total_chat_messages': n_messages,
                'total_mitigation_tests': n_tests,
                'average_accuracy': avg_accuracy
            }
        
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
        
        finally:
            conn.close()
