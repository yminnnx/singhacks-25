"""
ML Model Integration for AML Streamlit App
Integrates the trained Gradient Boosting model into the application
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any

class AMLModelPredictor:
    """
    ML Model integration for real-time AML risk prediction
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the ML model predictor"""
        self.model_data = None
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained ML model"""
        try:
            self.model_data = joblib.load(model_path)
            self.is_loaded = True
            print(f"✅ Loaded {self.model_data['model_name']} model")
            print(f"   Accuracy: {self.model_data['performance_metrics']['accuracy']:.1%}")
            print(f"   Features: {len(self.model_data['feature_columns'])}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_loaded = False
    
    def predict_transaction_risk(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk for a single transaction using the trained model
        
        Args:
            transaction: Dictionary with transaction features
        
        Returns:
            Dictionary with risk prediction results
        """
        
        if not self.is_loaded:
            # Fallback to rule-based scoring
            return self._fallback_prediction(transaction)
        
        try:
            # Extract model components
            model = self.model_data['model']
            feature_columns = self.model_data['feature_columns']
            label_encoders = self.model_data['label_encoders']
            
            # Get optimal threshold if available (default to 0.5)
            optimal_threshold = self.model_data.get('optimal_threshold', 0.5)
            
            # Create feature vector
            features = self._prepare_features(transaction, feature_columns, label_encoders)
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            risk_probability = model.predict_proba(features_array)[0][1]
            
            # Use optimal threshold for classification
            risk_prediction = 1 if risk_probability >= optimal_threshold else 0
            
            # Convert to risk score (0-100)
            risk_score = risk_probability * 100
            
            return {
                'risk_score': float(risk_score),
                'risk_probability': float(risk_probability),
                'is_high_risk': bool(risk_prediction),
                'confidence': float(max(model.predict_proba(features_array)[0])),
                'model_used': self.model_data['model_name'],
                'prediction_type': 'ML_MODEL',
                'threshold_used': float(optimal_threshold)
            }
            
        except Exception as e:
            print(f"ML prediction failed: {e}, falling back to rules")
            return self._fallback_prediction(transaction)
    
    def _prepare_features(self, transaction: Dict[str, Any], feature_columns: list, label_encoders: dict) -> list:
        """Prepare features for ML model prediction"""
        features = []
        
        # Numerical features
        amount = transaction.get('amount', 0)
        features.append(np.log1p(amount))  # amount_log
        features.append(1 if amount > 100000 else 0)  # is_large_amount
        features.append(1 if amount > 1000000 else 0)  # is_very_large_amount
        features.append(1 if transaction.get('channel') == 'Cash' else 0)  # is_cash
        features.append(1 if transaction.get('customer_is_pep', False) else 0)  # customer_is_pep_int
        features.append(1 if transaction.get('sanctions_screening') == 'potential' else 0)  # sanctions_screening_int
        
        # Risk rating encoding
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        features.append(risk_mapping.get(transaction.get('customer_risk_rating'), 0))
        
        # Encode categorical features
        categorical_columns = ['booking_jurisdiction', 'currency', 'channel', 'customer_risk_rating', 
                              'product_type', 'originator_country', 'beneficiary_country']
        
        for col in categorical_columns:
            if col + '_encoded' in feature_columns and col in label_encoders:
                try:
                    value = str(transaction.get(col, 'Unknown'))
                    encoded_value = label_encoders[col].transform([value])[0]
                    features.append(encoded_value)
                except:
                    features.append(0)  # Unknown category
        
        return features
    
    def _fallback_prediction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based prediction when ML model is not available"""
        risk_score = 0.0
        
        # Amount-based risk
        amount = transaction.get('amount', 0)
        if amount > 1000000:
            risk_score += 30
        elif amount > 500000:
            risk_score += 20
        elif amount > 100000:
            risk_score += 10
        
        # PEP risk
        if transaction.get('customer_is_pep', False):
            risk_score += 20
        
        # Sanctions risk
        if transaction.get('sanctions_screening') == 'potential':
            risk_score += 40
        
        # Customer risk rating
        risk_rating_scores = {'High': 25, 'Medium': 15, 'Low': 0}
        risk_score += risk_rating_scores.get(transaction.get('customer_risk_rating'), 0)
        
        # Channel risk
        if transaction.get('channel') == 'Cash':
            risk_score += 15
        
        # Add some variation based on transaction ID
        if 'transaction_id' in transaction:
            variation = (hash(str(transaction['transaction_id'])) % 10) - 5
            risk_score += variation
        
        risk_score = min(max(risk_score, 0), 100)
        
        return {
            'risk_score': float(risk_score),
            'risk_probability': float(risk_score / 100),
            'is_high_risk': bool(risk_score > 60),
            'confidence': 0.8,  # Moderate confidence for rule-based
            'model_used': 'Rule-Based',
            'prediction_type': 'RULE_BASED'
        }
    
    def batch_predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Predict risk for a batch of transactions"""
        results = []
        
        for _, transaction in transactions.iterrows():
            prediction = self.predict_transaction_risk(transaction.to_dict())
            results.append(prediction)
        
        return pd.DataFrame(results)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {
                'status': 'No model loaded',
                'model_type': 'Rule-based fallback',
                'accuracy': 'N/A'
            }
        
        return {
            'status': 'Model loaded successfully',
            'model_type': self.model_data['model_name'],
            'accuracy': f"{self.model_data['performance_metrics']['accuracy']:.1%}",
            'precision': f"{self.model_data['performance_metrics']['precision']:.1%}",
            'recall': f"{self.model_data['performance_metrics']['recall']:.1%}",
            'f1_score': f"{self.model_data['performance_metrics']['f1_score']:.1%}",
            'features': len(self.model_data['feature_columns'])
        }

# Initialize global predictor
_global_predictor = None

def get_ml_predictor(model_path: str = None) -> AMLModelPredictor:
    """Get or create the global ML predictor instance"""
    global _global_predictor
    
    if _global_predictor is None:
        if model_path is None:
            # Try to find model in standard locations
            possible_paths = [
                'aml_risk_model.pkl',  # In current src directory
                'models/aml_risk_model.pkl',
                '../models/aml_risk_model.pkl',
                '../../models/aml_risk_model.pkl',
                os.path.join(os.path.dirname(__file__), 'aml_risk_model.pkl')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        _global_predictor = AMLModelPredictor(model_path)
    
    return _global_predictor

def predict_transaction_risk(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for single transaction prediction"""
    predictor = get_ml_predictor()
    return predictor.predict_transaction_risk(transaction)