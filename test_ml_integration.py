"""
Test script to verify ML model integration is working
"""

import sys
import os
sys.path.append('src')

from ml_model_integration import get_ml_predictor

def test_ml_model():
    """Test the ML model integration"""
    print("🔬 Testing ML Model Integration")
    print("=" * 50)
    
    # Load the model
    predictor = get_ml_predictor('src/aml_risk_model.pkl')
    
    # Check if model loaded
    if predictor.is_loaded:
        print("✅ ML Model loaded successfully!")
        model_info = predictor.get_model_info()
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Accuracy: {model_info['accuracy']}")
        print(f"   Status: {model_info['status']}")
        
        # Test prediction
        test_transaction = {
            'amount': 1500000,  # High amount
            'channel': 'SWIFT',
            'customer_is_pep': True,  # PEP customer
            'sanctions_screening': 'potential',  # Sanctions hit
            'customer_risk_rating': 'High',
            'booking_jurisdiction': 'SG',
            'currency': 'USD',
            'product_type': 'Wire Transfer',
            'originator_country': 'SG',
            'beneficiary_country': 'US'
        }
        
        print(f"\n🧪 Testing prediction for high-risk transaction...")
        prediction = predictor.predict_transaction_risk(test_transaction)
        
        print(f"   Risk Score: {prediction['risk_score']:.1f}")
        print(f"   High Risk: {prediction['is_high_risk']}")
        print(f"   Confidence: {prediction.get('confidence', 'N/A')}")
        print(f"   Model Used: {prediction['model_used']}")
        
        # Test low-risk transaction
        low_risk_transaction = {
            'amount': 5000,  # Low amount
            'channel': 'RTGS',
            'customer_is_pep': False,
            'sanctions_screening': 'none',
            'customer_risk_rating': 'Low',
            'booking_jurisdiction': 'SG',
            'currency': 'SGD',
            'product_type': 'Local Transfer',
            'originator_country': 'SG',
            'beneficiary_country': 'SG'
        }
        
        print(f"\n🧪 Testing prediction for low-risk transaction...")
        prediction2 = predictor.predict_transaction_risk(low_risk_transaction)
        
        print(f"   Risk Score: {prediction2['risk_score']:.1f}")
        print(f"   High Risk: {prediction2['is_high_risk']}")
        print(f"   Confidence: {prediction2.get('confidence', 'N/A')}")
        print(f"   Model Used: {prediction2['model_used']}")
        
        print(f"\n🎯 ML Model Integration: SUCCESSFUL!")
        
    else:
        print("❌ ML Model failed to load")
        print("   Falling back to rule-based system")
        
        # Test fallback
        print(f"\n🧪 Testing rule-based fallback...")
        test_transaction = {
            'amount': 1500000,
            'channel': 'Cash',
            'customer_is_pep': True,
            'sanctions_screening': 'potential',
            'customer_risk_rating': 'High',
            'transaction_id': 'TEST-001'
        }
        
        prediction = predictor.predict_transaction_risk(test_transaction)
        print(f"   Risk Score: {prediction['risk_score']:.1f}")
        print(f"   Prediction Type: {prediction['prediction_type']}")

if __name__ == "__main__":
    test_ml_model()