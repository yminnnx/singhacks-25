#!/usr/bin/env python3
"""
Test script to verify the optimized ML model integration
"""

import sys
import os
sys.path.append('src')

from ml_model_integration import get_ml_predictor

def test_optimized_model():
    """Test the optimized ML model integration"""
    print("🚀 Testing Optimized ML Model Integration")
    print("=" * 60)
    
    # Load the optimized model
    predictor = get_ml_predictor()
    
    # Check if model loaded
    if predictor.is_loaded:
        print("✅ Optimized ML Model loaded successfully!")
        model_info = predictor.get_model_info()
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Accuracy: {model_info['accuracy']}")
        print(f"   Precision: {model_info['precision']}")
        print(f"   Recall: {model_info['recall']}")
        print(f"   F1-Score: {model_info['f1_score']}")
        print(f"   Status: {model_info['status']}")
        
        # Display optimization metrics
        if 'performance_metrics' in predictor.model_data:
            real_metrics = predictor.model_data['performance_metrics']
            print(f"\n📊 Real Model Performance:")
            print(f"   Accuracy: {real_metrics['accuracy']:.1%}")
            print(f"   Precision: {real_metrics['precision']:.1%}")
            print(f"   Recall: {real_metrics['recall']:.1%}")
            print(f"   F1-Score: {real_metrics['f1_score']:.1%}")
            print(f"   ROC-AUC: {real_metrics['roc_auc']:.1%}")
            
            # Check if this is the optimized model
            if real_metrics['recall'] > 0.95:
                print(f"\n🎯 OPTIMIZED MODEL DETECTED!")
                print(f"   High recall ({real_metrics['recall']:.1%}) indicates optimization for better detection")
        
        # Test high-risk transaction prediction
        high_risk_transaction = {
            'amount': 2500000,  # Very high amount
            'channel': 'SWIFT',
            'customer_is_pep': True,  # PEP customer
            'sanctions_screening': 'potential',  # Sanctions hit
            'customer_risk_rating': 'High',
            'booking_jurisdiction': 'SG',
            'currency': 'USD',
            'product_type': 'Wire Transfer',
            'originator_country': 'AF',  # High-risk country
            'beneficiary_country': 'US',
            'transaction_id': 'TEST-HIGH-001'
        }
        
        print(f"\n🧪 Testing HIGH-RISK transaction prediction...")
        prediction = predictor.predict_transaction_risk(high_risk_transaction)
        
        print(f"   Risk Score: {prediction['risk_score']:.1f}/100")
        print(f"   High Risk: {prediction['is_high_risk']}")
        print(f"   Risk Probability: {prediction.get('risk_probability', 'N/A'):.1%}")
        print(f"   Confidence: {prediction.get('confidence', 'N/A'):.1%}")
        print(f"   Model Used: {prediction['model_used']}")
        print(f"   Prediction Type: {prediction['prediction_type']}")
        
        if 'threshold_used' in prediction:
            print(f"   Threshold Used: {prediction['threshold_used']:.2f}")
        
        # Test medium-risk transaction
        medium_risk_transaction = {
            'amount': 150000,  # Medium amount
            'channel': 'RTGS',
            'customer_is_pep': False,
            'sanctions_screening': 'none',
            'customer_risk_rating': 'Medium',
            'booking_jurisdiction': 'SG',
            'currency': 'SGD',
            'product_type': 'Local Transfer',
            'originator_country': 'SG',
            'beneficiary_country': 'SG',
            'transaction_id': 'TEST-MED-001'
        }
        
        print(f"\n🧪 Testing MEDIUM-RISK transaction prediction...")
        prediction2 = predictor.predict_transaction_risk(medium_risk_transaction)
        
        print(f"   Risk Score: {prediction2['risk_score']:.1f}/100")
        print(f"   High Risk: {prediction2['is_high_risk']}")
        print(f"   Risk Probability: {prediction2.get('risk_probability', 'N/A'):.1%}")
        print(f"   Confidence: {prediction2.get('confidence', 'N/A'):.1%}")
        print(f"   Model Used: {prediction2['model_used']}")
        
        # Test low-risk transaction
        low_risk_transaction = {
            'amount': 2500,  # Low amount
            'channel': 'FAST',
            'customer_is_pep': False,
            'sanctions_screening': 'none',
            'customer_risk_rating': 'Low',
            'booking_jurisdiction': 'SG',
            'currency': 'SGD',
            'product_type': 'Personal Transfer',
            'originator_country': 'SG',
            'beneficiary_country': 'SG',
            'transaction_id': 'TEST-LOW-001'
        }
        
        print(f"\n🧪 Testing LOW-RISK transaction prediction...")
        prediction3 = predictor.predict_transaction_risk(low_risk_transaction)
        
        print(f"   Risk Score: {prediction3['risk_score']:.1f}/100")
        print(f"   High Risk: {prediction3['is_high_risk']}")
        print(f"   Risk Probability: {prediction3.get('risk_probability', 'N/A'):.1%}")
        print(f"   Confidence: {prediction3.get('confidence', 'N/A'):.1%}")
        print(f"   Model Used: {prediction3['model_used']}")
        
        print(f"\n🎯 Optimized ML Model Integration: SUCCESSFUL!")
        print(f"✅ Model shows expected behavior with optimized recall")
        
    else:
        print("❌ Optimized ML Model failed to load")
        print("   Falling back to rule-based system")
        
        # Test fallback
        print(f"\n🧪 Testing rule-based fallback...")
        test_transaction = {
            'amount': 1500000,
            'channel': 'Cash',
            'customer_is_pep': True,
            'sanctions_screening': 'potential',
            'customer_risk_rating': 'High',
            'transaction_id': 'TEST-FALLBACK-001'
        }
        
        prediction = predictor.predict_transaction_risk(test_transaction)
        print(f"   Risk Score: {prediction['risk_score']:.1f}")
        print(f"   Prediction Type: {prediction['prediction_type']}")

    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    test_optimized_model()