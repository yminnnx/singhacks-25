"""
Groq-Enhanced Document Corroboration System
Advanced AI-powered document verification for AML compliance using Groq API
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# Import required packages with error handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️ Groq package not available")
    Groq = None
    GROQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import io
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

@dataclass
class DocumentVerificationResult:
    """Comprehensive document verification result"""
    document_id: str
    risk_score: int  # 0-100
    authenticity_score: int  # 0-100
    completeness_score: int  # 0-100
    consistency_score: int  # 0-100
    
    # AI Analysis Results
    compliance_assessment: str
    fraud_indicators: List[str]
    missing_elements: List[str]
    inconsistencies: List[str]
    recommendations: List[str]
    
    # Technical Analysis
    document_type: str
    format_issues: List[str]
    quality_assessment: str
    metadata_analysis: Dict[str, Any]
    
    # Processing Details
    approval_status: str  # APPROVED, REJECTED, REVIEW
    confidence_level: float  # 0.0-1.0
    processing_timestamp: str

class GroqDocumentCorroborator:
    """Advanced document corroboration using Groq AI"""
    
    def __init__(self):
        self.groq_client = None
        self.api_key = os.getenv('GROQ_API_KEY')
        
        if self.api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=self.api_key)
                print("✅ Groq Document Corroborator initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Groq client: {e}")
        else:
            print("⚠️ GROQ_API_KEY not found or Groq not available")
    
    def analyze_document(self, file_content: bytes, filename: str, context: Dict[str, Any]) -> DocumentVerificationResult:
        """
        Perform comprehensive document analysis using Groq AI
        """
        
        # Generate unique document ID
        doc_id = f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(filename) % 10000:04d}"
        
        # Extract document text
        document_text = self._extract_text_from_document(file_content, filename)
        
        # Perform Groq AI analysis
        if self.groq_client and document_text:
            try:
                groq_analysis = self._groq_document_analysis(document_text, filename, context)
                return self._parse_groq_analysis(groq_analysis, doc_id, filename, context)
            except Exception as e:
                print(f"Groq analysis failed: {e}")
                return self._fallback_analysis(doc_id, filename, document_text, context)
        else:
            return self._fallback_analysis(doc_id, filename, document_text, context)
    
    def _extract_text_from_document(self, file_content: bytes, filename: str) -> str:
        """Extract text content from various document formats"""
        try:
            if filename.lower().endswith('.pdf') and PDF_AVAILABLE:
                return self._extract_pdf_text(file_content)
            elif filename.lower().endswith(('.jpg', '.jpeg', '.png')) and IMAGE_AVAILABLE:
                return self._extract_image_text(file_content)
            elif filename.lower().endswith('.txt'):
                return file_content.decode('utf-8', errors='ignore')
            else:
                return f"Document: {filename} (binary content)"
        except Exception as e:
            print(f"Text extraction failed: {e}")
            return f"Document: {filename} (extraction failed)"
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"PDF text extraction failed: {e}"
    
    def _extract_image_text(self, file_content: bytes) -> str:
        """Extract text from image using OCR (if available)"""
        try:
            # Try OCR if pytesseract is available
            import pytesseract
            
            image = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            return "Image document (OCR not available)"
        except Exception as e:
            return f"Image text extraction failed: {e}"
    
    def _groq_document_analysis(self, document_text: str, filename: str, context: Dict[str, Any]) -> str:
        """Perform Groq AI analysis on document text"""
        
        prompt = f"""
You are an expert AML (Anti-Money Laundering) compliance officer analyzing a document for authenticity, completeness, and regulatory compliance.

Document Information:
- Filename: {filename}
- Customer Type: {context.get('customer_type', 'Unknown')}
- Transaction Purpose: {context.get('transaction_purpose', 'Unknown')}
- Document Purpose: {context.get('document_purpose', 'Unknown')}
- Regulatory Framework: {context.get('regulatory_requirements', 'Unknown')}

Document Content:
{document_text[:4000]}

Please provide a comprehensive analysis in the following JSON format:
{{
    "authenticity_assessment": "Detailed assessment of document authenticity",
    "completeness_assessment": "Analysis of document completeness", 
    "compliance_assessment": "Regulatory compliance evaluation",
    "fraud_indicators": ["List of potential fraud indicators"],
    "missing_elements": ["List of missing required elements"],
    "inconsistencies": ["List of internal inconsistencies"],
    "format_issues": ["List of format-related issues"],
    "risk_score": 25,
    "authenticity_score": 85,
    "completeness_score": 90,
    "consistency_score": 80,
    "approval_status": "APPROVED",
    "confidence_level": 0.85,
    "recommendations": ["List of specific recommendations"],
    "document_type": "Identified document type",
    "quality_assessment": "Overall quality assessment"
}}

Focus on AML compliance requirements, document authenticity indicators, and provide specific, actionable recommendations.
"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert AML compliance officer with deep knowledge of document verification, fraud detection, and regulatory requirements."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=2000
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Groq API call failed: {e}")
    
    def _parse_groq_analysis(self, groq_response: str, doc_id: str, filename: str, context: Dict[str, Any]) -> DocumentVerificationResult:
        """Parse Groq AI response into DocumentVerificationResult"""
        
        try:
            # Try to extract JSON from the response
            start_idx = groq_response.find('{')
            end_idx = groq_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = groq_response[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                analysis = self._extract_analysis_from_text(groq_response)
            
            return DocumentVerificationResult(
                document_id=doc_id,
                risk_score=analysis.get('risk_score', 50),
                authenticity_score=analysis.get('authenticity_score', 75),
                completeness_score=analysis.get('completeness_score', 80),
                consistency_score=analysis.get('consistency_score', 75),
                compliance_assessment=analysis.get('compliance_assessment', groq_response[:500] + "..."),
                fraud_indicators=analysis.get('fraud_indicators', []),
                missing_elements=analysis.get('missing_elements', []),
                inconsistencies=analysis.get('inconsistencies', []),
                recommendations=analysis.get('recommendations', []),
                document_type=analysis.get('document_type', 'Unknown'),
                format_issues=analysis.get('format_issues', []),
                quality_assessment=analysis.get('quality_assessment', 'Standard'),
                metadata_analysis={
                    'filename': filename,
                    'analysis_method': 'Groq AI',
                    'context': context,
                    'raw_response_length': len(groq_response)
                },
                approval_status=analysis.get('approval_status', 'REVIEW'),
                confidence_level=analysis.get('confidence_level', 0.8),
                processing_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            print(f"Failed to parse Groq response: {e}")
            return self._fallback_analysis(doc_id, filename, groq_response, context)
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Extract analysis data from unstructured text"""
        analysis = {
            'risk_score': 50,
            'authenticity_score': 75,
            'completeness_score': 80,
            'consistency_score': 75,
            'compliance_assessment': text[:500] + "...",
            'fraud_indicators': [],
            'missing_elements': [],
            'inconsistencies': [],
            'recommendations': [],
            'document_type': 'Unknown',
            'format_issues': [],
            'quality_assessment': 'Standard',
            'approval_status': 'REVIEW',
            'confidence_level': 0.7
        }
        
        # Simple text analysis to extract key information
        text_lower = text.lower()
        
        # Determine approval status
        if 'approved' in text_lower or 'accept' in text_lower:
            analysis['approval_status'] = 'APPROVED'
            analysis['risk_score'] = 25
        elif 'rejected' in text_lower or 'deny' in text_lower or 'fraud' in text_lower:
            analysis['approval_status'] = 'REJECTED'
            analysis['risk_score'] = 85
        
        # Extract recommendations (simple heuristic)
        if 'recommend' in text_lower:
            lines = text.split('\n')
            recommendations = [line.strip() for line in lines if 'recommend' in line.lower()]
            analysis['recommendations'] = recommendations[:3]  # Top 3
        
        return analysis
    
    def _fallback_analysis(self, doc_id: str, filename: str, document_text: str, context: Dict[str, Any]) -> DocumentVerificationResult:
        """Provide fallback analysis when Groq AI is unavailable"""
        
        # Simple rule-based analysis
        risk_score = 30
        authenticity_score = 80
        approval_status = "REVIEW"
        
        fraud_indicators = []
        recommendations = ["Manual review recommended", "Verify document authenticity"]
        
        # Basic heuristics
        if len(document_text) < 100:
            fraud_indicators.append("Document appears to have minimal content")
            risk_score += 20
            authenticity_score -= 15
        
        if context.get('customer_type') == 'PEP':
            fraud_indicators.append("Enhanced due diligence required for PEP customer")
            risk_score += 15
        
        if risk_score > 70:
            approval_status = "REJECTED"
        elif risk_score < 40:
            approval_status = "APPROVED"
        
        return DocumentVerificationResult(
            document_id=doc_id,
            risk_score=min(risk_score, 100),
            authenticity_score=max(authenticity_score, 0),
            completeness_score=75,
            consistency_score=70,
            compliance_assessment="Fallback analysis completed. Groq AI was not available for enhanced verification.",
            fraud_indicators=fraud_indicators,
            missing_elements=["Advanced AI analysis not performed"],
            inconsistencies=[],
            recommendations=recommendations,
            document_type="Unknown",
            format_issues=[],
            quality_assessment="Standard (fallback)",
            metadata_analysis={
                'filename': filename,
                'analysis_method': 'Fallback Rule-Based',
                'context': context,
                'document_length': len(document_text)
            },
            approval_status=approval_status,
            confidence_level=0.6,
            processing_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

# Global instance
_groq_corroborator = None

def get_groq_corroborator() -> GroqDocumentCorroborator:
    """Get global instance of Groq document corroborator"""
    global _groq_corroborator
    if _groq_corroborator is None:
        _groq_corroborator = GroqDocumentCorroborator()
    return _groq_corroborator

# Test function
def test_groq_corroboration():
    """Test the Groq document corroboration system"""
    print("🔍 Testing Groq Document Corroboration System...")
    
    corroborator = get_groq_corroborator()
    
    if corroborator.groq_client:
        print("✅ Groq Document Corroborator initialized")
    else:
        print("⚠️ Groq client not available - fallback mode")
    
    # Test with sample data
    sample_content = b"Sample document content for testing"
    sample_context = {
        'customer_type': 'Individual',
        'transaction_purpose': 'Investment',
        'document_purpose': 'Identity Verification',
        'regulatory_requirements': 'Swiss AML'
    }
    
    result = corroborator.analyze_document(sample_content, "test_document.txt", sample_context)
    
    print(f"📊 ANALYSIS RESULTS:")
    print(f"🎯 Document ID: {result.document_id}")
    print(f"🔴 Risk Score: {result.risk_score}/100")
    print(f"🔒 Authenticity: {result.authenticity_score}/100")
    print(f"📋 Completeness: {result.completeness_score}/100")
    print(f"⚖️ Approval Status: {result.approval_status}")
    print(f"🤖 AI Confidence: {result.confidence_level:.1%}")
    
    if result.fraud_indicators:
        print(f"⚠️ FRAUD INDICATORS:")
        for indicator in result.fraud_indicators:
            print(f"   • {indicator}")
    
    if result.recommendations:
        print(f"💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_groq_corroboration()
