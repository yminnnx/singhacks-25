"""
Document Processing System for AML Corroboration
Handles document upload, processing, OCR, and validation for compliance verification
"""

import os
import io
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import re

# For document processing
try:
    import PyPDF2
    import pytesseract
    from PIL import Image, ImageEnhance, ExifTags
    import cv2
    import numpy as np
except ImportError:
    print("Some document processing libraries not available. Install with: pip install PyPDF2 pytesseract Pillow opencv-python")

class DocumentType(Enum):
    PDF = "PDF"
    IMAGE = "Image"
    TEXT = "Text"
    UNKNOWN = "Unknown"

class ValidationIssue(Enum):
    FORMATTING_ERROR = "Formatting Error"
    SPELLING_MISTAKE = "Spelling Mistake"
    MISSING_SECTION = "Missing Section"
    INCONSISTENT_DATA = "Inconsistent Data"
    POOR_IMAGE_QUALITY = "Poor Image Quality"
    SUSPICIOUS_METADATA = "Suspicious Metadata"
    POTENTIAL_TAMPERING = "Potential Tampering"

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class DocumentIssue:
    issue_type: ValidationIssue
    severity: RiskLevel
    description: str
    location: str  # Page number, line number, etc.
    evidence: Dict[str, Any]
    recommendation: str

@dataclass
class DocumentAnalysisResult:
    document_id: str
    document_type: DocumentType
    file_name: str
    file_size: int
    processing_timestamp: datetime
    text_content: str
    metadata: Dict[str, Any]
    issues: List[DocumentIssue]
    risk_score: float
    overall_assessment: str
    recommendations: List[str]

class DocumentProcessor:
    """
    Core document processing engine for AML corroboration
    """
    
    def __init__(self):
        self.setup_logging()
        self.supported_formats = {
            '.pdf': DocumentType.PDF,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.tiff': DocumentType.IMAGE,
            '.bmp': DocumentType.IMAGE,
            '.txt': DocumentType.TEXT,
            '.doc': DocumentType.TEXT,
            '.docx': DocumentType.TEXT
        }
        
        # Expected document sections for compliance documents
        self.expected_sections = [
            'client information', 'address', 'identification', 'purpose',
            'source of funds', 'beneficial owner', 'signature', 'date'
        ]
        
        # Common spelling mistakes in financial documents
        self.common_misspellings = {
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'existance': 'existence',
            'maintainance': 'maintenance'
        }
    
    def setup_logging(self):
        """Setup logging for audit trail"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, file_path: str) -> DocumentAnalysisResult:
        """Main function to process any type of document"""
        try:
            # Determine document type
            doc_type = self._determine_document_type(file_path)
            
            # Generate document ID
            doc_id = self._generate_document_id(file_path)
            
            # Get file info
            file_stats = os.stat(file_path)
            file_name = os.path.basename(file_path)
            
            # Extract content based on type
            text_content = ""
            metadata = {}
            
            if doc_type == DocumentType.PDF:
                text_content, metadata = self._process_pdf(file_path)
            elif doc_type == DocumentType.IMAGE:
                text_content, metadata = self._process_image(file_path)
            elif doc_type == DocumentType.TEXT:
                text_content, metadata = self._process_text(file_path)
            
            # Analyze content for issues
            issues = self._analyze_content(text_content, metadata, doc_type)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(issues)
            
            # Generate assessment and recommendations
            assessment = self._generate_assessment(risk_score, issues)
            recommendations = self._generate_recommendations(issues)
            
            result = DocumentAnalysisResult(
                document_id=doc_id,
                document_type=doc_type,
                file_name=file_name,
                file_size=file_stats.st_size,
                processing_timestamp=datetime.now(),
                text_content=text_content,
                metadata=metadata,
                issues=issues,
                risk_score=risk_score,
                overall_assessment=assessment,
                recommendations=recommendations
            )
            
            self.logger.info(f"Processed document {doc_id}: {len(issues)} issues found, risk score: {risk_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _determine_document_type(self, file_path: str) -> DocumentType:
        """Determine document type from file extension"""
        _, ext = os.path.splitext(file_path.lower())
        return self.supported_formats.get(ext, DocumentType.UNKNOWN)
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"DOC-{timestamp}-{file_hash}"
    
    def _process_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process PDF document"""
        text_content = ""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                        'modification_date': pdf_reader.metadata.get('/ModDate', '')
                    })
                
                metadata['page_count'] = len(pdf_reader.pages)
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        self.logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {e}")
            metadata['processing_error'] = str(e)
        
        return text_content, metadata
    
    def _process_image(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process image document with OCR"""
        text_content = ""
        metadata = {}
        
        try:
            # Load image
            image = Image.open(file_path)
            
            # Extract EXIF metadata
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            metadata.update({
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'exif_data': exif_data
            })
            
            # Enhance image for better OCR
            enhanced_image = self._enhance_image_for_ocr(image)
            
            # Perform OCR
            text_content = pytesseract.image_to_string(enhanced_image)
            
            # Get OCR confidence scores
            ocr_data = pytesseract.image_to_data(enhanced_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            
            if confidences:
                metadata['ocr_confidence_avg'] = sum(confidences) / len(confidences)
                metadata['ocr_confidence_min'] = min(confidences)
            
        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {e}")
            metadata['processing_error'] = str(e)
        
        return text_content, metadata
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            return image
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _process_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process text document"""
        text_content = ""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Basic text statistics
            lines = text_content.split('\n')
            words = text_content.split()
            
            metadata.update({
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(text_content),
                'encoding': 'utf-8'
            })
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text_content = file.read()
                    metadata['encoding'] = encoding
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            metadata['processing_error'] = str(e)
        
        return text_content, metadata
    
    def _analyze_content(self, text_content: str, metadata: Dict[str, Any], doc_type: DocumentType) -> List[DocumentIssue]:
        """Analyze document content for various issues"""
        issues = []
        
        # Format validation
        issues.extend(self._check_formatting_issues(text_content))
        
        # Spelling and grammar
        issues.extend(self._check_spelling_issues(text_content))
        
        # Missing sections
        issues.extend(self._check_missing_sections(text_content))
        
        # Data consistency
        issues.extend(self._check_data_consistency(text_content))
        
        # Metadata analysis
        issues.extend(self._check_metadata_issues(metadata, doc_type))
        
        # Image-specific checks
        if doc_type == DocumentType.IMAGE:
            issues.extend(self._check_image_quality_issues(metadata))
        
        return issues
    
    def _check_formatting_issues(self, text_content: str) -> List[DocumentIssue]:
        """Check for formatting inconsistencies"""
        issues = []
        
        # Check for double spaces
        if '  ' in text_content:
            double_space_count = text_content.count('  ')
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.FORMATTING_ERROR,
                severity=RiskLevel.LOW,
                description=f"Found {double_space_count} instances of double spacing",
                location="Throughout document",
                evidence={'double_space_count': double_space_count},
                recommendation="Review document formatting for consistency"
            ))
        
        # Check for inconsistent line endings
        crlf_count = text_content.count('\r\n')
        lf_count = text_content.count('\n') - crlf_count
        
        if crlf_count > 0 and lf_count > 0:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.FORMATTING_ERROR,
                severity=RiskLevel.LOW,
                description="Inconsistent line endings detected",
                location="Throughout document",
                evidence={'crlf_count': crlf_count, 'lf_count': lf_count},
                recommendation="Standardize line endings"
            ))
        
        # Check for unusual characters
        unusual_chars = set(char for char in text_content if ord(char) > 127 and char not in 'áéíóúñü€£¥')
        if unusual_chars:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.FORMATTING_ERROR,
                severity=RiskLevel.MEDIUM,
                description=f"Unusual characters found: {', '.join(unusual_chars)}",
                location="Various locations",
                evidence={'unusual_characters': list(unusual_chars)},
                recommendation="Verify character encoding and document authenticity"
            ))
        
        return issues
    
    def _check_spelling_issues(self, text_content: str) -> List[DocumentIssue]:
        """Check for spelling mistakes"""
        issues = []
        text_lower = text_content.lower()
        
        for misspelling, correct in self.common_misspellings.items():
            if misspelling in text_lower:
                count = text_lower.count(misspelling)
                issues.append(DocumentIssue(
                    issue_type=ValidationIssue.SPELLING_MISTAKE,
                    severity=RiskLevel.LOW,
                    description=f"Misspelling found: '{misspelling}' (should be '{correct}') - {count} occurrences",
                    location="Various locations",
                    evidence={'misspelling': misspelling, 'correct': correct, 'count': count},
                    recommendation=f"Correct spelling to '{correct}'"
                ))
        
        return issues
    
    def _check_missing_sections(self, text_content: str) -> List[DocumentIssue]:
        """Check for missing required sections"""
        issues = []
        text_lower = text_content.lower()
        
        missing_sections = []
        for section in self.expected_sections:
            # Look for section keywords
            section_keywords = section.split()
            found = any(keyword in text_lower for keyword in section_keywords)
            
            if not found:
                missing_sections.append(section)
        
        if missing_sections:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.MISSING_SECTION,
                severity=RiskLevel.HIGH,
                description=f"Missing required sections: {', '.join(missing_sections)}",
                location="Document structure",
                evidence={'missing_sections': missing_sections},
                recommendation="Ensure all required sections are present and properly labeled"
            ))
        
        return issues
    
    def _check_data_consistency(self, text_content: str) -> List[DocumentIssue]:
        """Check for data inconsistencies"""
        issues = []
        
        # Find dates in different formats
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
        
        found_date_formats = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_content)
            if matches:
                found_date_formats.append(pattern)
        
        if len(found_date_formats) > 1:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.INCONSISTENT_DATA,
                severity=RiskLevel.MEDIUM,
                description="Multiple date formats detected",
                location="Various locations",
                evidence={'date_formats_found': len(found_date_formats)},
                recommendation="Standardize date format throughout document"
            ))
        
        # Find monetary amounts
        money_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,000.00
            r'USD\s*[\d,]+\.?\d*',  # USD 1,000.00
            r'EUR\s*[\d,]+\.?\d*',  # EUR 1,000.00
        ]
        
        amounts = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text_content)
            amounts.extend(matches)
        
        # Check for amount inconsistencies
        if len(set(amounts)) != len(amounts):
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.INCONSISTENT_DATA,
                severity=RiskLevel.HIGH,
                description="Duplicate monetary amounts found",
                location="Financial sections",
                evidence={'duplicate_amounts': len(amounts) - len(set(amounts))},
                recommendation="Verify accuracy of monetary amounts"
            ))
        
        return issues
    
    def _check_metadata_issues(self, metadata: Dict[str, Any], doc_type: DocumentType) -> List[DocumentIssue]:
        """Check metadata for suspicious indicators"""
        issues = []
        
        # Check for processing errors
        if 'processing_error' in metadata:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.POOR_IMAGE_QUALITY,
                severity=RiskLevel.HIGH,
                description=f"Document processing error: {metadata['processing_error']}",
                location="Document processing",
                evidence={'error': metadata['processing_error']},
                recommendation="Verify document integrity and format"
            ))
        
        # Check PDF metadata
        if doc_type == DocumentType.PDF:
            creator = metadata.get('creator', '').lower()
            producer = metadata.get('producer', '').lower()
            
            # Suspicious creators/producers
            suspicious_keywords = ['fake', 'temp', 'test', 'demo', 'trial']
            if any(keyword in creator or keyword in producer for keyword in suspicious_keywords):
                issues.append(DocumentIssue(
                    issue_type=ValidationIssue.SUSPICIOUS_METADATA,
                    severity=RiskLevel.HIGH,
                    description="Suspicious creator/producer in metadata",
                    location="Document metadata",
                    evidence={'creator': creator, 'producer': producer},
                    recommendation="Verify document authenticity"
                ))
        
        return issues
    
    def _check_image_quality_issues(self, metadata: Dict[str, Any]) -> List[DocumentIssue]:
        """Check image-specific quality issues"""
        issues = []
        
        # Check OCR confidence
        avg_confidence = metadata.get('ocr_confidence_avg', 100)
        min_confidence = metadata.get('ocr_confidence_min', 100)
        
        if avg_confidence < 70:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.POOR_IMAGE_QUALITY,
                severity=RiskLevel.MEDIUM,
                description=f"Low OCR confidence: {avg_confidence:.1f}% average",
                location="Image quality",
                evidence={'avg_confidence': avg_confidence, 'min_confidence': min_confidence},
                recommendation="Improve image quality for better text recognition"
            ))
        
        # Check image dimensions
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        
        if width < 800 or height < 600:
            issues.append(DocumentIssue(
                issue_type=ValidationIssue.POOR_IMAGE_QUALITY,
                severity=RiskLevel.LOW,
                description=f"Low resolution image: {width}x{height}",
                location="Image specifications",
                evidence={'width': width, 'height': height},
                recommendation="Use higher resolution images for better analysis"
            ))
        
        return issues
    
    def _calculate_risk_score(self, issues: List[DocumentIssue]) -> float:
        """Calculate overall risk score based on issues found"""
        score = 0.0
        
        severity_weights = {
            RiskLevel.LOW: 5,
            RiskLevel.MEDIUM: 15,
            RiskLevel.HIGH: 30,
            RiskLevel.CRITICAL: 50
        }
        
        for issue in issues:
            score += severity_weights.get(issue.severity, 0)
        
        # Cap at 100
        return min(score, 100.0)
    
    def _generate_assessment(self, risk_score: float, issues: List[DocumentIssue]) -> str:
        """Generate overall assessment based on risk score and issues"""
        if risk_score >= 70:
            return "HIGH RISK - Multiple critical issues detected. Document requires immediate review."
        elif risk_score >= 40:
            return "MEDIUM RISK - Several issues found. Enhanced review recommended."
        elif risk_score >= 15:
            return "LOW RISK - Minor issues detected. Standard review sufficient."
        else:
            return "MINIMAL RISK - Document appears to be in good condition."
    
    def _generate_recommendations(self, issues: List[DocumentIssue]) -> List[str]:
        """Generate specific recommendations based on issues found"""
        recommendations = []
        
        # Group by issue type
        issue_types = {}
        for issue in issues:
            issue_type = issue.issue_type
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Generate type-specific recommendations
        if ValidationIssue.MISSING_SECTION in issue_types:
            recommendations.append("Request complete document with all required sections")
        
        if ValidationIssue.POOR_IMAGE_QUALITY in issue_types:
            recommendations.append("Request higher quality scan or original document")
        
        if ValidationIssue.SUSPICIOUS_METADATA in issue_types:
            recommendations.append("Conduct enhanced verification of document authenticity")
        
        if ValidationIssue.INCONSISTENT_DATA in issue_types:
            recommendations.append("Verify accuracy of financial amounts and dates")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Continue with standard processing")
        
        return recommendations

# Demo function
def demo_document_processing():
    """Demonstrate document processing functionality"""
    processor = DocumentProcessor()
    
    print("Document Processing System Demo")
    print("===============================")
    
    # This would normally process actual files
    print("\nProcessor initialized with support for:")
    for ext, doc_type in processor.supported_formats.items():
        print(f"  {ext}: {doc_type.value}")
    
    print(f"\nExpected document sections: {', '.join(processor.expected_sections)}")

if __name__ == "__main__":
    demo_document_processing()