"""
Image Analysis Engine for Document Corroboration
Handles image authenticity verification, AI-generated detection, and tampering analysis
"""

import os
import hashlib
import json
import logging
import requests
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import base64

# Add Groq integration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.env_config import Config
from groq import Groq

try:
    import cv2
    import numpy as np
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
except ImportError:
    print("Image analysis libraries not available. Install with: pip install opencv-python Pillow")
    # Create dummy numpy for basic functionality
    class DummyNumpy:
        @staticmethod
        def clip(value, min_val, max_val):
            return max(min_val, min(max_val, value))
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def var(arr):
            if not arr: return 0
            mean_val = sum(arr) / len(arr)
            return sum((x - mean_val) ** 2 for x in arr) / len(arr)
        @staticmethod
        def std(arr):
            return DummyNumpy.var(arr) ** 0.5
    np = DummyNumpy()

class AuthenticityResult(Enum):
    AUTHENTIC = "Authentic"
    SUSPICIOUS = "Suspicious"
    LIKELY_FAKE = "Likely Fake"
    AI_GENERATED = "AI Generated"
    TAMPERED = "Tampered"

class AnalysisType(Enum):
    METADATA_ANALYSIS = "Metadata Analysis"
    PIXEL_ANALYSIS = "Pixel Analysis"
    REVERSE_IMAGE_SEARCH = "Reverse Image Search"
    AI_DETECTION = "AI Detection"
    TAMPERING_DETECTION = "Tampering Detection"
    GROQ_AI_ANALYSIS = "Groq AI Analysis"  # New analysis type

@dataclass
class ImageAnalysisResult:
    analysis_type: AnalysisType
    confidence: float  # 0-100
    result: AuthenticityResult
    evidence: Dict[str, Any]
    description: str
    recommendations: List[str]

@dataclass
class ComprehensiveImageAnalysis:
    image_id: str
    file_path: str
    analysis_timestamp: datetime
    file_hash: str
    image_properties: Dict[str, Any]
    metadata_analysis: ImageAnalysisResult
    pixel_analysis: ImageAnalysisResult
    ai_detection_analysis: ImageAnalysisResult
    tampering_analysis: ImageAnalysisResult
    groq_ai_analysis: ImageAnalysisResult  # New Groq analysis
    reverse_search_analysis: Optional[ImageAnalysisResult]
    overall_assessment: AuthenticityResult
    confidence_score: float
    risk_indicators: List[str]
    recommendations: List[str]

class ImageAnalysisEngine:
    """
    Advanced image analysis engine for detecting fake, AI-generated, or tampered images
    """
    
    def __init__(self):
        self.setup_logging()
        
        # Initialize Groq client for AI analysis
        self.groq_client = None
        self.groq_enabled = False
        self._initialize_groq()
        
        # Known AI generator signatures
        self.ai_generators = [
            'midjourney', 'dalle', 'stable diffusion', 'firefly', 'playground ai',
            'artbreeder', 'runway', 'imagen', 'parti'
        ]
        
        # Suspicious metadata indicators
        self.suspicious_software = [
            'photoshop', 'gimp', 'paint.net', 'canva', 'figma', 'sketch'
        ]
        
        # Error level analysis parameters
        self.ela_params = {
            'quality': 90,
            'scale': 15
        }
    
    def setup_logging(self):
        """Setup logging for audit trail"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_groq(self):
        """Initialize Groq client for AI-powered image analysis"""
        try:
            Config.validate_config()
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            self.groq_enabled = True
            self.logger.info("✅ Groq AI integration enabled for enhanced image analysis")
        except Exception as e:
            self.logger.warning(f"⚠️ Groq AI not available: {e}")
            self.groq_enabled = False
    
    def _encode_image_for_groq(self, image_path: str) -> str:
        """Encode image to base64 for Groq analysis"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image: {e}")
            return None
    
    def analyze_image(self, image_path: str) -> ComprehensiveImageAnalysis:
        """Perform comprehensive image analysis"""
        try:
            # Generate image ID and hash
            image_id = self._generate_image_id(image_path)
            file_hash = self._calculate_file_hash(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            pil_image = Image.open(image_path)
            
            # Get basic image properties
            image_properties = self._extract_image_properties(pil_image)
            
            # Perform different types of analysis
            metadata_analysis = self._analyze_metadata(pil_image)
            pixel_analysis = self._analyze_pixel_patterns(image)
            ai_detection = self._detect_ai_generation(image, pil_image)
            tampering_analysis = self._detect_tampering(image)
            
            # NEW: Groq AI-powered analysis
            groq_ai_analysis = self._analyze_with_groq(image_path, pil_image)
            
            # Reverse image search (optional, requires API)
            reverse_search = None  # Would implement with Google Vision API or similar
            
            # Calculate overall assessment (now includes Groq analysis)
            overall_assessment, confidence_score = self._calculate_overall_assessment([
                metadata_analysis, pixel_analysis, ai_detection, tampering_analysis, groq_ai_analysis
            ])
            
            # Generate risk indicators and recommendations
            risk_indicators = self._extract_risk_indicators([
                metadata_analysis, pixel_analysis, ai_detection, tampering_analysis, groq_ai_analysis
            ])
            
            recommendations = self._generate_recommendations(overall_assessment, risk_indicators)
            
            result = ComprehensiveImageAnalysis(
                image_id=image_id,
                file_path=image_path,
                analysis_timestamp=datetime.now(),
                file_hash=file_hash,
                image_properties=image_properties,
                metadata_analysis=metadata_analysis,
                pixel_analysis=pixel_analysis,
                ai_detection_analysis=ai_detection,
                tampering_analysis=tampering_analysis,
                groq_ai_analysis=groq_ai_analysis,  # Include Groq analysis
                reverse_search_analysis=reverse_search,
                overall_assessment=overall_assessment,
                confidence_score=confidence_score,
                risk_indicators=risk_indicators,
                recommendations=recommendations
            )
            
            self.logger.info(f"Analyzed image {image_id}: {overall_assessment.value} (confidence: {confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            raise
    
    def _generate_image_id(self, image_path: str) -> str:
        """Generate unique image ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename_hash = hashlib.md5(os.path.basename(image_path).encode()).hexdigest()[:8]
        return f"IMG-{timestamp}-{filename_hash}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image properties"""
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'file_size': os.path.getsize(image.filename) if hasattr(image, 'filename') else 0
        }
    
    def _analyze_metadata(self, image: Image.Image) -> ImageAnalysisResult:
        """Analyze image metadata for authenticity indicators"""
        evidence = {}
        suspicious_indicators = []
        confidence = 50.0  # Default confidence
        
        try:
            # Extract EXIF data
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
            
            evidence['exif_data'] = exif_data
            
            # Check for camera information
            camera_info = self._extract_camera_info(exif_data)
            evidence['camera_info'] = camera_info
            
            # Check for software signatures
            software = exif_data.get('Software', '').lower()
            if any(sus_soft in software for sus_soft in self.suspicious_software):
                suspicious_indicators.append(f"Edited with: {software}")
                confidence += 20
            
            # Check for AI generator signatures
            if any(ai_gen in software for ai_gen in self.ai_generators):
                suspicious_indicators.append(f"AI generator detected: {software}")
                confidence += 40
                return ImageAnalysisResult(
                    analysis_type=AnalysisType.METADATA_ANALYSIS,
                    confidence=min(confidence, 95),
                    result=AuthenticityResult.AI_GENERATED,
                    evidence=evidence,
                    description=f"AI generation detected in metadata: {software}",
                    recommendations=["Flag for manual review", "Request original document"]
                )
            
            # Check for missing EXIF data (suspicious for photos)
            if not exif_data and image.format in ['JPEG', 'TIFF']:
                suspicious_indicators.append("Missing EXIF data")
                confidence += 15
            
            # Check timestamp consistency
            timestamp_issues = self._check_timestamp_consistency(exif_data)
            if timestamp_issues:
                suspicious_indicators.extend(timestamp_issues)
                confidence += 10
            
            evidence['suspicious_indicators'] = suspicious_indicators
            
            # Determine result
            if confidence >= 70:
                result = AuthenticityResult.SUSPICIOUS
            elif confidence >= 90:
                result = AuthenticityResult.LIKELY_FAKE
            else:
                result = AuthenticityResult.AUTHENTIC
            
            description = f"Metadata analysis complete. {len(suspicious_indicators)} indicators found."
            recommendations = self._get_metadata_recommendations(suspicious_indicators)
            
        except Exception as e:
            self.logger.warning(f"Metadata analysis failed: {e}")
            confidence = 30
            result = AuthenticityResult.SUSPICIOUS
            description = f"Metadata analysis failed: {e}"
            recommendations = ["Manual verification required"]
            evidence['error'] = str(e)
        
        return ImageAnalysisResult(
            analysis_type=AnalysisType.METADATA_ANALYSIS,
            confidence=confidence,
            result=result,
            evidence=evidence,
            description=description,
            recommendations=recommendations
        )
    
    def _extract_camera_info(self, exif_data: Dict[str, str]) -> Dict[str, str]:
        """Extract camera-related information from EXIF"""
        camera_info = {}
        
        camera_fields = {
            'Make': 'camera_make',
            'Model': 'camera_model',
            'DateTime': 'datetime',
            'DateTimeOriginal': 'datetime_original',
            'DateTimeDigitized': 'datetime_digitized',
            'GPS GPSLatitude': 'gps_latitude',
            'GPS GPSLongitude': 'gps_longitude'
        }
        
        for exif_key, info_key in camera_fields.items():
            if exif_key in exif_data:
                camera_info[info_key] = exif_data[exif_key]
        
        return camera_info
    
    def _check_timestamp_consistency(self, exif_data: Dict[str, str]) -> List[str]:
        """Check for timestamp inconsistencies"""
        issues = []
        
        datetime_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        timestamps = {}
        
        for field in datetime_fields:
            if field in exif_data:
                try:
                    timestamp = datetime.strptime(exif_data[field], '%Y:%m:%d %H:%M:%S')
                    timestamps[field] = timestamp
                except ValueError:
                    issues.append(f"Invalid timestamp format in {field}")
        
        # Check if timestamps are logical
        if len(timestamps) >= 2:
            times = list(timestamps.values())
            if max(times) - min(times) > timedelta(hours=24):
                issues.append("Timestamps differ by more than 24 hours")
        
        return issues
    
    def _analyze_pixel_patterns(self, image: np.ndarray) -> ImageAnalysisResult:
        """Analyze pixel patterns for signs of manipulation"""
        evidence = {}
        suspicious_indicators = []
        confidence = 50.0
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze noise patterns
            noise_analysis = self._analyze_noise_patterns(gray)
            evidence['noise_analysis'] = noise_analysis
            
            if noise_analysis['inconsistent_noise']:
                suspicious_indicators.append("Inconsistent noise patterns detected")
                confidence += 15
            
            # Analyze color distribution
            color_analysis = self._analyze_color_distribution(hsv)
            evidence['color_analysis'] = color_analysis
            
            if color_analysis['unnatural_distribution']:
                suspicious_indicators.append("Unnatural color distribution")
                confidence += 10
            
            # Edge consistency analysis
            edge_analysis = self._analyze_edge_consistency(gray)
            evidence['edge_analysis'] = edge_analysis
            
            if edge_analysis['inconsistent_edges']:
                suspicious_indicators.append("Inconsistent edge patterns")
                confidence += 20
            
            # JPEG compression analysis
            compression_analysis = self._analyze_compression_artifacts(image)
            evidence['compression_analysis'] = compression_analysis
            
            if compression_analysis['multiple_compression']:
                suspicious_indicators.append("Multiple compression artifacts")
                confidence += 25
            
            evidence['suspicious_indicators'] = suspicious_indicators
            
            # Determine result
            if confidence >= 80:
                result = AuthenticityResult.LIKELY_FAKE
            elif confidence >= 65:
                result = AuthenticityResult.SUSPICIOUS
            else:
                result = AuthenticityResult.AUTHENTIC
            
            description = f"Pixel analysis complete. {len(suspicious_indicators)} anomalies detected."
            recommendations = self._get_pixel_analysis_recommendations(suspicious_indicators)
            
        except Exception as e:
            self.logger.warning(f"Pixel analysis failed: {e}")
            confidence = 30
            result = AuthenticityResult.SUSPICIOUS
            description = f"Pixel analysis failed: {e}"
            recommendations = ["Technical analysis required"]
            evidence['error'] = str(e)
        
        return ImageAnalysisResult(
            analysis_type=AnalysisType.PIXEL_ANALYSIS,
            confidence=confidence,
            result=result,
            evidence=evidence,
            description=description,
            recommendations=recommendations
        )
    
    def _analyze_noise_patterns(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns in the image"""
        # Apply Gaussian blur and subtract to get noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise = cv2.absdiff(gray_image, blurred)

        h, w = gray_image.shape
        regions = [
            noise[0:h//2, 0:w//2],
            noise[0:h//2, w//2:w],
            noise[h//2:h, 0:w//2],
            noise[h//2:h, w//2:w]
        ]

        noise_stats = [np.std(region) for region in regions]
        noise_mean = np.mean(noise_stats)
        noise_variance = np.var(noise_stats)
    
        # Dynamic threshold: depends on image brightness
        brightness = np.mean(gray_image)
        adaptive_thresh = 0.02 * brightness + 50
    
        inconsistent = noise_variance > adaptive_thresh
    
        return {
            'region_noise_std': noise_stats,
            'noise_mean': float(noise_mean),
            'noise_variance': float(noise_variance),
            'inconsistent_noise': inconsistent
        }
    
    def _analyze_color_distribution(self, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution for unnaturalness"""
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Look for unnatural spikes or gaps
        s_var = np.var(s_hist)
        v_var = np.var(v_hist)
        color_balance = np.mean(h_hist) / (np.mean(s_hist) + 1e-6)

        unnatural = (s_var / (v_var + 1e-6) > 2.5) or (color_balance < 0.5 or color_balance > 1.5)
    
        
        return {
            'color_balance_ratio' : float(color_balance),
            'saturation_variance': float(s_var),
            'value_variance': float(v_var),
            'unnatural_distribution': unnatural
        }
    
    
    def _analyze_edge_consistency(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency across the image"""
        # Detect edges using Canny
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Analyze edge density in different regions
        h, w = gray_image.shape
        regions = [
            edges[0:h//2, 0:w//2],
            edges[0:h//2, w//2:w],
            edges[h//2:h, 0:w//2],
            edges[h//2:h, w//2:w]
        ]
        
        edge_densities = [np.sum(region > 0) / region.size for region in regions]
        edge_variance = np.var(edge_densities)
        
        return {
            'edge_densities': edge_densities,
            'edge_variance': float(edge_variance),
            'inconsistent_edges': edge_variance > 0.01
        }
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze JPEG compression artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for 8x8 block artifacts (JPEG compression signature)
        block_artifacts = self._detect_block_artifacts(gray)
        
        # Analyze DCT coefficients if available
        dct_analysis = self._analyze_dct_patterns(gray)
        
        return {
            'block_artifacts_detected': block_artifacts,
            'dct_irregularities': dct_analysis,
            'multiple_compression': block_artifacts and dct_analysis
        }
    
    def _detect_block_artifacts(self, gray_image: np.ndarray) -> bool:
        """Detect 8x8 block artifacts typical of JPEG compression"""
        # This is a simplified implementation
        # In practice, would use more sophisticated DCT analysis
        h, w = gray_image.shape
        
        # Check for grid patterns at 8-pixel intervals
        vertical_diffs = []
        horizontal_diffs = []
        
        for i in range(8, min(h, 200), 8):
            diff = np.mean(np.abs(gray_image[i] - gray_image[i-1]))
            vertical_diffs.append(diff)
        
        for j in range(8, min(w, 200), 8):
            diff = np.mean(np.abs(gray_image[:, j] - gray_image[:, j-1]))
            horizontal_diffs.append(diff)
        
        # If differences at 8-pixel intervals are consistently higher, suspect JPEG artifacts
        avg_vertical = np.mean(vertical_diffs) if vertical_diffs else 0
        avg_horizontal = np.mean(horizontal_diffs) if horizontal_diffs else 0
        
        return avg_vertical > 5 or avg_horizontal > 5
    
    def _analyze_dct_patterns(self, gray_image: np.ndarray) -> bool:
        """Analyze DCT patterns for compression irregularities"""
        # Simplified DCT analysis - would be more complex in production
        try:
            # Convert to float for DCT
            float_image = np.float32(gray_image)
            
            # Apply DCT to 8x8 blocks
            h, w = float_image.shape
            dct_variances = []
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = float_image[i:i+8, j:j+8]
                    dct_block = cv2.dct(block)
                    # Analyze high-frequency components
                    high_freq = dct_block[4:, 4:]
                    dct_variances.append(np.var(high_freq))
            
            # Look for unusual variance patterns
            variance_of_variances = np.var(dct_variances)
            
            return variance_of_variances > 1000  # Threshold for irregularity
            
        except Exception:
            return False
    
    def _detect_ai_generation(self, image: np.ndarray, pil_image: Image.Image) -> ImageAnalysisResult:
        """Detect signs of AI generation"""
        evidence = {}
        suspicious_indicators = []
        confidence = 50.0
        
        try:
            # Check for AI-typical artifacts
            ai_artifacts = self._check_ai_artifacts(image)
            evidence['ai_artifacts'] = ai_artifacts
            
            if ai_artifacts['high_frequency_noise']:
                suspicious_indicators.append("AI-typical high-frequency noise")
                confidence += 20
            
            if ai_artifacts['unnatural_textures']:
                suspicious_indicators.append("Unnatural texture patterns")
                confidence += 25
            
            # Check for perfect symmetries (AI often creates too-perfect images)
            symmetry_analysis = self._analyze_symmetry(image)
            evidence['symmetry_analysis'] = symmetry_analysis
            
            if symmetry_analysis['too_perfect']:
                suspicious_indicators.append("Unnaturally perfect symmetry")
                confidence += 15
            
            # Check for impossible lighting
            lighting_analysis = self._analyze_lighting_consistency(image)
            evidence['lighting_analysis'] = lighting_analysis
            
            if lighting_analysis['inconsistent_lighting']:
                suspicious_indicators.append("Inconsistent lighting patterns")
                confidence += 20
            
            evidence['suspicious_indicators'] = suspicious_indicators
            
            # Determine result
            if confidence >= 85:
                result = AuthenticityResult.AI_GENERATED
            elif confidence >= 70:
                result = AuthenticityResult.SUSPICIOUS
            else:
                result = AuthenticityResult.AUTHENTIC
            
            description = f"AI detection analysis complete. {len(suspicious_indicators)} AI indicators found."
            recommendations = self._get_ai_detection_recommendations(result)
            
        except Exception as e:
            self.logger.warning(f"AI detection failed: {e}")
            confidence = 30
            result = AuthenticityResult.SUSPICIOUS
            description = f"AI detection failed: {e}"
            recommendations = ["Manual verification required"]
            evidence['error'] = str(e)
        
        return ImageAnalysisResult(
            analysis_type=AnalysisType.AI_DETECTION,
            confidence=confidence,
            result=result,
            evidence=evidence,
            description=description,
            recommendations=recommendations
        )
    
    def _check_ai_artifacts(self, image: np.ndarray) -> Dict[str, bool]:
        """Check for AI-specific artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # High-frequency noise analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_variance = np.var(laplacian)
        
        # Texture analysis using Gabor filters
        texture_responses = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_responses.append(np.std(filtered))
        
        texture_variance = np.var(texture_responses)
        
        return {
            'high_frequency_noise': high_freq_variance > 1000,
            'unnatural_textures': texture_variance > 500
        }
    
    def _analyze_symmetry(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image symmetry for unnaturalness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check horizontal symmetry
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        horizontal_correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]
        
        return {
            'horizontal_correlation': float(horizontal_correlation),
            'too_perfect': horizontal_correlation > 0.95
        }
    
    def _analyze_lighting_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting consistency across the image"""
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Analyze gradient consistency
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Check for inconsistent lighting directions
        direction_variance = np.var(gradient_direction[gradient_magnitude > np.mean(gradient_magnitude)])
        
        return {
            'gradient_variance': float(direction_variance),
            'inconsistent_lighting': direction_variance > 2.0
        }
    
    def _detect_tampering(self, image: np.ndarray) -> ImageAnalysisResult:
        """Detect signs of image tampering"""
        evidence = {}
        suspicious_indicators = []
        confidence = 50.0
        
        try:
            # Error Level Analysis (ELA)
            ela_result = self._perform_ela(image)
            evidence['ela_analysis'] = ela_result
            
            if ela_result['tampering_detected']:
                suspicious_indicators.append("ELA indicates potential tampering")
                confidence += 30
            
            # Copy-move detection
            copy_move_result = self._detect_copy_move(image)
            evidence['copy_move_analysis'] = copy_move_result
            
            if copy_move_result['duplicated_regions']:
                suspicious_indicators.append("Duplicated regions detected")
                confidence += 25
            
            # Splicing detection
            splicing_result = self._detect_splicing(image)
            evidence['splicing_analysis'] = splicing_result
            
            if splicing_result['splicing_detected']:
                suspicious_indicators.append("Image splicing detected")
                confidence += 35
            
            evidence['suspicious_indicators'] = suspicious_indicators
            
            # Determine result
            if confidence >= 80:
                result = AuthenticityResult.TAMPERED
            elif confidence >= 65:
                result = AuthenticityResult.SUSPICIOUS
            else:
                result = AuthenticityResult.AUTHENTIC
            
            description = f"Tampering analysis complete. {len(suspicious_indicators)} tampering indicators found."
            recommendations = self._get_tampering_recommendations(result)
            
        except Exception as e:
            self.logger.warning(f"Tampering detection failed: {e}")
            confidence = 30
            result = AuthenticityResult.SUSPICIOUS
            description = f"Tampering detection failed: {e}"
            recommendations = ["Technical analysis required"]
            evidence['error'] = str(e)
        
        return ImageAnalysisResult(
            analysis_type=AnalysisType.TAMPERING_DETECTION,
            confidence=confidence,
            result=result,
            evidence=evidence,
            description=description,
            recommendations=recommendations
        )
    
    def _perform_ela(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform Error Level Analysis"""
        # Simplified ELA implementation
        # Save image at specific quality, reload, and compare
        temp_path = "/tmp/ela_temp.jpg"
        
        try:
            cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, self.ela_params['quality']])
            recompressed = cv2.imread(temp_path)
            
            # Calculate difference
            diff = cv2.absdiff(image, recompressed)
            diff_enhanced = cv2.multiply(diff, self.ela_params['scale'])
            
            # Analyze the difference
            gray_diff = cv2.cvtColor(diff_enhanced, cv2.COLOR_BGR2GRAY)
            mean_diff = np.mean(gray_diff)
            std_diff = np.std(gray_diff)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'mean_error_level': float(mean_diff),
                'std_error_level': float(std_diff),
                'tampering_detected': mean_diff > 15 or std_diff > 20
            }
            
        except Exception as e:
            self.logger.warning(f"ELA failed: {e}")
            return {'tampering_detected': False, 'error': str(e)}
    
    def _detect_copy_move(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect copy-move forgery"""
        # Simplified copy-move detection using feature matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            # Use SIFT to detect features
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is None:
                return {'duplicated_regions': False, 'reason': 'No features detected'}
            
            # Use FLANN matcher to find similar features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            if len(descriptors) > 2:
                matches = flann.knnMatch(descriptors, descriptors, k=2)
                
                # Filter good matches (excluding self-matches)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                            # Check if keypoints are sufficiently far apart
                            pt1 = keypoints[m.queryIdx].pt
                            pt2 = keypoints[m.trainIdx].pt
                            distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                            if distance > 50:  # Minimum distance threshold
                                good_matches.append(m)
                
                return {
                    'duplicated_regions': len(good_matches) > 10,
                    'similar_feature_count': len(good_matches),
                    'total_features': len(keypoints)
                }
            else:
                return {'duplicated_regions': False, 'reason': 'Insufficient features'}
                
        except Exception as e:
            self.logger.warning(f"Copy-move detection failed: {e}")
            return {'duplicated_regions': False, 'error': str(e)}
    
    def _detect_splicing(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect image splicing"""
        # Simplified splicing detection using color and lighting analysis
        try:
            # Analyze color consistency across regions
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            h, w = lab.shape[:2]
            
            # Divide image into regions
            regions = [
                lab[0:h//2, 0:w//2],      # Top-left
                lab[0:h//2, w//2:w],      # Top-right
                lab[h//2:h, 0:w//2],      # Bottom-left
                lab[h//2:h, w//2:w]       # Bottom-right
            ]
            
            # Calculate color statistics for each region
            color_stats = []
            for region in regions:
                l_mean, a_mean, b_mean = np.mean(region, axis=(0, 1))
                l_std, a_std, b_std = np.std(region, axis=(0, 1))
                color_stats.append([l_mean, a_mean, b_mean, l_std, a_std, b_std])
            
            # Check for inconsistencies
            stats_array = np.array(color_stats)
            color_variance = np.var(stats_array, axis=0)
            
            # Threshold for detecting inconsistencies
            splicing_detected = np.any(color_variance > [100, 50, 50, 50, 25, 25])
            
            return {
                'color_variance': color_variance.tolist(),
                'splicing_detected': bool(splicing_detected),
                'region_count': len(regions)
            }
            
        except Exception as e:
            self.logger.warning(f"Splicing detection failed: {e}")
            return {'splicing_detected': False, 'error': str(e)}
    
    def _analyze_with_groq(self, image_path: str, pil_image: Image.Image) -> ImageAnalysisResult:
        """
        Analyze image using Groq's Vision API (Llama 4 Scout) for AI generation detection
        Now uses actual image vision capabilities instead of just technical analysis
        """
        evidence = {}
        suspicious_indicators = []
        confidence = 50.0
        
        if not self.groq_enabled:
            return ImageAnalysisResult(
                analysis_type=AnalysisType.GROQ_AI_ANALYSIS,
                confidence=0.0,
                result=AuthenticityResult.AUTHENTIC,
                evidence={'error': 'Groq AI not available'},
                description="Groq AI analysis not available",
                recommendations=["Use traditional analysis methods"]
            )
        
        try:
            # Extract basic image properties
            props = self._extract_image_properties(pil_image)
            file_stats = os.stat(image_path)
            file_size = file_stats.st_size
            
            # Convert image to base64 for Groq Vision API
            image_base64 = self._encode_image_for_groq(image_path)
            if not image_base64:
                raise Exception("Failed to encode image for Groq analysis")
            
            # Create data URL for Groq Vision API
            image_data_url = f"data:image/jpeg;base64,{image_base64}"
            
            # Get basic technical analysis for context
            import cv2
            cv_image = cv2.imread(image_path)
            technical_data = self._extract_technical_features(cv_image, pil_image, {})
            evidence['technical_analysis'] = technical_data
            
            # Create specialized AI detection prompt for Groq Vision
            vision_prompt = f"""You are an expert AI forensics investigator specializing in detecting AI-generated images and deepfakes for financial document verification. Analyze this image carefully for signs of artificial generation.

DOCUMENT CONTEXT:
- File Format: {props.get('format', 'Unknown')}
- Dimensions: {props.get('width', 0)}x{props.get('height', 0)} pixels
- File Size: {file_size} bytes

DETECTION FOCUS AREAS:
1. **AI Generation Artifacts**: Look for telltale signs of AI generation (unnatural textures, impossible geometry, perfect symmetries, artificial patterns)
2. **Document Authenticity**: Assess if this appears to be a genuine scanned/photographed document vs digitally created
3. **Text Quality**: Examine text rendering, font consistency, and character quality
4. **Visual Inconsistencies**: Check for lighting anomalies, shadow inconsistencies, or impossible perspectives
5. **Background Analysis**: Evaluate background patterns and textures for artificial generation
6. **Object Realism**: Assess if objects, text, and elements appear photographically realistic

Pay special attention to:
- Unnatural perfection in text or graphics
- Inconsistent lighting or shadows
- Impossible or perfect geometric patterns
- Text that looks too clean or artificially rendered
- Backgrounds that appear artificially generated
- Any visual elements that seem "too perfect" for a real document

Provide your analysis in this EXACT format:
AI_GENERATION_VERDICT: [AUTHENTIC_DOCUMENT/SUSPICIOUS_PATTERNS/LIKELY_AI_GENERATED/DEFINITELY_AI_GENERATED]
CONFIDENCE_PERCENTAGE: [number from 0-100]
PRIMARY_AI_INDICATORS: [list up to 3 main AI generation signs, separated by semicolons]
DOCUMENT_TYPE_ASSESSMENT: [appears to be genuine scanned document / digitally created document / AI-generated content]
RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
COMPLIANCE_ACTION: [specific recommendation for AML compliance team]
DETAILED_ANALYSIS: [explain what you see that indicates AI generation or authenticity]

Focus on visual analysis of the actual image content, not technical metadata."""

            try:
                # Use Groq's Vision API with Llama 4 Scout
                response = self.groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq Vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": vision_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_completion_tokens=800,
                    top_p=0.9
                )
                
                groq_vision_analysis = response.choices[0].message.content
                evidence['groq_vision_analysis'] = groq_vision_analysis
                
                # Parse Groq Vision response
                parsed_results = self._parse_groq_vision_response(groq_vision_analysis)
                evidence.update(parsed_results)
                
                # Extract confidence and indicators from vision analysis
                base_confidence = parsed_results.get('confidence', 50.0)
                confidence_adjustments = 0
                
                # Add technical analysis findings to the vision analysis
                if technical_data.get('noise_variance', 0) > 200:
                    confidence_adjustments += 5  # Lower weight since vision is primary
                    suspicious_indicators.append("High noise variance detected")
                
                if technical_data.get('edge_consistency', 1.0) < 0.3:
                    confidence_adjustments += 8
                    suspicious_indicators.append("Poor edge consistency")
                
                # Extract Groq's AI indicators
                if parsed_results.get('primary_ai_indicators'):
                    suspicious_indicators.extend(parsed_results['primary_ai_indicators'])
                
                # Calculate final confidence (vision analysis takes precedence)
                confidence = min(base_confidence + confidence_adjustments, 95.0)
                
                # Determine result based on Groq Vision analysis
                ai_verdict = parsed_results.get('ai_generation_verdict', 'SUSPICIOUS_PATTERNS').upper()
                
                if ai_verdict == 'DEFINITELY_AI_GENERATED':
                    result = AuthenticityResult.AI_GENERATED
                elif ai_verdict == 'LIKELY_AI_GENERATED':
                    result = AuthenticityResult.AI_GENERATED
                elif ai_verdict == 'SUSPICIOUS_PATTERNS' or confidence > 70:
                    result = AuthenticityResult.SUSPICIOUS
                elif confidence > 60:
                    result = AuthenticityResult.SUSPICIOUS  
                else:
                    result = AuthenticityResult.AUTHENTIC
                
                description = f"Groq Vision AI analysis: {parsed_results.get('detailed_analysis', 'Visual analysis complete')[:200]}..."
                recommendations = [parsed_results.get('compliance_action', 'Follow standard verification procedures')]
                
                self.logger.info(f"✅ Groq Vision analysis complete: {result.value} (confidence: {confidence:.1f}%)")
                
            except Exception as api_error:
                self.logger.warning(f"Groq Vision API call failed: {api_error}")
                # Fallback to technical analysis only
                confidence = self._calculate_technical_confidence(technical_data, {})
                result = AuthenticityResult.SUSPICIOUS if confidence > 60 else AuthenticityResult.AUTHENTIC
                description = f"Vision analysis failed, using technical analysis: {confidence:.1f}% confidence"
                recommendations = ["Manual verification recommended", "Vision analysis unavailable"]
                evidence['api_error'] = str(api_error)
                suspicious_indicators = ["Groq Vision analysis failed - using technical analysis only"]
        
        except Exception as e:
            self.logger.error(f"Groq Vision analysis error: {e}")
            confidence = 30.0
            result = AuthenticityResult.SUSPICIOUS
            description = f"Analysis failed: {e}"
            recommendations = ["Manual verification required"]
            evidence['error'] = str(e)
        
        evidence['suspicious_indicators'] = suspicious_indicators
        evidence['confidence_breakdown'] = {
            'base_confidence': base_confidence if 'base_confidence' in locals() else 50.0,
            'technical_adjustments': confidence_adjustments if 'confidence_adjustments' in locals() else 0,
            'final_confidence': confidence
        }
        
        return ImageAnalysisResult(
            analysis_type=AnalysisType.GROQ_AI_ANALYSIS,
            confidence=confidence,
            result=result,
            evidence=evidence,
            description=description,
            recommendations=recommendations
        )
    
    def _parse_groq_response(self, groq_text: str) -> Dict[str, Any]:
        """Parse Groq AI response into structured data"""
        parsed = {
            'authenticity': 'SUSPICIOUS',
            'confidence': 50.0,
            'key_findings': [],
            'risk_indicators': [],
            'recommendation': 'Manual verification required',
            'explanation': groq_text[:500]  # Truncate for storage
        }
        
        try:
            lines = groq_text.split('\n')
            for line in lines:
                line = line.strip()
                
                if line.startswith('AUTHENTICITY:'):
                    parsed['authenticity'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        parsed['confidence'] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('KEY_FINDINGS:'):
                    findings = line.split(':', 1)[1].strip()
                    if findings:
                        parsed['key_findings'] = [f.strip() for f in findings.split(',')]
                elif line.startswith('RISK_INDICATORS:'):
                    indicators = line.split(':', 1)[1].strip()
                    if indicators:
                        parsed['risk_indicators'] = [i.strip() for i in indicators.split(',')]
                elif line.startswith('RECOMMENDATION:'):
                    parsed['recommendation'] = line.split(':', 1)[1].strip()
                elif line.startswith('EXPLANATION:'):
                    parsed['explanation'] = line.split(':', 1)[1].strip()
        
        except Exception as e:
            self.logger.warning(f"Failed to parse Groq response: {e}")
        
        return parsed
    
    def _extract_technical_features(self, cv_image, pil_image, exif_data):
        """Extract comprehensive technical features from the image"""
        features = {}
        
        try:
            if cv_image is not None:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Noise analysis
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = cv2.absdiff(gray, blurred)
                features['noise_variance'] = float(np.var(noise))
                
                # Edge consistency
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['edge_consistency'] = float(edge_density)
                
                # Color analysis
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                features['color_anomaly'] = float(np.var(s_hist) / (np.var(h_hist) + 1e-6))
                
                # Compression artifacts
                features['compression_artifacts'] = self._detect_compression_artifacts_simple(gray)
                
                # Symmetry analysis
                h, w = gray.shape
                left_half = gray[:, :w//2]
                right_half = cv2.flip(gray[:, w//2:], 1)
                min_width = min(left_half.shape[1], right_half.shape[1])
                if min_width > 0:
                    left_resized = left_half[:, :min_width]
                    right_resized = right_half[:, :min_width]
                    correlation = cv2.matchTemplate(left_resized, right_resized, cv2.TM_CCOEFF_NORMED)
                    features['symmetry_score'] = float(correlation[0, 0]) if correlation.size > 0 else 0.0
                else:
                    features['symmetry_score'] = 0.0
            
            # Timestamp analysis
            features['timestamp_analysis'] = self._analyze_timestamps(exif_data)
            
        except Exception as e:
            self.logger.warning(f"Technical feature extraction failed: {e}")
            features = {
                'noise_variance': 0,
                'edge_consistency': 0.5,
                'color_anomaly': 1.0,
                'compression_artifacts': False,
                'symmetry_score': 0.5,
                'timestamp_analysis': 'Unknown'
            }
        
        return features
    
    def _detect_compression_artifacts_simple(self, gray_image):
        """Simple compression artifact detection"""
        try:
            # Look for 8x8 block boundaries (JPEG signature)
            h, w = gray_image.shape
            block_diffs = []
            
            for i in range(8, min(h, 200), 8):
                diff = np.mean(np.abs(gray_image[i] - gray_image[i-1]))
                block_diffs.append(diff)
            
            return len(block_diffs) > 0 and np.mean(block_diffs) > 5
        except:
            return False
    
    def _analyze_timestamps(self, exif_data):
        """Analyze timestamp consistency in EXIF data"""
        try:
            timestamps = []
            for field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                if field in exif_data:
                    try:
                        ts = datetime.strptime(exif_data[field], '%Y:%m:%d %H:%M:%S')
                        timestamps.append(ts)
                    except:
                        pass
            
            if len(timestamps) >= 2:
                time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() for i in range(1, len(timestamps))]
                if any(abs(diff) > 86400 for diff in time_diffs):  # More than 24 hours
                    return "Inconsistent (>24h difference)"
                else:
                    return "Consistent"
            elif len(timestamps) == 1:
                return "Single timestamp"
            else:
                return "No timestamps"
        except:
            return "Analysis failed"
    
    def _format_exif_for_analysis(self, exif_data):
        """Format EXIF data for Groq analysis"""
        if not exif_data:
            return "No EXIF metadata found (SUSPICIOUS for camera photos)"
        
        important_fields = ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal', 
                          'DateTimeDigitized', 'ImageWidth', 'ImageLength', 'Orientation']
        
        formatted = []
        for field in important_fields:
            if field in exif_data:
                formatted.append(f"- {field}: {exif_data[field]}")
        
        if not formatted:
            return "EXIF present but missing camera/device information (SUSPICIOUS)"
        
        return "\n".join(formatted)
    
    def _calculate_technical_confidence(self, technical_data, exif_data):
        """Calculate confidence based on technical analysis alone"""
        confidence = 40.0  # Base confidence when Groq fails
        
        # Adjust based on technical findings (more conservative)
        if technical_data.get('noise_variance', 0) > 300:  # Higher threshold
            confidence += 15
        
        if technical_data.get('edge_consistency', 1.0) < 0.2:  # Lower threshold
            confidence += 20
        
        if not exif_data:
            confidence += 20  # Reduced penalty
        
        if any(editor in exif_data.get('Software', '').lower() for editor in ['photoshop', 'gimp']):
            confidence += 30
        
        return min(confidence, 85.0)  # Lower cap
    
    def _encode_image_for_groq(self, image_path: str) -> str:
        """
        Encode image to base64 for Groq Vision API
        """
        try:
            import base64
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            self.logger.error(f"Failed to encode image for Groq: {e}")
            return None
    
    def _parse_groq_vision_response(self, response_text: str) -> dict:
        """
        Parse Groq Vision API response for AI generation analysis
        """
        parsed = {}
        
        try:
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('AI_GENERATION_VERDICT:'):
                    verdict = line.split(':', 1)[1].strip()
                    parsed['ai_generation_verdict'] = verdict
                
                elif line.startswith('CONFIDENCE_PERCENTAGE:'):
                    try:
                        confidence_str = line.split(':', 1)[1].strip()
                        confidence_num = ''.join(filter(str.isdigit, confidence_str))
                        if confidence_num:
                            parsed['confidence'] = float(confidence_num)
                    except:
                        parsed['confidence'] = 50.0
                
                elif line.startswith('PRIMARY_AI_INDICATORS:'):
                    indicators_str = line.split(':', 1)[1].strip()
                    if indicators_str and indicators_str != 'None':
                        indicators = [ind.strip() for ind in indicators_str.split(';') if ind.strip()]
                        parsed['primary_ai_indicators'] = indicators
                
                elif line.startswith('DOCUMENT_TYPE_ASSESSMENT:'):
                    assessment = line.split(':', 1)[1].strip()
                    parsed['document_type_assessment'] = assessment
                
                elif line.startswith('RISK_LEVEL:'):
                    risk = line.split(':', 1)[1].strip()
                    parsed['risk_level'] = risk
                
                elif line.startswith('COMPLIANCE_ACTION:'):
                    action = line.split(':', 1)[1].strip()
                    parsed['compliance_action'] = action
                
                elif line.startswith('DETAILED_ANALYSIS:'):
                    analysis = line.split(':', 1)[1].strip()
                    parsed['detailed_analysis'] = analysis
            
            # Set defaults if not found
            if 'confidence' not in parsed:
                parsed['confidence'] = 60.0
            if 'ai_generation_verdict' not in parsed:
                parsed['ai_generation_verdict'] = 'SUSPICIOUS_PATTERNS'
            if 'detailed_analysis' not in parsed:
                parsed['detailed_analysis'] = 'Vision analysis completed with standard assessment'
            if 'compliance_action' not in parsed:
                parsed['compliance_action'] = 'Manual verification recommended'
                
        except Exception as e:
            self.logger.warning(f"Error parsing Groq vision response: {e}")
            parsed = {
                'confidence': 50.0,
                'ai_generation_verdict': 'SUSPICIOUS_PATTERNS',
                'detailed_analysis': 'Response parsing failed - manual review required',
                'compliance_action': 'Manual verification required'
            }
        
        return parsed
    
    def _parse_groq_forensic_response(self, groq_text: str) -> Dict[str, Any]:
        """Parse Groq forensic response into structured data"""
        parsed = {
            'authenticity': 'SUSPICIOUS',
            'confidence': 50.0,
            'primary_concerns': [],
            'technical_indicators': '',
            'risk_assessment': 'MEDIUM',
            'recommended_action': 'Manual verification required',
            'forensic_reasoning': groq_text[:300]  # Truncate for storage
        }
        
        try:
            lines = groq_text.split('\n')
            for line in lines:
                line = line.strip()
                
                if line.startswith('AUTHENTICITY_VERDICT:'):
                    parsed['authenticity'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE_SCORE:'):
                    try:
                        parsed['confidence'] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('PRIMARY_CONCERNS:'):
                    concerns = line.split(':', 1)[1].strip()
                    if concerns and concerns != 'None':
                        parsed['primary_concerns'] = [c.strip() for c in concerns.split(';') if c.strip()]
                elif line.startswith('TECHNICAL_INDICATORS:'):
                    parsed['technical_indicators'] = line.split(':', 1)[1].strip()
                elif line.startswith('RISK_ASSESSMENT:'):
                    parsed['risk_assessment'] = line.split(':', 1)[1].strip()
                elif line.startswith('RECOMMENDED_ACTION:'):
                    parsed['recommended_action'] = line.split(':', 1)[1].strip()
                elif line.startswith('FORENSIC_REASONING:'):
                    parsed['forensic_reasoning'] = line.split(':', 1)[1].strip()
        
        except Exception as e:
            self.logger.warning(f"Failed to parse Groq forensic response: {e}")
        
        return parsed
    
    def _calculate_overall_assessment(self, analyses: List[ImageAnalysisResult]) -> Tuple[AuthenticityResult, float]:
        """Calculate overall assessment based on individual analyses"""
        # Weight different analysis types (including Groq)
        weights = {
            AnalysisType.METADATA_ANALYSIS: 0.15,
            AnalysisType.PIXEL_ANALYSIS: 0.20,
            AnalysisType.AI_DETECTION: 0.25,
            AnalysisType.TAMPERING_DETECTION: 0.20,
            AnalysisType.GROQ_AI_ANALYSIS: 0.20  # Give Groq significant weight
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            if analysis.analysis_type in weights:
                weight = weights[analysis.analysis_type]
                weighted_score += analysis.confidence * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        else:
            weighted_score = 50.0
        
        random_variance = random.uniform(-3, 3)  # add minor variability
        final_conf = np.clip(weighted_score + random_variance, 0, 100)
        
        # Special handling for AI detection results
        ai_results = [a for a in analyses if a.result == AuthenticityResult.AI_GENERATED]
        if ai_results and any(a.confidence > 70 for a in ai_results):
            return AuthenticityResult.AI_GENERATED, max(final_conf, 75)
        
        # Determine result more smoothly
        if final_conf > 85:
            overall = AuthenticityResult.LIKELY_FAKE
        elif final_conf > 70:
            overall = AuthenticityResult.SUSPICIOUS
        elif final_conf > 55:
            overall = AuthenticityResult.AUTHENTIC
        else:
            overall = AuthenticityResult.AUTHENTIC
        
        return overall, final_conf
    
    def _extract_risk_indicators(self, analyses: List[ImageAnalysisResult]) -> List[str]:
        """Extract all risk indicators from analyses"""
        risk_indicators = []
        for analysis in analyses:
            if 'suspicious_indicators' in analysis.evidence:
                risk_indicators.extend(analysis.evidence['suspicious_indicators'])
        return risk_indicators
    
    def _generate_recommendations(self, overall_assessment: AuthenticityResult, risk_indicators: List[str]) -> List[str]:
        """Generate recommendations based on overall assessment"""
        recommendations = []
        
        if overall_assessment == AuthenticityResult.LIKELY_FAKE:
            recommendations.extend([
                "REJECT DOCUMENT - High probability of forgery detected",
                "Request original physical document",
                "Escalate to fraud investigation team"
            ])
        elif overall_assessment == AuthenticityResult.AI_GENERATED:
            recommendations.extend([
                "REJECT DOCUMENT - AI-generated content detected",
                "Request authentic documentation",
                "Flag customer account for enhanced monitoring"
            ])
        elif overall_assessment == AuthenticityResult.TAMPERED:
            recommendations.extend([
                "REJECT DOCUMENT - Image tampering detected",
                "Request unmodified original document",
                "Consider manual verification"
            ])
        elif overall_assessment == AuthenticityResult.SUSPICIOUS:
            recommendations.extend([
                "ENHANCED REVIEW REQUIRED",
                "Manual verification recommended",
                "Request additional supporting documents"
            ])
        else:
            recommendations.append("Document appears authentic - proceed with standard verification")
        
        return recommendations
    
    def _get_metadata_recommendations(self, indicators: List[str]) -> List[str]:
        """Get recommendations based on metadata indicators"""
        recommendations = []
        if "AI generator detected" in str(indicators):
            recommendations.append("Flag as AI-generated content")
        if "Missing EXIF data" in str(indicators):
            recommendations.append("Request original unprocessed image")
        if "Edited with" in str(indicators):
            recommendations.append("Verify reason for image editing")
        return recommendations
    
    def _get_pixel_analysis_recommendations(self, indicators: List[str]) -> List[str]:
        """Get recommendations based on pixel analysis"""
        recommendations = []
        if "compression artifacts" in str(indicators):
            recommendations.append("Investigate compression history")
        if "noise patterns" in str(indicators):
            recommendations.append("Analyze image acquisition process")
        return recommendations
    
    def _get_ai_detection_recommendations(self, result: AuthenticityResult) -> List[str]:
        """Get recommendations based on AI detection"""
        if result == AuthenticityResult.AI_GENERATED:
            return ["Reject AI-generated content", "Request authentic documentation"]
        elif result == AuthenticityResult.SUSPICIOUS:
            return ["Enhanced verification required", "Consider manual review"]
        return ["Continue with standard processing"]
    
    def _get_tampering_recommendations(self, result: AuthenticityResult) -> List[str]:
        """Get recommendations based on tampering detection"""
        if result == AuthenticityResult.TAMPERED:
            return ["Reject tampered document", "Request original unmodified version"]
        elif result == AuthenticityResult.SUSPICIOUS:
            return ["Investigate potential tampering", "Manual verification recommended"]
        return ["No tampering detected"]

# Demo function
def demo_image_analysis():
    """Demonstrate image analysis functionality"""
    analyzer = ImageAnalysisEngine()
    
    print("Image Analysis Engine Demo")
    print("==========================")
    print("\nSupported analysis types:")
    for analysis_type in AnalysisType:
        print(f"  - {analysis_type.value}")
    
    print("\nAuthenticity results:")
    for result in AuthenticityResult:
        print(f"  - {result.value}")

if __name__ == "__main__":
    demo_image_analysis()