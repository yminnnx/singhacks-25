"""
Image Analysis Engine for Document Corroboration
Handles image authenticity verification, AI-generated detection, and tampering analysis
"""

import os
import hashlib
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import base64

try:
    import cv2
    import numpy as np
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
except ImportError:
    print("Image analysis libraries not available. Install with: pip install opencv-python Pillow")

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
            
            # Reverse image search (optional, requires API)
            reverse_search = None  # Would implement with Google Vision API or similar
            
            # Calculate overall assessment
            overall_assessment, confidence_score = self._calculate_overall_assessment([
                metadata_analysis, pixel_analysis, ai_detection, tampering_analysis
            ])
            
            # Generate risk indicators and recommendations
            risk_indicators = self._extract_risk_indicators([
                metadata_analysis, pixel_analysis, ai_detection, tampering_analysis
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
        
        # Calculate noise statistics in different regions
        h, w = gray_image.shape
        regions = [
            noise[0:h//2, 0:w//2],      # Top-left
            noise[0:h//2, w//2:w],      # Top-right
            noise[h//2:h, 0:w//2],      # Bottom-left
            noise[h//2:h, w//2:w]       # Bottom-right
        ]
        
        noise_stats = [np.std(region) for region in regions]
        noise_variance = np.var(noise_stats)
        
        return {
            'region_noise_std': noise_stats,
            'noise_variance': float(noise_variance),
            'inconsistent_noise': noise_variance > 100  # Threshold for inconsistency
        }
    
    def _analyze_color_distribution(self, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution for unnaturalness"""
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Look for unnatural spikes or gaps
        h_peaks = len([i for i in range(1, 179) if h_hist[i] > h_hist[i-1] and h_hist[i] > h_hist[i+1]])
        s_variance = np.var(s_hist)
        v_variance = np.var(v_hist)
        
        return {
            'hue_peaks': int(h_peaks),
            'saturation_variance': float(s_variance),
            'value_variance': float(v_variance),
            'unnatural_distribution': h_peaks > 20 or s_variance > 1000000
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
    
    def _calculate_overall_assessment(self, analyses: List[ImageAnalysisResult]) -> Tuple[AuthenticityResult, float]:
        """Calculate overall assessment based on individual analyses"""
        # Weight different analysis types
        weights = {
            AnalysisType.METADATA_ANALYSIS: 0.2,
            AnalysisType.PIXEL_ANALYSIS: 0.25,
            AnalysisType.AI_DETECTION: 0.3,
            AnalysisType.TAMPERING_DETECTION: 0.25
        }
        
        weighted_confidence = 0.0
        critical_flags = 0
        high_risk_flags = 0
        
        for analysis in analyses:
            weight = weights.get(analysis.analysis_type, 0.2)
            weighted_confidence += analysis.confidence * weight
            
            if analysis.result in [AuthenticityResult.AI_GENERATED, AuthenticityResult.TAMPERED]:
                critical_flags += 1
            elif analysis.result == AuthenticityResult.LIKELY_FAKE:
                critical_flags += 1
            elif analysis.result == AuthenticityResult.SUSPICIOUS:
                high_risk_flags += 1
        
        # Determine overall result
        if critical_flags >= 2:
            overall_result = AuthenticityResult.LIKELY_FAKE
        elif critical_flags >= 1:
            overall_result = AuthenticityResult.SUSPICIOUS
        elif high_risk_flags >= 2:
            overall_result = AuthenticityResult.SUSPICIOUS
        else:
            overall_result = AuthenticityResult.AUTHENTIC
        
        return overall_result, weighted_confidence
    
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