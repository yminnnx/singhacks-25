"""
Enhanced Image Analysis Module for Streamlit Frontend
Integrates Groq AI with traditional image forensics for AML document verification
"""

import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.part2_document_corroboration.image_analysis import ImageAnalysisEngine, AuthenticityResult

def show_groq_enhanced_image_analysis():
    """Enhanced image analysis page with Groq AI integration"""
    
    st.header("🤖 AI-Powered Document Image Analysis")
    st.subheader("Groq-Enhanced Authenticity Verification")
    
    # Show capabilities
    with st.expander("🔍 AI Analysis Capabilities", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Traditional Analysis:**
            - 📊 Metadata examination
            - 🔍 Pixel pattern analysis  
            - 📈 Compression artifact detection
            - ✂️ Tampering detection
            """)
        
        with col2:
            st.markdown("""
            **Groq AI Analysis:**
            - 🤖 AI generation detection
            - 📄 Document authenticity assessment
            - 💭 Natural language explanations
            - ⚖️ Compliance risk evaluation
            """)
    
    # Initialize analyzer
    @st.cache_resource
    def get_analyzer():
        return ImageAnalysisEngine()
    
    analyzer = get_analyzer()
    
    # Show Groq status
    if analyzer.groq_enabled:
        st.success("✅ Groq AI integration is active")
    else:
        st.warning("⚠️ Groq AI integration is disabled - using fallback analysis")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload document image for analysis",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a document image for comprehensive authenticity analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Uploaded Document")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"File: {uploaded_file.name}", use_column_width=True)
            
            # Show image properties
            st.markdown("**Image Properties:**")
            st.write(f"- **Format:** {image.format}")
            st.write(f"- **Size:** {image.width} × {image.height}")
            st.write(f"- **Mode:** {image.mode}")
            st.write(f"- **File Size:** {len(uploaded_file.getvalue())} bytes")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Analyze button
            if st.button("🚀 Run AI Analysis", type="primary"):
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Initializing analysis...")
                    progress_bar.progress(20)
                    
                    # Run analysis
                    status_text.text("🤖 Running AI-powered analysis...")
                    progress_bar.progress(60)
                    
                    result = analyzer.analyze_image(tmp_path)
                    
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    show_analysis_results(result)
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                
                finally:
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

def show_analysis_results(result):
    """Display comprehensive analysis results"""
    
    # Overall assessment
    st.markdown("### 🎯 Overall Assessment")
    
    # Color-code based on result
    result_colors = {
        AuthenticityResult.AUTHENTIC: "🟢",
        AuthenticityResult.SUSPICIOUS: "🟡", 
        AuthenticityResult.LIKELY_FAKE: "🔴",
        AuthenticityResult.AI_GENERATED: "🟠",
        AuthenticityResult.TAMPERED: "🔴"
    }
    
    result_color = result_colors.get(result.overall_assessment, "⚪")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Authenticity", f"{result_color} {result.overall_assessment.value}")
    with col2:
        st.metric("Confidence", f"{result.confidence_score:.1f}%")
    with col3:
        st.metric("Risk Level", "High" if result.confidence_score > 70 else "Medium" if result.confidence_score > 50 else "Low")
    
    # Individual analysis results
    st.markdown("### 📊 Detailed Analysis")
    
    analyses = [
        ("Metadata Analysis", result.metadata_analysis),
        ("Pixel Analysis", result.pixel_analysis),
        ("AI Detection", result.ai_detection_analysis),
        ("Tampering Detection", result.tampering_analysis),
        ("Groq AI Analysis", result.groq_ai_analysis)
    ]
    
    for name, analysis in analyses:
        with st.expander(f"{name} - {analysis.result.value} ({analysis.confidence:.1f}%)", expanded=(name == "Groq AI Analysis")):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {analysis.description}")
                
                # Show evidence
                if analysis.evidence:
                    if name == "Groq AI Analysis" and 'groq_raw_analysis' in analysis.evidence:
                        st.markdown("**🤖 Groq AI Insights:**")
                        groq_analysis = analysis.evidence['groq_raw_analysis']
                        st.text_area("AI Analysis", groq_analysis, height=150, disabled=True)
                    else:
                        st.markdown("**Technical Evidence:**")
                        # Show key evidence points
                        evidence_display = {}
                        for key, value in analysis.evidence.items():
                            if key != 'groq_raw_analysis' and not key.endswith('_error'):
                                if isinstance(value, (list, dict)):
                                    evidence_display[key] = str(value)[:100] + "..." if len(str(value)) > 100 else value
                                else:
                                    evidence_display[key] = value
                        
                        if evidence_display:
                            st.json(evidence_display)
            
            with col2:
                # Confidence gauge
                confidence_color = "🔴" if analysis.confidence > 70 else "🟡" if analysis.confidence > 50 else "🟢"
                st.metric("Confidence", f"{confidence_color} {analysis.confidence:.1f}%")
                
                # Recommendations
                if analysis.recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in analysis.recommendations:
                        st.write(f"• {rec}")
    
    # Risk indicators
    if result.risk_indicators:
        st.markdown("### ⚠️ Risk Indicators")
        for indicator in result.risk_indicators:
            st.warning(f"• {indicator}")
    
    # Final recommendations
    st.markdown("### 💡 Compliance Recommendations")
    for recommendation in result.recommendations:
        if "REJECT" in recommendation.upper():
            st.error(f"🚫 {recommendation}")
        elif "ENHANCED" in recommendation.upper() or "MANUAL" in recommendation.upper():
            st.warning(f"⚠️ {recommendation}")
        else:
            st.info(f"ℹ️ {recommendation}")

# Demo function for testing
def demo_streamlit_integration():
    """Demo the Streamlit integration"""
    st.set_page_config(
        page_title="Groq Image Analysis Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🏦 Julius Baer AML - Groq-Enhanced Image Analysis")
    
    show_groq_enhanced_image_analysis()

if __name__ == "__main__":
    demo_streamlit_integration()