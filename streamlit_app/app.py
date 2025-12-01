"""
Ridge MMM - Marketing Mix Modeling Tool
Home Page
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Ridge MMM - Marketing Mix Modeling",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📈 Ridge MMM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Marketing Mix Modeling Tool</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<h3>🎯 What is Marketing Mix Modeling (MMM)?</h3>

Marketing Mix Modeling은 기업이 마케팅 활동이 매출과 수익에 미치는 영향을 이해하는 데 도움을 주는 통계 분석 기법입니다. 과거 데이터를 분석하여 MMM은 다음을 수행할 수 있습니다:

- **정량화**: 각 마케팅 채널이 전체 수익에 기여하는 정도를 측정
- **최적화**: 채널 간 마케팅 예산 배분을 최적화
- **예측**: 향후 마케팅 전략의 영향을 예측
- **식별**: 마케팅 지출의 시너지 효과와 수익 체감 현상을 파악

이 도구는 **Ridge Regression**을 사용합니다. 이는 마케팅 채널 간의 다중공선성을 처리하고 안정적이고 해석 가능한 결과를 제공하는 강력한 머신러닝 기법입니다.
</div>
""", unsafe_allow_html=True)

# Key Features
st.markdown("### ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h4>📊 Data Upload & Validation</h4>
    <ul>
        <li>간편한 CSV 업로드</li>
        <li>자동 데이터 품질 검사</li>
        <li>결측값 탐지</li>
        <li>이상치 식별</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h4>🤖 Ridge Regression Modeling</h4>
    <ul>
        <li>다중공선성 처리</li>
        <li>안정성을 위한 정규화</li>
        <li>채널 기여도 분석</li>
        <li>ROI 계산</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h4>📈 Interactive Visualizations</h4>
    <ul>
        <li>채널 성과 차트</li>
        <li>기여도 분석</li>
        <li>예산 최적화</li>
        <li>시나리오 계획</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Sample Data Download
st.markdown("---")
st.markdown("### 📥 샘플 데이터로 시작하기")

st.info("""
👉 **MMM이 처음이신가요?** 샘플 데이터셋을 다운로드하여 도구의 기능을 탐색해보세요. 
샘플에는 여러 마케팅 채널이 포함된 2년간의 주간 데이터가 포함되어 있습니다.
""")

# Load sample data
sample_data_path = Path(__file__).parent.parent / "data" / "sample" / "sample_data.csv"

if sample_data_path.exists():
    try:
        sample_df = pd.read_csv(sample_data_path)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**샘플 데이터 미리보기:**")
            st.dataframe(sample_df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("**데이터셋 정보:**")
            st.metric("전체 주차", len(sample_df))
            st.metric("기간", f"{sample_df['date'].iloc[0]} ~ {sample_df['date'].iloc[-1]}")
            st.metric("마케팅 채널", 5)
            
            # Download button
            csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 샘플 데이터 다운로드",
                data=csv,
                file_name="sample_marketing_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"샘플 데이터 로딩 오류: {str(e)}")
else:
    st.warning("샘플 데이터 파일을 찾을 수 없습니다. data/sample/sample_data.csv 파일이 존재하는지 확인하세요.")

# Navigation Guide
st.markdown("---")
st.markdown("### 🗺️ 사용 가이드")

st.markdown("""
<div class="info-box">
<h4>도구 사용 방법:</h4>

1. **📊 Data Upload** (사이드바 → Data Upload)
   - 마케팅 데이터 CSV 파일 업로드
   - 필수 필드에 컬럼 매핑 (날짜, 수익, 미디어 채널)
   - 데이터 품질 검사 및 검증 결과 확인
   - 설정 저장

2. **⚙️ Model Configuration**
   - Adstock 및 Hill 변환 파라미터 설정
   - Ridge 회귀 알파 값 선택
   - 학습/테스트 데이터 분할 구성
   - 모델 학습 실행

3. **📈 Results & Analysis**
   - 모델 성능 지표 확인
   - 채널별 기여도 분석
   - 시계열 분해 시각화
   - 반응 곡선 및 최적화 권장사항

4. **💰 Budget Optimizer**
   - 최적 예산 배분 찾기
   - 사용자 정의 시나리오 생성 및 비교
   - 민감도 분석 수행
   - 수익 체감 지점 식별

5. **📄 Report**
   - 자동 생성된 요약 보고서
   - Excel/HTML 형식으로 다운로드
   - 주요 인사이트 및 권장사항

**👈 사이드바의 "Data Upload" 페이지로 이동하여 시작하세요!**
</div>
""", unsafe_allow_html=True)

# Data Format Requirements
with st.expander("📋 데이터 형식 요구사항"):
    st.markdown("""
    CSV 파일에는 다음 컬럼이 포함되어야 합니다:
    
    **필수:**
    - **날짜 컬럼**: 주간 또는 일간 날짜 (형식: YYYY-MM-DD)
    - **수익 컬럼**: 목표 변수 (매출, 수익, 전환 등)
    - **미디어 지출 컬럼**: 최소 하나의 마케팅 채널 (예: Google Ads, Facebook, TV)
    
    **선택사항:**
    - **외생 변수**: 계절성 지표, 프로모션, 공휴일 등
    
    **예시 구조:**
    ```
    date,revenue,google_uac,meta,apple_search,youtube,tiktok,promotion,seasonality
    2022-01-03,125000,15000,12000,8000,10000,5000,0,1
    2022-01-10,130000,16000,13000,8500,11000,5500,1,1
    ...
    ```
    
    **모범 사례:**
    - 신뢰할 수 있는 모델링을 위해 최소 52주의 데이터 사용
    - 일관된 시간 단위 사용 (주간 권장)
    - 모든 금액 값은 동일한 통화 단위 사용
    - 가능한 한 결측값 방지
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Streamlit • Ridge Regression • Python으로 제작</p>
    <p>Version 0.1.0 - 완전한 MMM 솔루션</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'config' not in st.session_state:
    st.session_state.config = {}

# Sidebar: Cache Management
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 System Tools")
    
    if st.button("🗑️ Clear Cache", use_container_width=True, help="Clear Streamlit cache and reload modules"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared!")
        st.info("💡 Please refresh the page (F5) to reload all modules")
        st.rerun()
