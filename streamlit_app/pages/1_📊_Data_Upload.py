"""
Ridge MMM - Data Upload Page
Upload and validate marketing data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processor import DataProcessor
from utils import get_numeric_columns, get_date_columns, validate_column_selection, get_column_info

# Page configuration
st.set_page_config(
    page_title="Data Upload - Ridge MMM",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .upload-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Data Upload & Validation")
st.markdown("마케팅 데이터를 업로드하고 분석을 위한 컬럼 매핑을 구성하세요")

# Initialize DataProcessor
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()

# File Upload Section
st.markdown("### 📁 데이터 업로드")

uploaded_file = st.file_uploader(
    "CSV 파일 선택 (최대 10MB)",
    type=['csv'],
    help="CSV 형식의 마케팅 데이터를 업로드하세요. 파일에는 날짜, 수익, 미디어 지출 컬럼이 포함되어야 합니다."
)

if uploaded_file is not None:
    # Check file size (10MB limit)
    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
    
    if file_size > 10:
        st.error(f"❌ 파일 크기({file_size:.2f} MB)가 10MB 제한을 초과합니다. 더 작은 파일을 업로드하세요.")
    else:
        try:
            # Load data
            with st.spinner("데이터 로딩 중..."):
                df = st.session_state.processor.load_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_uploaded = True
            
            st.success(f"✅ 파일 업로드 성공! ({file_size:.2f} MB)")
            
            # Data Preview Section
            st.markdown("---")
            st.markdown("### 👀 데이터 미리보기")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 행", len(df))
            with col2:
                st.metric("전체 컬럼", len(df.columns))
            with col3:
                st.metric("숫자 컬럼", len(get_numeric_columns(df)))
            
            # Show first few rows
            st.markdown("**처음 10개 행:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show data types
            with st.expander("📋 Column Information"):
                col_info = get_column_info(df)
                st.dataframe(col_info, use_container_width=True)
            
            # Column Mapping Section
            st.markdown("---")
            st.markdown("### 🗺️ 컬럼 매핑")
            st.info("모델링에 필요한 필드에 데이터 컬럼을 매핑하세요.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Date column selection
                st.markdown("#### 📅 날짜 컬럼")
                date_columns = get_date_columns(df)
                if not date_columns:
                    date_columns = df.columns.tolist()
                
                date_col = st.selectbox(
                    "날짜 컬럼 선택",
                    options=date_columns,
                    help="날짜가 포함된 컬럼 (주간 또는 일간)"
                )
                
                # Revenue column selection
                st.markdown("#### 💰 수익 컬럼")
                numeric_columns = get_numeric_columns(df)
                
                revenue_col = st.selectbox(
                    "수익/목표 컬럼 선택",
                    options=numeric_columns,
                    help="수익, 매출 또는 전환이 포함된 컬럼"
                )
            
            with col2:
                # Media columns selection
                st.markdown("#### 📺 미디어 지출 컬럼")
                available_media_cols = [col for col in numeric_columns if col != revenue_col]
                
                media_cols = st.multiselect(
                    "미디어 지출 컬럼 선택 (최소 1개 필요)",
                    options=available_media_cols,
                    help="마케팅 채널 지출 데이터가 포함된 컬럼"
                )
                
                # Exogenous variables selection
                st.markdown("#### 🔧 외생 변수 (선택사항)")
                available_exog_cols = [col for col in df.columns if col not in [date_col, revenue_col] + media_cols]
                
                exog_cols = st.multiselect(
                    "외생 변수 선택 (선택사항)",
                    options=available_exog_cols,
                    help="계절성, 프로모션, 공휴일 등의 추가 변수"
                )
            
            # Validate column selection
            if date_col and revenue_col and len(media_cols) > 0:
                validation_result = validate_column_selection(df, date_col, revenue_col, media_cols)
                
                if not validation_result['is_valid']:
                    st.error("❌ Column selection validation failed:")
                    for error in validation_result['errors']:
                        st.error(f"  • {error}")
                
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        st.warning(f"⚠️ {warning}")
            
            # Data Validation Section
            if date_col and revenue_col and len(media_cols) > 0:
                st.markdown("---")
                st.markdown("### ✅ Data Validation")
                
                # Validate data
                validation = st.session_state.processor.validate_data(
                    df, 
                    date_col=date_col,
                    revenue_col=revenue_col,
                    media_cols=media_cols
                )
                
                # Display validation results
                if validation['is_valid']:
                    st.success("✅ 데이터 검증 통과!")
                else:
                    st.error("❌ 데이터 검증 실패:")
                    for error in validation['errors']:
                        st.error(f"  • {error}")
                
                if validation['warnings']:
                    st.warning("⚠️ Warnings:")
                    for warning in validation['warnings']:
                        st.warning(f"  • {warning}")
                
                # Missing Values Check
                st.markdown("#### 🔍 Missing Values Analysis")
                missing_data = st.session_state.processor.check_missing_values(df)
                
                # Filter to show only columns with missing values
                missing_with_values = missing_data[missing_data['Missing_Count'] > 0]
                
                if len(missing_with_values) > 0:
                    st.warning(f"⚠️ {len(missing_with_values)}개 컬럼에서 결측값 발견")
                    st.dataframe(missing_with_values, use_container_width=True)
                else:
                    st.success("✅ 결측값이 감지되지 않았습니다!")
                
                # Outlier Detection
                st.markdown("#### 📊 Outlier Detection")
                
                columns_to_check = [revenue_col] + media_cols
                outliers = st.session_state.processor.detect_outliers(df, columns=columns_to_check)
                
                # Display outlier summary
                outlier_summary = []
                for col, info in outliers.items():
                    if info['outlier_count'] > 0:
                        outlier_summary.append({
                            'Column': col,
                            'Outlier_Count': info['outlier_count'],
                            'Lower_Bound': info['lower_bound'],
                            'Upper_Bound': info['upper_bound']
                        })
                
                if outlier_summary:
                    st.warning(f"⚠️ Found outliers in {len(outlier_summary)} column(s)")
                    outlier_df = pd.DataFrame(outlier_summary)
                    st.dataframe(outlier_df, use_container_width=True)
                    
                    # Show details in expander
                    with st.expander("📋 View Outlier Details"):
                        for col, info in outliers.items():
                            if info['outlier_count'] > 0:
                                st.markdown(f"**{col}:**")
                                st.write(f"  • Outlier count: {info['outlier_count']}")
                                st.write(f"  • Valid range: [{info['lower_bound']}, {info['upper_bound']}]")
                                st.write(f"  • Outlier values: {info['outlier_values'][:10]}{'...' if len(info['outlier_values']) > 10 else ''}")
                else:
                    st.success("✅ IQR 방법으로 이상치가 감지되지 않았습니다!")
                
                # Summary Statistics
                st.markdown("#### 📈 Summary Statistics")
                
                stats_columns = [revenue_col] + media_cols
                summary_stats = st.session_state.processor.get_summary_stats(df, columns=stats_columns)
                
                st.dataframe(summary_stats, use_container_width=True)
                
                # Date Range Information
                if date_col:
                    try:
                        start_date, end_date, num_periods = st.session_state.processor.get_date_range(df, date_col)
                        
                        st.markdown("#### 📅 Date Range")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Start Date", start_date)
                        with col2:
                            st.metric("End Date", end_date)
                        with col3:
                            st.metric("Total Periods", num_periods)
                    except Exception as e:
                        st.warning(f"⚠️ Could not parse date range: {str(e)}")
                
                # Save Configuration Button
                st.markdown("---")
                st.markdown("### 💾 Save Configuration")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info("컬럼 매핑을 저장하고 모델 구성으로 진행하세요.")
                
                with col2:
                    if st.button("💾 설정 저장", type="primary", use_container_width=True):
                        # Save configuration to session state
                        st.session_state.config = {
                            'date_column': date_col,
                            'revenue_column': revenue_col,
                            'media_columns': media_cols,
                            'exog_columns': exog_cols,
                            'validation_passed': validation['is_valid']
                        }
                        
                        st.success("✅ 설정이 성공적으로 저장되었습니다!")
                        st.balloons()
                        
                        # Show saved configuration
                        with st.expander("📋 View Saved Configuration"):
                            st.json(st.session_state.config)
            else:
                st.info("👆 검증을 진행하려면 날짜 컬럼, 수익 컬럼, 최소 하나의 미디어 컬럼을 선택하세요.")
        
        except Exception as e:
            st.error(f"❌ 파일 로딩 오류: {str(e)}")
            st.exception(e)

else:
    # Show upload instructions
    st.markdown("""
    <div class="upload-box">
        <h3>📤 Upload Your Marketing Data</h3>
        <p>Drag and drop your CSV file here, or click to browse</p>
        <p style="color: #666; font-size: 0.9rem;">Maximum file size: 10MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 데이터 형식 요구사항")
    
    st.markdown("""
    CSV 파일에는 다음이 포함되어야 합니다:
    
    **필수 컬럼:**
    - **날짜**: 주간 또는 일간 날짜 (YYYY-MM-DD 형식)
    - **수익**: 목표 변수 (매출, 수익, 전환)
    - **미디어 채널**: 최소 하나의 마케팅 지출 컬럼
    
    **선택 컬럼:**
    - 계절성 지표
    - 프로모션 플래그
    - 공휴일 지표
    - 기타 외생 변수
    
    **예시:**
    ```
    date,revenue,google_uac,meta,apple_search,youtube,tiktok,promotion,seasonality
    2022-01-03,125000,15000,12000,8000,10000,5000,0,1
    2022-01-10,130000,16000,13000,8500,11000,5500,1,1
    ```
    """)
    
    # Show sample data download option
    st.markdown("### 📥 샘플 데이터가 필요하신가요?")
    st.info("홈 페이지로 돌아가서 샘플 데이터를 다운로드하고 도구를 탐색하세요.")

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Data Upload Status")
    
    if st.session_state.get('data_uploaded', False):
        st.success("✅ Data uploaded")
        if st.session_state.get('config', {}):
            st.success("✅ Configuration saved")
            
            config = st.session_state.config
            st.markdown("**Current Configuration:**")
            st.write(f"📅 Date: `{config.get('date_column', 'N/A')}`")
            st.write(f"💰 Revenue: `{config.get('revenue_column', 'N/A')}`")
            st.write(f"📺 Media Channels: {len(config.get('media_columns', []))}")
            st.write(f"🔧 Exog Variables: {len(config.get('exog_columns', []))}")
        else:
            st.warning("⚠️ Configuration not saved")
    else:
        st.info("📁 아직 데이터가 업로드되지 않았습니다")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - 최소 52주의 데이터 사용
    - 일관된 시간 간격 보장
    - 결측값 확인
    - 이상치 신중히 검토
    - 모든 금액 값은 동일한 통화 단위 사용
    """)
