"""
한글 번역용 텍스트 매핑

이 파일은 Streamlit 앱의 모든 영어 텍스트를 한글로 매핑합니다.
"""

# 공통 메시지
COMMON_MESSAGES = {
    # 경고 및 오류
    "No trained model found. Please train a model in the Model Configuration page first.": 
        "학습된 모델이 없습니다. Model Configuration 페이지에서 먼저 모델을 학습시켜주세요.",
    
    "No trained model found. Please train a model first.":
        "학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.",
    
    # 성공 메시지
    "Optimization successful!": "최적화 성공!",
    "Analysis complete!": "분석 완료!",
    "Model training completed successfully!": "모델 학습이 성공적으로 완료되었습니다!",
    
    # 버튼
    "Download": "다운로드",
    "Save": "저장",
    "Run": "실행",
    "Reset": "초기화",
}

# Data Upload 페이지
DATA_UPLOAD = {
    "subtitle": "마케팅 데이터를 업로드하고 검증하세요",
    "upload_section": "CSV 파일 업로드",
    "upload_help": "마케팅 데이터가 포함된 CSV 파일을 업로드하세요 (최대 10MB)",
    "preview_section": "데이터 미리보기",
    "column_mapping": "컬럼 매핑",
    "date_column": "날짜 컬럼",
    "revenue_column": "수익 컬럼",
    "media_columns": "미디어 채널 컬럼",
    "exog_columns": "외생 변수 (선택사항)",
    "validation": "데이터 검증",
    "missing_values": "결측값",
    "outliers": "이상치",
    "save_config": "설정 저장",
}

# Model Config 페이지
MODEL_CONFIG = {
    "subtitle": "변환 파라미터 및 모델 설정 구성",
    "adstock_section": "Adstock 변환 설정",
    "adstock_help": "마케팅 효과의 지속 효과를 모델링합니다",
    "decay_rate": "감쇠율",
    "hill_section": "Hill 포화 설정",
    "hill_help": "수익 체감 효과를 모델링합니다",
    "scale_param": "스케일 파라미터 (K)",
    "shape_param": "형상 파라미터 (S)",
    "model_params": "모델 파라미터",
    "alpha": "Ridge 알파",
    "train_split": "학습/테스트 분할",
    "train_model": "모델 학습",
    "preview": "변환 미리보기",
}

# Results 페이지
RESULTS = {
    "subtitle": "MMM 모델의 종합 분석",
    "performance_tab": "모델 성능",
    "contributions_tab": "채널 기여도",
    "decomposition_tab": "시계열 분해",
    "curves_tab": "반응 곡선",
    "diagnostics_tab": "모델 진단",
    "metrics": "성능 지표",
    "interpretation": "해석",
    "download_results": "결과 다운로드",
}

# Budget Optimizer 페이지
BUDGET_OPTIMIZER = {
    "subtitle": "채널 간 마케팅 예산의 최적 배분 찾기",
    "current_vs_optimal": "현재 vs 최적 배분",
    "find_best": "예산을 배분하는 최적의 방법을 찾으세요",
    "total_budget": "총 예산",
    "optimize_button": "예산 최적화",
    "scenario_builder": "시나리오 빌더",
    "create_scenarios": "사용자 정의 예산 배분 시나리오 생성 및 비교",
    "sensitivity": "민감도 분석",
    "understand_changes": "지출 변화에 따른 수익 변화 이해",
    "diminishing_returns": "수익 체감 분석",
    "saturation_points": "각 채널의 포화 지점 식별",
    "optimization_settings": "최적화 설정",
    "constraints": "제약 조건",
    "tips": "팁",
}

# Report 페이지
REPORT = {
    "subtitle": "종합 MMM 분석 보고서 생성",
    "config_section": "보고서 구성",
    "report_title": "보고서 제목",
    "include_raw": "원시 데이터 시트 포함",
    "executive_summary": "요약 보고서",
    "analysis_period": "분석 기간",
    "model_performance": "모델 성능",
    "top_channels": "상위 성과 채널",
    "recommendations": "주요 권장사항",
    "detailed_analysis": "상세 분석",
    "channel_performance": "채널 성과",
    "download_section": "보고서 다운로드",
    "download_excel": "Excel 다운로드",
    "download_html": "HTML 다운로드",
    "preview": "보고서 미리보기",
}

# 도움말 텍스트
HELP_TEXTS = {
    "decay_rate": "0에 가까우면 즉각적인 효과, 1에 가까우면 장기적인 효과",
    "hill_k": "포화 수준을 제어합니다",
    "hill_s": "포화 곡선의 가파름을 제어합니다",
    "alpha": "높을수록 더 많은 정규화 (과적합 방지)",
    "train_split": "모델 학습에 사용할 데이터 비율",
    "total_budget": "모든 채널에 배분할 총 예산",
    "constraints": "각 채널의 최소/최대 지출 제한",
}

# 인사이트 메시지
INSIGHTS = {
    "excellent_roas": "우수한 효율성",
    "good_roas": "좋은 효율성",
    "poor_roas": "개선 필요",
    "excellent_fit": "우수한 모델 적합도",
    "good_fit": "좋은 모델 적합도",
    "fair_fit": "보통 모델 적합도",
    "increase_spend": "지출 증가 고려",
    "decrease_spend": "지출 감소 고려",
    "near_optimal": "현재 지출이 최적에 가깝습니다",
}
