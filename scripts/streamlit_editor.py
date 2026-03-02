# -*- coding: utf-8 -*-
"""
테니스 매칭 시스템 - Streamlit GUI Editor
참가자 데이터 편집 및 저장 후 매칭 처리
"""

import streamlit as st
import pandas as pd
import os
import sys
import base64
from datetime import datetime

# PDF를 이미지로 변환
try:
    from pdf2image import convert_from_path
    PDF_TO_IMAGE_AVAILABLE = True
except ImportError:
    PDF_TO_IMAGE_AVAILABLE = False

# tennis_matching 모듈 import
from tennis_matching import TennisMatchingSystem

# 페이지 설정
st.set_page_config(
    page_title="테니스 참가자 관리",
    page_icon="🎾",
    layout="wide"
)

# 파일 경로 설정
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
PARTICIPATION_FILE = os.path.join(DATASET_DIR, 'participation_sample.xlsx')
ROSTER_FILE = os.path.join(DATASET_DIR, 'roster.xlsx')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# 결과 폴더 생성
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_participation_data():
    """참가자 데이터 로드"""
    try:
        df = pd.read_excel(PARTICIPATION_FILE, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None


def save_participation_data(df):
    """참가자 데이터 저장"""
    try:
        df.to_excel(PARTICIPATION_FILE, index=False, engine='openpyxl')
        return True
    except Exception as e:
        st.error(f"파일 저장 실패: {e}")
        return False


def run_matching_algorithm(iterations=1000):
    """매칭 알고리즘 실행 - 결과를 session_state에 저장"""
    try:
        # 매칭 시스템 초기화
        system = TennisMatchingSystem(ROSTER_FILE, PARTICIPATION_FILE)
        
        # 유효성 검증
        try:
            system.validate_configuration()
        except ValueError as e:
            st.error(f"❌ 매칭 실행 불가: {e}")
            return False
        
        # 매칭 최적화
        with st.spinner(f'매칭을 최적화하고 있습니다... ({iterations}회 반복)'):
            schedule = system.optimize(iterations=iterations)
        
        if schedule and len(schedule) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(RESULTS_DIR, f'테니스타임표_{timestamp}.xlsx')
            pdf_path = os.path.join(RESULTS_DIR, f'테니스타임표.pdf')
            
            # Excel 저장
            system.export_to_excel(excel_path)
            
            # Excel 바이트 읽기
            excel_bytes = None
            try:
                with open(excel_path, "rb") as f:
                    excel_bytes = f.read()
            except Exception as ex_err:
                st.warning(f"Excel 읽기 실패: {ex_err}")
            
            # PDF 생성
            pdf_bytes = None
            pdf_images = None
            base64_pdf = None
            pdf_generated = system.export_to_pdf(pdf_path)
            
            if pdf_generated:
                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    if PDF_TO_IMAGE_AVAILABLE:
                        try:
                            images = convert_from_path(pdf_path, dpi=200)
                            # PIL Image를 base64로 변환하여 저장 (session_state 직렬화 가능)
                            import io
                            pdf_images = []
                            for img in images:
                                buf = io.BytesIO()
                                img.save(buf, format='PNG')
                                pdf_images.append(buf.getvalue())
                        except Exception:
                            pass
                    if pdf_images is None:
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                except Exception as e:
                    st.warning(f"PDF 읽기 실패: {e}")
            
            # 통계 데이터 수집
            match_types = {'남복': 0, '여복': 0, '혼복': 0}
            for match in system.schedule:
                match_types[match.match_type] += 1
            
            player_stats = []
            for player in system.players:
                gender_str = "남" if player.gender == 1 else "여"
                player_stats.append({
                    '이름': player.name,
                    '성별': gender_str,
                    '총 경기': player.matches_played,
                    '혼복': player.mixed_matches,
                    '단일복식': player.same_doubles
                })
            stats_df = pd.DataFrame(player_stats).sort_values('총 경기', ascending=False)
            
            # 결과를 session_state에 저장
            st.session_state['matching_result'] = {
                'timestamp': timestamp,
                'excel_bytes': excel_bytes,
                'pdf_bytes': pdf_bytes,
                'pdf_generated': pdf_generated,
                'pdf_images': pdf_images,
                'base64_pdf': base64_pdf,
                'match_types': match_types,
                'stats_df': stats_df,
            }
            return True
        else:
            st.error("❌ 매칭 생성 실패. 조건을 만족하는 스케줄을 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        st.error(f"매칭 실행 중 오류 발생: {e}")
        return False


def display_matching_result():
    """session_state에 저장된 매칭 결과를 렌더링"""
    result = st.session_state.get('matching_result')
    if not result:
        return
    
    timestamp = result['timestamp']
    st.success("✅ 매칭 생성 완료!")
    
    # 다운로드 버튼 (항상 상단에 고정)
    col_pdf, col_excel = st.columns(2)
    with col_pdf:
        if result['pdf_bytes']:
            st.download_button(
                label=" PDF 다운로드",
                data=result['pdf_bytes'],
                file_name=f'테니스_매칭결과_{timestamp}.pdf',
                mime='application/pdf',
                use_container_width=True
            )
    with col_excel:
        if result['excel_bytes']:
            st.download_button(
                label=" Excel 다운로드",
                data=result['excel_bytes'],
                file_name=f'테니스_매칭결과_{timestamp}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
    
    # 매칭 결과 이미지 미리보기
    if result['pdf_generated']:
        st.markdown("---")
        st.subheader(" 매칭 결과")
        if result['pdf_images']:
            for i, img_bytes in enumerate(result['pdf_images']):
                st.image(img_bytes, caption=f'페이지 {i+1}', use_container_width=True)
                if i < len(result['pdf_images']) - 1:
                    st.markdown("---")
        elif result['base64_pdf']:
            if not PDF_TO_IMAGE_AVAILABLE:
                st.info("💡 이미지로 보려면 pdf2image 라이브러리를 설치하세요: `pip install pdf2image`")
            pdf_display = f'<iframe src="data:application/pdf;base64,{result["base64_pdf"]}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.warning("⚠️ PDF 생성 실패 (reportlab 라이브러리 필요)")
    
    # 통계
    st.markdown("---")
    st.subheader(" 매칭 통계")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("남복 경기", result['match_types']['남복'])
    with col2:
        st.metric("여복 경기", result['match_types']['여복'])
    with col3:
        st.metric("혼복 경기", result['match_types']['혼복'])
    
    st.subheader("선수별 참여 횟수")
    st.dataframe(result['stats_df'], use_container_width=True)


def main():
    """메인 함수"""
    st.title("🎾 사방팔방 매칭")
    
    # 커스텀 CSS 스타일
    st.markdown("""
    <style>
    /* 우측 상단 툴바 (Share, GitHub 아이콘 등) 숨기기 */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 24px;
        background-color: #6c757d;
        border-radius: 8px 8px 0px 0px;
        font-size: 18px;
        font-weight: 600;
        color: white !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: white !important;
        background-color: #5a6268;
    }
    
    /* Primary 버튼 스타일 */
    .stButton > button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #45a049;
        border: none;
    }
    
    /* 일반 버튼 스타일 */
    .stButton > button[kind="secondary"] {
        background-color: #2196F3;
        color: white;
        border: none;
        font-weight: 500;
        font-size: 15px;
        padding: 10px 20px;
        border-radius: 6px;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #0b7dda;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 상단 탭 메뉴
    tab1, tab2 = st.tabs(["📝 참가자 편집", "⚙️ 매칭 생성"])
    
    # 참가자 편집 탭
    with tab1:
        
        # 데이터 로드
        if 'df' not in st.session_state:
            st.session_state.df = load_participation_data()
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # 데이터 정보
            # st.info(f"총 {len(df)}명의 데이터가 있습니다.")
            
            # # 필터링 옵션
            # col1, col2 = st.columns([1, 3])
            # with col1:
            #     show_only_participants = st.checkbox("참가자만 보기", value=False)
            
            # # 데이터 필터링
            # if show_only_participants:
            #     display_df = df[df['참여 (1)'].isin(['O', '1', 1])].copy()
            # else:
            #     display_df = df.copy()
            display_df = df.copy()
            # st.markdown(f"**표시 중: {len(display_df)}명**")
            
            # 데이터 편집기
            st.markdown("### 데이터 편집")
            st.markdown("※ '참여 (1)' 열에 'O', '1' 또는 1을 입력하면 참가자로 등록")
            
            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                num_rows="dynamic",
                height=600
            )
            # show_only_participants = st.checkbox("참가자만 보기", value=False)
            # 저장 버튼
            col1, col2, col3 = st.columns([1, 1, 5])
            with col1:
                if st.button("💾 변경사항 저장", type="primary"):
                    # # 필터링된 경우 원본 데이터와 병합
                    # if show_only_participants:
                    #     # 편집된 행만 업데이트
                    #     for idx in edited_df.index:
                    #         st.session_state.df.loc[idx] = edited_df.loc[idx]
                    #     save_df = st.session_state.df
                    # else:
                    #     save_df = edited_df
                    save_df = edited_df
                    if save_participation_data(save_df):
                        st.session_state.df = save_df
                        st.success("✅ 저장 완료!")
                        st.rerun()
                    else:
                        st.error("❌ 저장 실패")
            
            with col2:
                if st.button("↩️ 초기화", type="secondary"):
                    # 참여 여부 초기화 (모든 참가자 체크 해제)
                    reset_df = st.session_state.df.copy()
                    reset_df['참여 (1)'] = None  # 또는 '' 빈 문자열
                    if save_participation_data(reset_df):
                        st.session_state.df = reset_df
                        st.success("✅ 참여 여부를 모두 초기화했습니다!")
                        st.rerun()
                    else:
                        st.error("❌ 초기화 실패")
            
            # 참가자 요약
            st.markdown("---")
            st.subheader(" 참가자 현황")
            
            participants = df[df['참여 (1)'].isin(['O', '1', 1])]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 참가자", len(participants))
            with col2:
                # roster 파일에서 성별 정보 가져오기
                try:
                    roster_df = pd.read_excel(ROSTER_FILE, engine='openpyxl')
                    merged = participants.merge(roster_df[['성명', '성별']], on='성명', how='left')
                    male_count = len(merged[merged['성별'] == 1])
                    st.metric("남자", male_count)
                except:
                    st.metric("남자", "-")
            with col3:
                try:
                    female_count = len(merged[merged['성별'] == 2])
                    st.metric("여자", female_count)
                except:
                    st.metric("여자", "-")
            
            # 참가자 목록
            if len(participants) > 0:
                st.markdown("### 참가자 명단")
                try:
                    participant_list = participants.merge(
                        roster_df[['성명', '성별']], 
                        on='성명', 
                        how='left'
                    )
                    participant_list['성별'] = participant_list['성별'].map({1: '남', 2: '여'})
                    st.dataframe(
                        participant_list[['성명', '성별']],
                        use_container_width=True
                    )
                except:
                    st.dataframe(participants[['성명']], use_container_width=True)
    
    # 매칭 생성 탭
    with tab2:
        # 현재 참가자 정보 표시
        df = load_participation_data()
        if df is not None:
            participants = df[df['참여 (1)'].isin(['O', '1', 1])]            
            try:
                roster_df = pd.read_excel(ROSTER_FILE, engine='openpyxl')
                merged = participants.merge(roster_df[['성명', '성별']], on='성명', how='left')
                male_count = len(merged[merged['성별'] == 1])
                female_count = len(merged[merged['성별'] == 2])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("남자", male_count)
                with col2:
                    st.metric("여자", female_count)
                with col3:
                    st.metric("총 참가자", len(participants))
                
                # 매칭 조건 체크
                # if male_count < 4:
                #     st.error("⚠️ 남자 참가자가 최소 4명 이상이어야 합니다.")
                if len(participants) < 4:
                    st.error("⚠️ 총 참가자가 최소 4명 이상이어야 합니다.")
                else:
                    st.success("✅ 매칭 생성 가능")
                    
                    # 매칭 옵션
                    st.markdown("### 매칭 옵션")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        iterations = st.number_input(
                            "최대 반복 횟수",
                            min_value=100,
                            max_value=10000,
                            value=1000,
                            step=100,
                            help="더 많은 반복으로 더 좋은 결과를 얻을 수 있습니다."
                        )
                    
                    # 매칭 실행 버튼
                    if st.button("🎾 매칭 생성 시작", type="primary"):
                        run_matching_algorithm(iterations=iterations)
                
            except Exception as e:
                st.error(f"데이터 확인 중 오류: {e}")
        else:
            st.error("참가자 데이터를 불러올 수 없습니다.")
        
        # 매칭 결과 표시 (session_state에서 렌더링 - 다운로드 버튼 클릭 후에도 유지됨)
        st.markdown("---")
        display_matching_result()


if __name__ == "__main__":
    main()
