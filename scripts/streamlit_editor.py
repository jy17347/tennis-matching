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

# AG Grid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# 커스텀 드래그앤드롭 컴포넌트
import streamlit.components.v1 as components
_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), 'components', 'match_editor')
_match_editor = components.declare_component('match_editor', path=_COMPONENT_DIR)

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
            
            # 스케줄 DataFrame 생성 (테이블 편집용)
            schedule_rows = []
            for match in sorted(system.schedule, key=lambda m: (m.time_slot, m.court)):
                schedule_rows.append({
                    '타임': match.time_slot,
                    '코트': match.court,
                    '경기타입': match.match_type,
                    '팀1_선수1': match.team1[0].name,
                    '팀1_선수2': match.team1[1].name,
                    '팀2_선수1': match.team2[0].name,
                    '팀2_선수2': match.team2[1].name,
                })
            schedule_df = pd.DataFrame(schedule_rows)
            player_names = sorted([p.name for p in system.players])

            # 컴포넌트용 schedule_data 생성 (타임별 코트+벤치)
            player_gender_map = {p.name: ('female' if p.gender == 2 else 'male') for p in system.players}
            schedule_data = _build_schedule_data(system)
            player_genders = [{'name': n, 'gender': g} for n, g in player_gender_map.items()]

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
                'schedule_df': schedule_df,
                'player_names': player_names,
                'schedule_data': schedule_data,
                'player_genders': player_genders,
            }
            return True
        else:
            st.error("❌ 매칭 생성 실패. 조건을 만족하는 스케줄을 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        st.error(f"매칭 실행 중 오류 발생: {e}")
        return False


def _build_schedule_data(system):
    """TennisMatchingSystem에서 컴포넌트용 schedule_data 딕셔너리 생성"""
    time_slots_dict = {}
    for match in sorted(system.schedule, key=lambda m: (m.time_slot, m.court)):
        t = match.time_slot
        if t not in time_slots_dict:
            time_slots_dict[t] = {'time': t, 'courts': [], 'bench': []}
        time_slots_dict[t]['courts'].append({
            'court': match.court,
            'type':  match.match_type,
            'team1': [match.team1[0].name, match.team1[1].name],
            'team2': [match.team2[0].name, match.team2[1].name],
        })

    # 벤치 계산 (참여한 적 있는 선수 중 해당 타임에 경기 안 하는 선수)
    for t, slot in time_slots_dict.items():
        playing = set()
        for court in slot['courts']:
            playing.update(court['team1'])
            playing.update(court['team2'])
        slot['bench'] = [
            p.name for p in system.players
            if p.matches_played > 0 and p.name not in playing
        ]

    return {'time_slots': list(time_slots_dict.values())}


def _schedule_data_to_df(schedule_data):
    """컴포넌트에서 반환된 schedule_data를 DataFrame으로 변환"""
    rows = []
    for slot in schedule_data['time_slots']:
        for court in slot['courts']:
            t1 = court['team1']
            t2 = court['team2']
            rows.append({
                '타임': slot['time'],
                '코트': court['court'],
                '경기타입': court['type'],
                '팀1_선수1': t1[0] if len(t1) > 0 else '',
                '팀1_선수2': t1[1] if len(t1) > 1 else '',
                '팀2_선수1': t2[0] if len(t2) > 0 else '',
                '팀2_선수2': t2[1] if len(t2) > 1 else '',
            })
    return pd.DataFrame(rows)


def regenerate_excel_from_df(schedule_df):
    """수정된 스케줄 DataFrame으로 Excel 재생성, bytes 반환"""
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        schedule_df.to_excel(writer, sheet_name='매칭결과', index=False)

        time_slots = sorted(schedule_df['타임'].unique())
        courts     = sorted(schedule_df['코트'].unique())
        player_cols = ['팀1_선수1', '팀1_선수2', '팀2_선수1', '팀2_선수2']
        timetable_rows = []
        for t in time_slots:
            row = {'타임': t}
            time_matches = schedule_df[schedule_df['타임'] == t]
            playing = set()
            for _, m in time_matches.iterrows():
                for col in player_cols:
                    playing.add(m[col])
            for c in courts:
                court_match = time_matches[time_matches['코트'] == c]
                if len(court_match) > 0:
                    m = court_match.iloc[0]
                    t1 = f"{m['팀1_선수1']} & {m['팀1_선수2']}"
                    t2 = f"{m['팀2_선수1']} & {m['팀2_선수2']}"
                    row[f'코트{c}'] = f"[{m['경기타입']}]\n{t1}\nvs\n{t2}"
                else:
                    row[f'코트{c}'] = '-'
            timetable_rows.append(row)
        pd.DataFrame(timetable_rows).to_excel(writer, sheet_name='타임표', index=False)
    return output.getvalue()


def display_matching_result():
    """session_state에 저장된 매칭 결과를 렌더링"""
    result = st.session_state.get('matching_result')
    if not result:
        return

    timestamp = result['timestamp']
    st.success("✅ 매칭 생성 완료!")

    # 다운로드 버튼 (항상 상단 고정)
    col_pdf, col_excel = st.columns(2)
    with col_pdf:
        if result['pdf_bytes']:
            st.download_button(
                label="📥 PDF 다운로드",
                data=result['pdf_bytes'],
                file_name=f'테니스_매칭결과_{timestamp}.pdf',
                mime='application/pdf',
                use_container_width=True
            )
    with col_excel:
        dl_key = st.session_state.get('excel_dl_key', 'excel_dl_0')
        dl_bytes = st.session_state.get('edited_excel_bytes') or result['excel_bytes']
        if dl_bytes:
            st.download_button(
                label="📊 Excel 다운로드",
                data=dl_bytes,
                file_name=f'테니스_매칭결과_{timestamp}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True,
                key=dl_key
            )

    st.markdown("---")

    # 결과 탭: PDF 미리보기 / 테이블 편집
    tab_pdf, tab_table, tab_stats = st.tabs(["📄 PDF 미리보기", "📋 테이블 편집", "📊 통계"])

    # ── 탭1: PDF 미리보기 ──────────────────────────────────
    with tab_pdf:
        if result['pdf_generated']:
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

    # ── 탭2: 드래그앤드롭 편집 ────────────────────────────
    with tab_table:
        schedule_data  = result.get('schedule_data')
        player_genders = result.get('player_genders', [])

        if schedule_data is None:
            st.info("스케줄 데이터가 없습니다.")
        else:
            st.caption("💡 같은 타임 안에서 선수 카드를 드래그해 코트 또는 벤치 간 이동이 가능합니다.")

            # 컴포넌트 호출 → 변경 시 updated_data 반환
            updated_data = _match_editor(
                schedule_data=schedule_data,
                player_genders=player_genders,
                key='match_editor_component',
            )

            if st.button("💾 변경사항 적용 (Excel 재생성)", type="primary"):
                # 컴포넌트가 아직 값을 반환하지 않았으면 원본 사용
                data_to_save = updated_data if updated_data else schedule_data
                edited_df = _schedule_data_to_df(data_to_save)
                new_excel_bytes = regenerate_excel_from_df(edited_df)
                st.session_state['edited_excel_bytes'] = new_excel_bytes
                # 다운로드 버튼 key 갱신
                prev = st.session_state.get('excel_dl_key', 'excel_dl_0')
                n = int(prev.split('_')[-1]) + 1
                st.session_state['excel_dl_key'] = f'excel_dl_{n}'
                st.success("✅ Excel이 재생성되었습니다. 상단 'Excel 다운로드' 버튼으로 받으세요.")
                st.rerun()

    # ── 탭3: 통계 ─────────────────────────────────────────
    with tab_stats:
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
