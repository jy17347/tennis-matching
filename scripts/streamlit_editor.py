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

# streamlit-sortables (드래그앤드롭 편집)
try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False

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


def run_matching_algorithm(iterations=1000, time_slots=7):
    """매칭 알고리즘 실행 - 결과를 session_state에 저장"""
    try:
        # 매칭 시스템 초기화
        system = TennisMatchingSystem(ROSTER_FILE, PARTICIPATION_FILE, time_slots=time_slots)
        
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
            # 편집 상태 초기화 (새 매칭 생성 시 이전 편집 내용 제거)
            st.session_state.pop('edit_schedule', None)
            st.session_state.pop('edited_excel_bytes', None)
            st.session_state.pop('edited_pdf_bytes', None)
            st.session_state.pop('edited_pdf_images', None)
            st.session_state.pop('edited_stats', None)
            st.session_state.pop('pdf_dl_key', None)
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


def _recalculate_stats(schedule_data, player_genders):
    """편집된 schedule_data로 match_types, stats_df 재계산"""
    match_types = {'남복': 0, '여복': 0, '혼복': 0}
    # 선수별 집계
    player_counts = {}  # name -> {'total': 0, '혼복': 0, '단일': 0}
    for pg in player_genders:
        player_counts[pg['name']] = {'total': 0, '혼복': 0, '단일': 0}

    for slot in schedule_data['time_slots']:
        for court in slot['courts']:
            ctype = court['type']
            if ctype in match_types:
                match_types[ctype] += 1
            for name in list(court['team1']) + list(court['team2']):
                if name not in player_counts:
                    player_counts[name] = {'total': 0, '혼복': 0, '단일': 0}
                player_counts[name]['total'] += 1
                if ctype == '혼복':
                    player_counts[name]['혼복'] += 1
                else:
                    player_counts[name]['단일'] += 1

    gender_map = {pg['name']: pg['gender'] for pg in player_genders}
    rows = []
    for name, cnt in player_counts.items():
        g = gender_map.get(name, 'male')
        rows.append({
            '이름': name,
            '성별': '여' if g == 'female' else '남',
            '총 경기': cnt['total'],
            '혼복': cnt['혼복'],
            '단일복식': cnt['단일'],
        })
    stats_df = pd.DataFrame(rows).sort_values('총 경기', ascending=False)
    return match_types, stats_df


def _infer_match_type(team1, team2, gender_map):
    """4명 선수의 성별로 경기 타입 자동 결정.
    전원 male -> '남복', 전원 female -> '여복', 혼성 -> '혼복'
    성별 미확인 선수는 male로 간주
    """
    players = list(team1) + list(team2)
    genders = {gender_map.get(p, 'male') for p in players if p}
    if genders == {'female'}:
        return '여복'
    elif 'female' not in genders:
        return '남복'
    else:
        return '혼복'


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


def regenerate_pdf_from_schedule_data(schedule_data):
    """편집된 schedule_data로 PDF bytes 생성 (reportlab 사용)"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
    except ImportError:
        return None, "reportlab 설치 필요 (pip install reportlab)"

    # 한글 폰트 등록
    font_registered = False
    for font_path in [
        'C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        './fonts/NanumGothic.ttf', '../fonts/NanumGothic.ttf',
    ]:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('KoreanEdit', font_path))
                font_registered = True
                break
            except Exception:
                continue
    korean_font = 'KoreanEdit' if font_registered else 'Helvetica'

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            rightMargin=1*cm, leftMargin=1*cm,
                            topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'],
                                 fontName=korean_font, fontSize=20, alignment=1)
    normal_style = ParagraphStyle('N', parent=styles['Normal'],
                                  fontName=korean_font, fontSize=10)

    elements = []
    elements.append(Paragraph("테니스 타임표 (편집본)", title_style))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(Paragraph(f"생성일: {datetime.now().strftime('%Y년 %m월 %d일')}", normal_style))
    elements.append(Spacer(1, 0.4*cm))

    slots = schedule_data['time_slots']
    courts_all = sorted({c['court'] for s in slots for c in s['courts']})
    num_courts = len(courts_all)

    # 컬럼 너비 계산 (landscape A4 ≒ 25.7 cm 사용 가능)
    available = 25.7
    time_w  = 1.5
    bench_w = 5.0
    court_w = round((available - time_w - bench_w) / max(num_courts, 1), 2)
    col_widths = [time_w*cm] + [court_w*cm]*num_courts + [bench_w*cm]

    header = ['타임'] + [f'코트 {c}' for c in courts_all] + ['벤치']
    table_data = [header]

    for slot in slots:
        t = slot['time']
        court_map = {c['court']: c for c in slot['courts']}
        row = [str(t)]
        for cn in courts_all:
            court = court_map.get(cn)
            if not court:
                row.append('-')
            else:
                p1 = f"{court['team1'][0]} & {court['team1'][1]}" if len(court['team1']) >= 2 else ' & '.join(court['team1'])
                p2 = f"{court['team2'][0]} & {court['team2'][1]}" if len(court['team2']) >= 2 else ' & '.join(court['team2'])
                row.append(f"[{court['type']}]\n{p1}\nvs\n{p2}")
        bench = slot.get('bench', [])
        if bench:
            bench_lines = [', '.join(bench[i:i+3]) for i in range(0, len(bench), 3)]
            row.append('\n'.join(bench_lines))
        else:
            row.append('-')
        table_data.append(row)

    table = Table(table_data, colWidths=col_widths)
    bench_col = num_courts + 1
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME',   (0, 0), (-1, -1), korean_font),
        ('FONTSIZE',   (0, 0), (-1, 0), 12),
        ('FONTSIZE',   (0, 1), (-1, -1), 9),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',       (0, 0), (-1, -1), 1, colors.black),
        ('TOPPADDING',    (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#D6DCE5')),
        ('BACKGROUND', (bench_col, 1), (bench_col, -1), colors.HexColor('#FFF2CC')),
    ])
    TYPE_COLOR = {'남복': '#DDEBF7', '여복': '#FCE4D6', '혼복': '#E2EFDA'}
    for ri, slot in enumerate(slots, start=1):
        court_map = {c['court']: c for c in slot['courts']}
        for ci, cn in enumerate(courts_all, start=1):
            court = court_map.get(cn)
            if court and court['type'] in TYPE_COLOR:
                ts.add('BACKGROUND', (ci, ri), (ci, ri), colors.HexColor(TYPE_COLOR[court['type']]))
    table.setStyle(ts)
    elements.append(table)

    elements.append(Spacer(1, 0.5*cm))
    legend = Table([['경기:', '남복', '여복', '혼복']],
                   colWidths=[2*cm, 4*cm, 4*cm, 4*cm])
    legend.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), korean_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#DDEBF7')),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#FCE4D6')),
        ('BACKGROUND', (3, 0), (3, 0), colors.HexColor('#E2EFDA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(legend)

    try:
        doc.build(elements)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)


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
        # 편집된 PDF 우선, 없으면 원본 사용
        pdf_dl_bytes = st.session_state.get('edited_pdf_bytes') or result['pdf_bytes']
        pdf_dl_label = "📥 PDF 다운로드 (편집본)" if st.session_state.get('edited_pdf_bytes') else "📥 PDF 다운로드"
        if pdf_dl_bytes:
            st.download_button(
                label=pdf_dl_label,
                data=pdf_dl_bytes,
                file_name=f'테니스_매칭결과_{timestamp}.pdf',
                mime='application/pdf',
                use_container_width=True,
                key=st.session_state.get('pdf_dl_key', 'pdf_dl_0')
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
        # 편집된 PDF/이미지 우선 사용
        edited_pdf_images = st.session_state.get('edited_pdf_images')
        edited_pdf_bytes  = st.session_state.get('edited_pdf_bytes')

        if edited_pdf_images:
            st.info("📝 편집된 스케줄 기준 PDF입니다.")
            for i, img_bytes in enumerate(edited_pdf_images):
                st.image(img_bytes, caption=f'페이지 {i+1}', use_container_width=True)
                if i < len(edited_pdf_images) - 1:
                    st.markdown("---")
        elif edited_pdf_bytes:
            st.info("📝 편집된 스케줄 기준 PDF입니다.")
            b64 = base64.b64encode(edited_pdf_bytes).decode('utf-8')
            st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800" type="application/pdf"></iframe>',
                        unsafe_allow_html=True)
        elif result['pdf_generated']:
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

    # ── 탭2: 테이블 보기 + 드래그앤드롭 편집 ──────────────
    with tab_table:
        schedule_data  = result.get('schedule_data')
        player_genders = result.get('player_genders', [])

        if schedule_data is None:
            st.info("스케줄 데이터가 없습니다.")
        else:
            import copy
            if 'edit_schedule' not in st.session_state:
                st.session_state['edit_schedule'] = copy.deepcopy(schedule_data)

            gender_map = {pg['name']: pg['gender'] for pg in player_genders}

            # ── PDF 형태 HTML 테이블 렌더링 ──────────────────
            TYPE_BG  = {'남복': '#DDEBF7', '여복': '#FCE4D6', '혼복': '#E2EFDA'}
            TYPE_HDR = {'남복': '#4472C4', '여복': '#C0504D', '혼복': '#4CAF50'}
            edit_sd  = st.session_state['edit_schedule']
            courts_all = sorted({c['court'] for s in edit_sd['time_slots'] for c in s['courts']})

            col_w   = max(120, min(200, 700 // (len(courts_all) + 1)))
            th_style = "background:#4472C4;color:#fff;font-weight:700;font-size:13px;padding:7px 6px;text-align:center;border:1px solid #2a55a3;"
            bench_th = "background:#4472C4;color:#fff;font-weight:700;font-size:13px;padding:7px 6px;text-align:center;border:1px solid #2a55a3;width:110px;"

            html  = f"<table style='border-collapse:collapse;width:100%;font-family:Malgun Gothic,sans-serif;font-size:12px;'>"
            html += "<thead><tr>"
            html += f"<th style='{th_style}width:52px;'>타임</th>"
            for c in courts_all:
                html += f"<th style='{th_style}width:{col_w}px;'>코트 {c}</th>"
            html += f"<th style='{bench_th}'>벤치</th>"
            html += "</tr></thead><tbody>"

            for slot in edit_sd['time_slots']:
                t = slot['time']
                court_map = {c['court']: c for c in slot['courts']}
                html += "<tr>"
                html += f"<td style='background:#D6DCE5;font-weight:700;text-align:center;vertical-align:middle;border:1px solid #bbb;padding:4px;'>{t}타임</td>"
                for cn in courts_all:
                    court = court_map.get(cn)
                    if not court:
                        html += "<td style='background:#f5f5f5;border:1px solid #bbb;text-align:center;color:#aaa;'>-</td>"
                        continue
                    bg   = TYPE_BG.get(court['type'], '#fff')
                    hdr  = TYPE_HDR.get(court['type'], '#555')
                    t1   = court['team1']
                    t2   = court['team2']
                    ok   = len(t1) == 2 and len(t2) == 2
                    warn = '' if ok else f"<div style='color:#c00;font-size:10px;'>⚠️ {len(t1)}+{len(t2)}명</div>"
                    p1 = ' & '.join(t1) if t1 else '-'
                    p2 = ' & '.join(t2) if t2 else '-'
                    html += (
                        f"<td style='background:{bg};border:1px solid #bbb;padding:5px 4px;vertical-align:top;'>"
                        f"<div style='background:{hdr};color:#fff;border-radius:9px;font-size:10px;font-weight:700;display:inline-block;padding:1px 7px;margin-bottom:3px;'>{court['type']}</div>"
                        f"<div style='font-size:11px;color:#333;'><b>팀1</b> {p1}</div>"
                        f"<div style='font-size:10px;color:#888;text-align:center;font-weight:700;'>vs</div>"
                        f"<div style='font-size:11px;color:#333;'><b>팀2</b> {p2}</div>"
                        f"{warn}</td>"
                    )
                bench = slot.get('bench', [])
                bench_txt = '<br>'.join(bench) if bench else '<span style="color:#aaa">없음</span>'
                html += f"<td style='background:#FFF2CC;border:1px solid #bbb;padding:5px 4px;vertical-align:top;font-size:11px;'>{bench_txt}</td>"
                html += "</tr>"

            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 타임 / 코트 교환 섹션 ───────────────────────
            with st.expander("↕️ 타임 / 코트 교환", expanded=False):
                import copy as _copy

                time_labels = [str(s['time']) + '타임' for s in edit_sd['time_slots']]
                court_labels = [f'코트 {cn}' for cn in courts_all]

                st.markdown("**타임 교환** — 두 타임 전체(선수+벤치+타입)를 서로 바꿉니다.")
                tc1, tc2, tc3 = st.columns([2, 2, 1])
                ta = tc1.selectbox("타임 A", time_labels, key='swap_time_a')
                tb = tc2.selectbox("타임 B", time_labels, index=min(1, len(time_labels)-1), key='swap_time_b')
                if tc3.button("교환", key='do_swap_time'):
                    if ta != tb:
                        idx_a = time_labels.index(ta)
                        idx_b = time_labels.index(tb)
                        slots = st.session_state['edit_schedule']['time_slots']
                        # 선수/벤치/코트 데이터만 교환 (time 번호는 유지)
                        data_a = _copy.deepcopy({'courts': slots[idx_a]['courts'], 'bench': slots[idx_a]['bench']})
                        data_b = _copy.deepcopy({'courts': slots[idx_b]['courts'], 'bench': slots[idx_b]['bench']})
                        slots[idx_a]['courts'] = data_b['courts']
                        slots[idx_a]['bench']  = data_b['bench']
                        slots[idx_b]['courts'] = data_a['courts']
                        slots[idx_b]['bench']  = data_a['bench']
                        st.success(f"✅ {ta} ↔ {tb} 교환 완료")
                        st.rerun()
                    else:
                        st.warning("다른 타임을 선택하세요.")

                st.markdown("---")
                st.markdown("**코트 교환** — 두 코트 열 전체(모든 타임)를 서로 바꿉니다.")
                cc1, cc2, cc3 = st.columns([2, 2, 1])
                ca = cc1.selectbox("코트 A", court_labels, key='swap_court_a')
                cb = cc2.selectbox("코트 B", court_labels, index=min(1, len(court_labels)-1), key='swap_court_b')
                if cc3.button("교환", key='do_swap_court'):
                    cn_a = courts_all[court_labels.index(ca)]
                    cn_b = courts_all[court_labels.index(cb)]
                    if cn_a != cn_b:
                        slots = st.session_state['edit_schedule']['time_slots']
                        for slot in slots:
                            court_map_s = {c['court']: c for c in slot['courts']}
                            if cn_a in court_map_s and cn_b in court_map_s:
                                # 팀/벤치 데이터 교환, court 번호는 유지
                                ca_data = _copy.deepcopy({'team1': court_map_s[cn_a]['team1'],
                                                          'team2': court_map_s[cn_a]['team2'],
                                                          'type':  court_map_s[cn_a]['type']})
                                cb_data = _copy.deepcopy({'team1': court_map_s[cn_b]['team1'],
                                                          'team2': court_map_s[cn_b]['team2'],
                                                          'type':  court_map_s[cn_b]['type']})
                                court_map_s[cn_a]['team1'] = cb_data['team1']
                                court_map_s[cn_a]['team2'] = cb_data['team2']
                                court_map_s[cn_a]['type']  = cb_data['type']
                                court_map_s[cn_b]['team1'] = ca_data['team1']
                                court_map_s[cn_b]['team2'] = ca_data['team2']
                                court_map_s[cn_b]['type']  = ca_data['type']
                        st.success(f"✅ {ca} ↔ {cb} 교환 완료")
                        st.rerun()
                    else:
                        st.warning("다른 코트를 선택하세요.")

            # ── 드래그앤드롭 편집 섹션 ───────────────────────
            with st.expander("✏️ 선수 배치 편집 (드래그앤드롭)", expanded=False):
                if not SORTABLES_AVAILABLE:
                    st.warning("`pip install streamlit-sortables` 설치 필요")
                else:
                    st.caption("💡 카드를 드래그해 같은 타임 안에서 코트/벤치 간 이동 후 '적용' 버튼을 누르세요.")
                    updated_slots = []
                    for slot in edit_sd['time_slots']:
                        t      = slot['time']
                        courts = slot['courts']
                        bench  = slot['bench']

                        st.markdown(
                            f"<div style='background:#4472C4;color:#fff;font-weight:700;"
                            f"padding:5px 12px;border-radius:6px 6px 0 0;"
                            f"margin-top:10px;font-size:13px;'>⏱ {t}타임</div>",
                            unsafe_allow_html=True
                        )
                        TYPE_TAG = {'남복': '🔵', '여복': '🔴', '혼복': '🟢'}
                        containers = [{'header': '🏃 벤치', 'items': list(bench)}]
                        for court in courts:
                            c     = court['court']
                            ctype = court['type']
                            tag   = TYPE_TAG.get(ctype, '')
                            containers.append({'header': f"{tag} 코트{c} 팀1 [{ctype}]", 'items': list(court['team1'])})
                            containers.append({'header': f"{tag} 코트{c} 팀2 [{ctype}]", 'items': list(court['team2'])})

                        result_containers = sort_items(
                            containers, multi_containers=True, direction='horizontal', key=f'sort_{t}'
                        )

                        import re
                        new_slot  = {'time': t, 'bench': [], 'courts': []}
                        court_tmp = {}
                        for rc in result_containers:
                            header = rc['header']
                            items  = rc['items']
                            if '벤치' in header:
                                new_slot['bench'] = items
                            else:
                                m2 = re.search(r'코트(\d+)\s*팀(\d+).*\[(.+?)\]', header)
                                if m2:
                                    cn2 = int(m2.group(1)); tn = int(m2.group(2)); ctype = m2.group(3)
                                    if cn2 not in court_tmp:
                                        court_tmp[cn2] = {'court': cn2, 'type': ctype, 'team1': [], 'team2': []}
                                    if tn == 1: court_tmp[cn2]['team1'] = items
                                    else:       court_tmp[cn2]['team2'] = items
                        for cn2 in sorted(court_tmp):
                            new_slot['courts'].append(court_tmp[cn2])
                        updated_slots.append(new_slot)

                    st.session_state['edit_schedule']['time_slots'] = updated_slots

            st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
            if st.button("💾 변경사항 적용 (Excel + PDF 재생성)", type="primary", key='apply_edit'):
                # 성별 기반 경기 타입 자동 할당
                for slot in st.session_state['edit_schedule']['time_slots']:
                    for court in slot['courts']:
                        court['type'] = _infer_match_type(court['team1'], court['team2'], gender_map)

                edited_df = _schedule_data_to_df(st.session_state['edit_schedule'])

                # Excel 재생성
                new_excel_bytes = regenerate_excel_from_df(edited_df)
                st.session_state['edited_excel_bytes'] = new_excel_bytes
                prev = st.session_state.get('excel_dl_key', 'excel_dl_0')
                st.session_state['excel_dl_key'] = f'excel_dl_{int(prev.split("_")[-1]) + 1}'

                # PDF 재생성
                with st.spinner('PDF를 재생성하는 중...'):
                    new_pdf_bytes, pdf_err = regenerate_pdf_from_schedule_data(st.session_state['edit_schedule'])
                if new_pdf_bytes:
                    st.session_state['edited_pdf_bytes'] = new_pdf_bytes
                    # PDF → 이미지 변환 (pdf2image 있으면)
                    st.session_state['edited_pdf_images'] = None
                    if PDF_TO_IMAGE_AVAILABLE:
                        try:
                            import io, tempfile
                            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                                tmp.write(new_pdf_bytes)
                                tmp_path = tmp.name
                            images = convert_from_path(tmp_path, dpi=200)
                            imgs_bytes = []
                            for img in images:
                                buf = io.BytesIO()
                                img.save(buf, format='PNG')
                                imgs_bytes.append(buf.getvalue())
                            st.session_state['edited_pdf_images'] = imgs_bytes
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                    # PDF 다운로드 키 갱신
                    prev_p = st.session_state.get('pdf_dl_key', 'pdf_dl_0')
                    st.session_state['pdf_dl_key'] = f'pdf_dl_{int(prev_p.split("_")[-1]) + 1}'
                    st.success("✅ Excel 및 PDF가 재생성되었습니다. 상단 버튼으로 다운로드하세요.")
                else:
                    st.warning(f"⚠️ PDF 재생성 실패: {pdf_err}. Excel만 재생성되었습니다.")
                    st.success("✅ Excel이 재생성되었습니다.")

                # 통계 재계산
                new_match_types, new_stats_df = _recalculate_stats(
                    st.session_state['edit_schedule'], result.get('player_genders', [])
                )
                st.session_state['edited_stats'] = {
                    'match_types': new_match_types,
                    'stats_df': new_stats_df,
                }
                st.rerun()

    # ── 탭3: 통계 ─────────────────────────────────────────
    with tab_stats:
        # 편집본 통계 우선, 없으면 원본
        edited_stats = st.session_state.get('edited_stats')
        if edited_stats:
            st.info("📝 편집된 스케줄 기준 통계입니다.")
            disp_match_types = edited_stats['match_types']
            disp_stats_df    = edited_stats['stats_df']
        else:
            disp_match_types = result['match_types']
            disp_stats_df    = result['stats_df']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("남복 경기", disp_match_types['남복'])
        with col2:
            st.metric("여복 경기", disp_match_types['여복'])
        with col3:
            st.metric("혼복 경기", disp_match_types['혼복'])
        st.subheader("선수별 참여 횟수")
        st.dataframe(disp_stats_df, use_container_width=True)


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
                    with col2:
                        time_slots = st.number_input(
                            "타임 수",
                            min_value=3,
                            max_value=10,
                            value=5,
                            step=1,
                            help="전체 타임 수를 설정합니다."
                        )
                    
                    # 매칭 실행 버튼
                    if st.button("🎾 매칭 생성 시작", type="primary"):
                        run_matching_algorithm(iterations=iterations, time_slots=time_slots)
                
            except Exception as e:
                st.error(f"데이터 확인 중 오류: {e}")
        else:
            st.error("참가자 데이터를 불러올 수 없습니다.")
        
        # 매칭 결과 표시 (session_state에서 렌더링 - 다운로드 버튼 클릭 후에도 유지됨)
        st.markdown("---")
        display_matching_result()


if __name__ == "__main__":
    main()
