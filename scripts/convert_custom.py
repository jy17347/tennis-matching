# -*- coding: utf-8 -*-
"""
Custom.xlsx를 tennis_matching.py 결과 형식으로 변환
- 엑셀: 매칭결과, 타임표, 참여통계, 전체요약 시트 생성
- PDF: tennis_matching.py와 동일한 형식의 타임표
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# PDF 생성 관련
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ Warning: reportlab이 설치되지 않았습니다. PDF 생성 기능이 비활성화됩니다.")


class CustomConverter:
    """custom.xlsx를 tennis_matching 형식으로 변환"""
    
    def __init__(self, custom_file='dataset/custom.xlsx'):
        self.custom_file = custom_file
        self.roster_info = {}
        self.participation_info = {}
        self.matches = []
        self.player_stats = {}
        
    def load_roster(self):
        """roster.xlsx에서 선수 정보 로드"""
        roster_path = os.path.join('dataset', 'roster.xlsx')
        if os.path.exists(roster_path):
            print(f"📂 선수 정보 로드: {roster_path}")
            df = pd.read_excel(roster_path)
            for _, row in df.iterrows():
                name = row['성명']
                self.roster_info[name] = {
                    'gender': 1 if row['성별'] == 1 else 2,  # 1=남, 2=여
                    'gender_str': '남' if row['성별'] == 1 else '여',
                    'skill': row['실력'] if pd.notna(row['실력']) and row['실력'] != 'N' else 3,
                    'number': row.get('번호', '')
                }
            print(f"✅ {len(self.roster_info)}명 선수 정보 로드 완료")
        else:
            print(f"⚠️  roster.xlsx 파일이 없습니다. 기본값 사용")
    
    def load_participation(self):
        """participation.xlsx에서 참여 선수 로드"""
        participation_path = 'dataset/participation.xlsx'
        if os.path.exists(participation_path):
            print(f"📂 참여 선수 로드: {participation_path}")
            try:
                df = pd.read_excel(participation_path)  # 첫 번째 시트 사용
                for _, row in df.iterrows():
                    name = row.get('성명', '')
                    if name and row.get('참여', '') == 'O':
                        self.participation_info[name] = True
                print(f"✅ {len(self.participation_info)}명 참여 선수 확인")
            except Exception as e:
                print(f"⚠️  participation.xlsx 로드 실패: {e}")
        else:
            print(f"⚠️  participation.xlsx 파일이 없습니다.")
    
    def load_custom_matches(self):
        """custom.xlsx에서 매칭 데이터 로드"""
        print(f"\n📂 매칭 데이터 로드: {self.custom_file}")
        df = pd.read_excel(self.custom_file, sheet_name='매칭결과')
        
        # 선수 초기화
        all_players = set()
        for _, row in df.iterrows():
            for col in ['팀1_선수1', '팀1_선수2', '팀2_선수1', '팀2_선수2']:
                player = str(row[col]).strip()
                if player and player != 'nan':
                    all_players.add(player)
        
        for player in all_players:
            info = self.roster_info.get(player, {
                'gender': 1,
                'gender_str': '남',
                'skill': 3,
                'number': ''
            })
            self.player_stats[player] = {
                'gender': info['gender'],
                'gender_str': info['gender_str'],
                'skill': info['skill'],
                'matches_played': 0,
                'mixed_matches': 0,
                'same_doubles': 0,
                'match_list': []
            }
        
        # 경기 파싱
        for _, row in df.iterrows():
            team1 = [str(row['팀1_선수1']).strip(), str(row['팀1_선수2']).strip()]
            team2 = [str(row['팀2_선수1']).strip(), str(row['팀2_선수2']).strip()]
            
            # 경기 타입 결정
            team1_genders = [self.player_stats[p]['gender'] for p in team1]
            team2_genders = [self.player_stats[p]['gender'] for p in team2]
            all_genders = team1_genders + team2_genders
            
            if all_genders.count(1) == 4:
                match_type = '남복'
            elif all_genders.count(2) == 4:
                match_type = '여복'
            else:
                match_type = '혼복'
            
            # 팀 평균 실력 계산
            team1_skills = [self.player_stats[p]['skill'] for p in team1]
            team2_skills = [self.player_stats[p]['skill'] for p in team2]
            team1_avg = sum(team1_skills) / 2
            team2_avg = sum(team2_skills) / 2
            skill_diff = abs(team1_avg - team2_avg)
            
            # 상위/하위 선수 실력차
            if match_type in ['남복', '여복']:
                team1_sorted = sorted(team1_skills)
                team2_sorted = sorted(team2_skills)
                top_diff = abs(team1_sorted[0] - team2_sorted[0])
                bottom_diff = abs(team1_sorted[1] - team2_sorted[1])
            elif match_type == '혼복':
                # 남자끼리, 여자끼리 비교
                team1_male_skill = [self.player_stats[p]['skill'] for p in team1 if self.player_stats[p]['gender'] == 1][0]
                team1_female_skill = [self.player_stats[p]['skill'] for p in team1 if self.player_stats[p]['gender'] == 2][0]
                team2_male_skill = [self.player_stats[p]['skill'] for p in team2 if self.player_stats[p]['gender'] == 1][0]
                team2_female_skill = [self.player_stats[p]['skill'] for p in team2 if self.player_stats[p]['gender'] == 2][0]
                top_diff = abs(team1_male_skill - team2_male_skill)
                bottom_diff = abs(team1_female_skill - team2_female_skill)
            else:
                top_diff = 0
                bottom_diff = 0
            
            match = {
                'court': int(row['코트']),
                'time': int(row['타임']),
                'type': match_type,
                'team1': team1,
                'team2': team2,
                'team1_avg': team1_avg,
                'team2_avg': team2_avg,
                'skill_diff': skill_diff,
                'top_diff': top_diff,
                'bottom_diff': bottom_diff
            }
            self.matches.append(match)
            
            # 선수 통계 업데이트
            for player in team1 + team2:
                self.player_stats[player]['matches_played'] += 1
                self.player_stats[player]['match_list'].append(match)
                if match_type == '혼복':
                    self.player_stats[player]['mixed_matches'] += 1
                else:
                    self.player_stats[player]['same_doubles'] += 1
        
        print(f"✅ {len(self.matches)}개 경기, {len(self.player_stats)}명 선수 파싱 완료")
    
    def create_excel(self, output_path):
        """tennis_matching.py 형식의 엑셀 생성"""
        print(f"\n📝 엑셀 파일 생성 중...")
        
        # 1. 매칭결과 시트
        match_data = []
        for match in sorted(self.matches, key=lambda m: (m['time'], m['court'])):
            match_data.append({
                '코트': match['court'],
                '타임': match['time'],
                '경기타입': match['type'],
                '팀1_선수1': match['team1'][0],
                '팀1_선수2': match['team1'][1],
                '팀1_평균실력': round(match['team1_avg'], 1),
                '팀2_선수1': match['team2'][0],
                '팀2_선수2': match['team2'][1],
                '팀2_평균실력': round(match['team2_avg'], 1),
                '팀평균_실력차': round(match['skill_diff'], 1),
                '상위선수_실력차': int(match['top_diff']),
                '하위선수_실력차': int(match['bottom_diff'])
            })
        df_matches = pd.DataFrame(match_data)
        
        # 2. 타임표 시트
        time_slots = sorted(set(m['time'] for m in self.matches))
        courts = sorted(set(m['court'] for m in self.matches))
        
        timetable_data = []
        for time in time_slots:
            row = {'타임': time}
            time_matches = [m for m in self.matches if m['time'] == time]
            for court in courts:
                court_match = next((m for m in time_matches if m['court'] == court), None)
                if court_match:
                    t1 = f"{court_match['team1'][0]} & {court_match['team1'][1]}"
                    t2 = f"{court_match['team2'][0]} & {court_match['team2'][1]}"
                    row[f'코트{court}'] = f"[{court_match['type']}]\n{t1}\nvs\n{t2}"
                else:
                    row[f'코트{court}'] = "-"
            timetable_data.append(row)
        df_timetable = pd.DataFrame(timetable_data)
        
        # 3. 참여통계 시트
        stats_data = []
        for name in sorted(self.player_stats.keys(), 
                          key=lambda x: (-self.player_stats[x]['matches_played'], 
                                        self.player_stats[x]['gender'], 
                                        self.player_stats[x]['skill'])):
            p = self.player_stats[name]
            if p['matches_played'] > 0:
                stats_row = {
                    '성명': name,
                    '성별': p['gender_str'],
                    '실력': p['skill'],
                    '참여횟수': p['matches_played']
                }
                
                # 남복/여복 컬럼
                if p['gender'] == 1:  # 남자
                    stats_row['남복'] = p['same_doubles'] if p['same_doubles'] > 0 else '-'
                    stats_row['혼복'] = p['mixed_matches'] if p['mixed_matches'] > 0 else '-'
                    stats_row['여복'] = None
                else:  # 여자
                    stats_row['남복'] = None
                    stats_row['혼복'] = p['mixed_matches'] if p['mixed_matches'] > 0 else '-'
                    stats_row['여복'] = p['same_doubles'] if p['same_doubles'] > 0 else '-'
                
                stats_data.append(stats_row)
        df_stats = pd.DataFrame(stats_data)
        
        # NaN 처리 (None을 빈 문자열로)
        df_stats = df_stats.fillna('')
        
        # 4. 전체요약 시트
        participations = [p['matches_played'] for p in self.player_stats.values() if p['matches_played'] > 0]
        skill_diffs = [m['skill_diff'] for m in self.matches]
        top_diffs = [m['top_diff'] for m in self.matches]
        bottom_diffs = [m['bottom_diff'] for m in self.matches]
        
        male_count = len([p for p in self.player_stats.values() if p['gender'] == 1 and p['matches_played'] > 0])
        female_count = len([p for p in self.player_stats.values() if p['gender'] == 2 and p['matches_played'] > 0])
        
        summary_data = [
            {'항목': '총 경기 수', '값': len(self.matches)},
            {'항목': '남복 경기 수', '값': len([m for m in self.matches if m['type'] == '남복'])},
            {'항목': '여복 경기 수', '값': len([m for m in self.matches if m['type'] == '여복'])},
            {'항목': '혼복 경기 수', '값': len([m for m in self.matches if m['type'] == '혼복'])},
            {'항목': '총 참가자 수', '값': len([p for p in self.player_stats.values() if p['matches_played'] > 0])},
            {'항목': '남자 참가자', '값': male_count},
            {'항목': '여자 참가자', '값': female_count},
            {'항목': '평균 참여 횟수', '값': round(np.mean(participations), 2) if participations else 0},
            {'항목': '최대 참여 횟수', '값': max(participations) if participations else 0},
            {'항목': '최소 참여 횟수', '값': min(participations) if participations else 0},
            {'항목': '평균 팀간 실력차', '값': round(np.mean(skill_diffs), 2) if skill_diffs else 0},
            {'항목': '평균 상위선수 실력차', '값': round(np.mean(top_diffs), 2) if top_diffs else 0},
            {'항목': '평균 하위선수 실력차', '값': round(np.mean(bottom_diffs), 2) if bottom_diffs else 0},
        ]
        df_summary = pd.DataFrame(summary_data)
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_matches.to_excel(writer, sheet_name='매칭결과', index=False)
            df_timetable.to_excel(writer, sheet_name='타임표', index=False)
            df_stats.to_excel(writer, sheet_name='참여통계', index=False)
            df_summary.to_excel(writer, sheet_name='전체요약', index=False)
        
        print(f"✅ 엑셀 저장 완료: {output_path}")
        return output_path
    
    def create_pdf(self, output_path):
        """tennis_matching.py 형식의 PDF 생성"""
        if not PDF_AVAILABLE:
            print("❌ PDF 생성 불가: reportlab이 설치되지 않았습니다.")
            return None
        
        print(f"\n📄 PDF 생성 중...")
        
        try:
            # 한글 폰트 등록
            font_registered = False
            font_paths = [
                'C:/Windows/Fonts/malgun.ttf',
                'C:/Windows/Fonts/NanumGothic.ttf',
                'C:/Windows/Fonts/gulim.ttc',
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('Korean', font_path))
                        font_registered = True
                        break
                    except:
                        continue
            
            korean_font = 'Korean' if font_registered else 'Helvetica'
            
            # PDF 문서 생성 (A4 가로) - tennis_matching.py와 동일
            doc = SimpleDocTemplate(
                output_path,
                pagesize=landscape(A4),
                rightMargin=1*cm,
                leftMargin=1*cm,
                topMargin=1*cm,
                bottomMargin=1*cm
            )
            
            elements = []
            styles = getSampleStyleSheet()
            
            # 스타일 정의 - tennis_matching.py와 동일
            title_style = ParagraphStyle('TitleKorean', 
                                        parent=styles['Title'],
                                        fontName=korean_font, 
                                        fontSize=20, 
                                        alignment=1)
            normal_style = ParagraphStyle('NormalKorean', 
                                         parent=styles['Normal'],
                                         fontName=korean_font, 
                                         fontSize=10)
            
            # 제목 및 날짜
            elements.append(Paragraph("테니스 타임표", title_style))
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph(f"생성일: {datetime.now().strftime('%Y년 %m월 %d일')}", normal_style))
            elements.append(Spacer(1, 0.5*cm))
            
            # 타임표 테이블 생성
            time_slots = sorted(set(m['time'] for m in self.matches))
            courts = sorted(set(m['court'] for m in self.matches))
            
            table_data = [['타임', '코트 1', '코트 2', '코트 3']]
            
            for time_slot in time_slots:
                row = [f'{time_slot}']
                time_matches = [m for m in self.matches if m['time'] == time_slot]
                for court in courts:
                    court_match = next((m for m in time_matches if m['court'] == court), None)
                    if court_match:
                        t1 = f"{court_match['team1'][0]} & {court_match['team1'][1]}"
                        t2 = f"{court_match['team2'][0]} & {court_match['team2'][1]}"
                        row.append(f"[{court_match['type']}]\n{t1}\nvs\n{t2}")
                    else:
                        row.append("-")
                table_data.append(row)
            
            # 테이블 생성
            table = Table(table_data, colWidths=[2*cm, 7*cm, 7*cm, 7*cm])
            
            # 기본 테이블 스타일 - tennis_matching.py와 동일
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TOPPADDING', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#D6DCE5')),
            ])
            
            # 경기 타입별 배경색 적용 - tennis_matching.py와 동일
            for i, time_slot in enumerate(time_slots, start=1):
                time_matches = [m for m in self.matches if m['time'] == time_slot]
                for court in courts:
                    court_match = next((m for m in time_matches if m['court'] == court), None)
                    if court_match:
                        if court_match['type'] == '남복':
                            bg = colors.HexColor('#DDEBF7')
                        elif court_match['type'] == '여복':
                            bg = colors.HexColor('#FCE4D6')
                        else:  # 혼복
                            bg = colors.HexColor('#E2EFDA')
                        table_style.add('BACKGROUND', (court, i), (court, i), bg)
            
            table.setStyle(table_style)
            elements.append(table)
            
            # 범례
            elements.append(Spacer(1, 0.5*cm))
            legend = Table([['범례:', '남복', '여복', '혼복']], 
                          colWidths=[2*cm, 4*cm, 4*cm, 4*cm])
            legend.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#DDEBF7')),
                ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#FCE4D6')),
                ('BACKGROUND', (3, 0), (3, 0), colors.HexColor('#E2EFDA')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOX', (1, 0), (1, 0), 1, colors.black),
                ('BOX', (2, 0), (2, 0), 1, colors.black),
                ('BOX', (3, 0), (3, 0), 1, colors.black),
            ]))
            elements.append(legend)
            
            # 요약 정보
            elements.append(Spacer(1, 0.5*cm))
            male_m = len([m for m in self.matches if m['type'] == '남복'])
            female_m = len([m for m in self.matches if m['type'] == '여복'])
            mixed_m = len([m for m in self.matches if m['type'] == '혼복'])
            
            parts = [p['matches_played'] for p in self.player_stats.values() if p['matches_played'] > 0]
            diffs = [m['skill_diff'] for m in self.matches]
            
            male_players = len([p for p in self.player_stats.values() if p['gender'] == 1 and p['matches_played'] > 0])
            female_players = len([p for p in self.player_stats.values() if p['gender'] == 2 and p['matches_played'] > 0])
            
            summary = f"""
            총 경기: {len(self.matches)}경기 (남복 {male_m}, 여복 {female_m}, 혼복 {mixed_m})<br/>
            참가자: 남자 {male_players}명, 여자 {female_players}명<br/>
            참여 횟수: 최소 {min(parts) if parts else 0}회 ~ 최대 {max(parts) if parts else 0}회<br/>
            평균 팀간 실력차: {np.mean(diffs):.2f}
            """
            elements.append(Paragraph(summary, normal_style))
            
            # PDF 빌드
            doc.build(elements)
            
            print(f"✅ PDF 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convert(self):
        """전체 변환 프로세스"""
        print("="*60)
        print("🎾 Custom.xlsx → Tennis Matching 형식 변환기")
        print("="*60)
        
        # 1. 데이터 로드
        self.load_roster()
        self.load_participation()
        self.load_custom_matches()
        
        # 2. 타임스탬프 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 3. results 폴더 생성
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 4. 엑셀 생성
        excel_path = os.path.join(results_dir, f'테니스_매칭결과_{timestamp}.xlsx')
        self.create_excel(excel_path)
        
        # 5. PDF 생성
        pdf_path = os.path.join(results_dir, f'테니스_타임표_{timestamp}.pdf')
        self.create_pdf(pdf_path)
        self.create_pdf("C:/project/matching/테니스_타임표.pdf")

        print("\n" + "="*60)
        print("✅ 변환 완료!")
        print(f"   📊 엑셀: {excel_path}")
        print(f"   📄 PDF: {pdf_path}")
        print("="*60)


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom.xlsx를 tennis_matching 형식으로 변환')
    parser.add_argument('--input', '-i', default='dataset/custom.xlsx', help='입력 파일 (기본: dataset/custom.xlsx)')
    parser.add_argument('--no-pdf', action='store_true', help='PDF 생성 스킵')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 파일이 존재하지 않습니다: {args.input}")
        return
    
    converter = CustomConverter(args.input)
    converter.convert()


if __name__ == '__main__':
    main()
