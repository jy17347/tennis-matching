# -*- coding: utf-8 -*-
"""
테니스 매칭 결과 시각화 및 검증 도구
- 기존 매칭 결과 엑셀 파일을 읽어서 검증
- 시각화된 PDF 타임표 생성
- 제약조건 위반 사항 체크
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
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


class MatchingVisualizer:
    """매칭 결과 시각화 및 검증 클래스"""
    
    def __init__(self, excel_file_path):
        """
        Args:
            excel_file_path: 매칭 결과 엑셀 파일 경로
        """
        self.excel_file = excel_file_path
        self.matches_df = None
        self.timetable_df = None
        self.stats_df = None
        self.summary_df = None
        self.players = {}
        self.matches = []
        self.validation_results = {}
        
    def load_excel(self):
        """엑셀 파일 로드"""
        print(f"📂 엑셀 파일 로드 중: {os.path.basename(self.excel_file)}")
        
        try:
            xl = pd.ExcelFile(self.excel_file)
            
            # 시트 이름 확인
            if '매칭결과' not in xl.sheet_names:
                print(f"❌ 필수 시트 '매칭결과'가 없습니다.")
                return False
            
            # 매칭결과 시트 로드
            self.matches_df = pd.read_excel(self.excel_file, sheet_name='매칭결과')
            
            # 선택적 시트 로드
            if '타임표' in xl.sheet_names:
                self.timetable_df = pd.read_excel(self.excel_file, sheet_name='타임표')
            
            if '참여통계' in xl.sheet_names:
                self.stats_df = pd.read_excel(self.excel_file, sheet_name='참여통계')
            
            if '전체요약' in xl.sheet_names:
                self.summary_df = pd.read_excel(self.excel_file, sheet_name='전체요약')
            
            print(f"✅ 로드 완료: {len(self.matches_df)}개 경기")
            return True
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return False
    
    def parse_matches(self):
        """매칭 데이터 파싱"""
        print("\n🔍 매칭 데이터 파싱 중...")
        
        try:
            # roster.xlsx에서 선수 정보 로드
            roster_path = os.path.join(os.path.dirname(self.excel_file), 'dataset', 'roster.xlsx')
            if os.path.exists(roster_path):
                roster_df = pd.read_excel(roster_path)
                roster_info = {}
                for _, row in roster_df.iterrows():
                    name = row['성명']
                    roster_info[name] = {
                        'gender': '남' if row['성별'] == 1 else '여',
                        'skill': row['실력'] if pd.notna(row['실력']) else 'N'
                    }
            else:
                roster_info = {}
            
            # 참여통계 시트가 있으면 사용, 없으면 매칭결과에서 추출
            if self.stats_df is not None:
                for _, row in self.stats_df.iterrows():
                    name = row['성명']
                    if pd.notna(name):
                        gender_str = str(row.get('성별', ''))
                        is_male = '남' in gender_str
                        
                        male_games = row.get('남복', 0)
                        female_games = row.get('여복', 0)
                        mixed_games = row.get('혼복', 0)
                        
                        # NaN 처리
                        male_games = 0 if pd.isna(male_games) else int(male_games)
                        female_games = 0 if pd.isna(female_games) else int(female_games)
                        mixed_games = 0 if pd.isna(mixed_games) else int(mixed_games)
                        
                        self.players[name] = {
                            'name': name,
                            'gender': '남' if is_male else '여',
                            'skill': row.get('실력', 'N'),
                            'total_games': row.get('참여횟수', 0),
                            'mixed_games': mixed_games,
                            'same_gender_games': male_games + female_games if is_male else female_games,
                            'matches': []
                        }
            else:
                # 매칭결과에서 선수 정보 추출
                all_players = set()
                for _, row in self.matches_df.iterrows():
                    for col in ['팀1_선수1', '팀1_선수2', '팀2_선수1', '팀2_선수2']:
                        player = str(row[col]).strip()
                        if player and player != 'nan':
                            all_players.add(player)
                
                for name in all_players:
                    info = roster_info.get(name, {'gender': '남', 'skill': 'N'})
                    self.players[name] = {
                        'name': name,
                        'gender': info['gender'],
                        'skill': info['skill'],
                        'total_games': 0,
                        'mixed_games': 0,
                        'same_gender_games': 0,
                        'matches': []
                    }
            
            # 경기 정보 파싱
            for _, row in self.matches_df.iterrows():
                # 경기 타입 결정
                team1 = [str(row['팀1_선수1']).strip(), str(row['팀1_선수2']).strip()]
                team2 = [str(row['팀2_선수1']).strip(), str(row['팀2_선수2']).strip()]
                
                # 경기 타입 추론
                if '경기타입' in row and pd.notna(row['경기타입']):
                    match_type = row['경기타입']
                else:
                    # 선수 성별로 경기 타입 추론
                    team1_genders = [self.players.get(p, {}).get('gender', '남') for p in team1]
                    team2_genders = [self.players.get(p, {}).get('gender', '남') for p in team2]
                    all_genders = team1_genders + team2_genders
                    
                    if all_genders.count('남') == 4:
                        match_type = '남복'
                    elif all_genders.count('여') == 4:
                        match_type = '여복'
                    else:
                        match_type = '혼복'
                
                # 실력 정보
                if '팀1_평균실력' in row:
                    team1_skill = row.get('팀1_평균실력', 0)
                    team2_skill = row.get('팀2_평균실력', 0)
                    skill_diff = row.get('팀평균_실력차', 0)
                else:
                    # 실력 계산
                    team1_skills = [self.players.get(p, {}).get('skill', 3) for p in team1]
                    team2_skills = [self.players.get(p, {}).get('skill', 3) for p in team2]
                    team1_skill = sum([s if isinstance(s, (int, float)) else 3 for s in team1_skills]) / 2
                    team2_skill = sum([s if isinstance(s, (int, float)) else 3 for s in team2_skills]) / 2
                    skill_diff = abs(team1_skill - team2_skill)
                
                match_info = {
                    'time': row['타임'],
                    'court': row['코트'],
                    'type': match_type,
                    'team1': team1,
                    'team2': team2,
                    'team1_skill': team1_skill,
                    'team2_skill': team2_skill,
                    'skill_diff': skill_diff
                }
                self.matches.append(match_info)
                
                # 선수별 경기 기록
                for player_name in team1 + team2:
                    if player_name in self.players:
                        self.players[player_name]['matches'].append(match_info)
                        self.players[player_name]['total_games'] += 1
                        
                        if match_type == '혼복':
                            self.players[player_name]['mixed_games'] += 1
                        else:
                            self.players[player_name]['same_gender_games'] += 1
            
            print(f"✅ 파싱 완료: {len(self.players)}명 선수, {len(self.matches)}개 경기")
            return True
            
        except Exception as e:
            print(f"❌ 파싱 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_constraints(self):
        """제약조건 검증"""
        print("\n📋 제약조건 검증 중...")
        
        violations = []
        warnings = []
        
        # 1. 총 경기 수 확인 (15경기)
        total_matches = len(self.matches)
        if total_matches != 15:
            violations.append(f"❌ 총 경기 수: {total_matches}경기 (기대값: 15경기)")
        else:
            print(f"✅ 총 경기 수: 15경기")
        
        # 2. 경기 타입 분포
        match_types = defaultdict(int)
        for match in self.matches:
            match_types[match['type']] += 1
        
        print(f"   경기 구성: 남복 {match_types.get('남복', 0)}, 여복 {match_types.get('여복', 0)}, 혼복 {match_types.get('혼복', 0)}")
        
        # 3. 타임슬롯당 최대 12명 (3코트 × 4명)
        time_players = defaultdict(set)
        for match in self.matches:
            time = match['time']
            for player in match['team1'] + match['team2']:
                time_players[time].add(player)
        
        for time, players in time_players.items():
            if len(players) > 12:
                violations.append(f"❌ 타임 {time}: {len(players)}명 참여 (최대 12명)")
        
        print(f"✅ 타임슬롯별 인원: {', '.join([f'T{t}={len(p)}명' for t, p in sorted(time_players.items())])}")
        
        # 4. 선수별 참여 횟수 균형
        participation_counts = [p['total_games'] for p in self.players.values()]
        if participation_counts:
            min_games = min(participation_counts)
            max_games = max(participation_counts)
            avg_games = sum(participation_counts) / len(participation_counts)
            diff = max_games - min_games
            
            print(f"   참여 횟수: 최소 {min_games}, 최대 {max_games}, 평균 {avg_games:.1f}, 차이 {diff}")
            
            if diff > 2:
                warnings.append(f"⚠️  참여 횟수 차이가 큽니다: {diff}경기 (권장: 2경기 이하)")
        
        # 5. 혼복 참여 확인 (모든 선수 최소 1회)
        zero_mixed = [name for name, info in self.players.items() if info['mixed_games'] == 0]
        if zero_mixed:
            violations.append(f"❌ 혼복 미참여 선수: {', '.join(zero_mixed)} ({len(zero_mixed)}명)")
        else:
            print(f"✅ 모든 선수 혼복 참여")
        
        # 6. 실력 밸런스 확인
        skill_diffs = [m['skill_diff'] for m in self.matches if pd.notna(m['skill_diff'])]
        if skill_diffs:
            avg_diff = sum(skill_diffs) / len(skill_diffs)
            max_diff = max(skill_diffs)
            print(f"   실력 밸런스: 평균 차이 {avg_diff:.2f}, 최대 차이 {max_diff:.2f}")
            
            if avg_diff > 0.5:
                warnings.append(f"⚠️  실력 불균형: 평균 차이 {avg_diff:.2f} (권장: 0.5 이하)")
        
        # 7. 연속 경기 확인
        consecutive_violations = []
        for name, info in self.players.items():
            times = sorted([m['time'] for m in info['matches']])
            for i in range(len(times) - 1):
                if times[i+1] - times[i] == 1:
                    consecutive_violations.append(f"{name} (T{times[i]}→T{times[i+1]})")
        
        if consecutive_violations:
            warnings.append(f"⚠️  연속 경기: {', '.join(consecutive_violations[:5])}" + 
                          (f" 외 {len(consecutive_violations)-5}건" if len(consecutive_violations) > 5 else ""))
        else:
            print(f"✅ 연속 경기 없음")
        
        # 8. 같은 팀/상대 중복 확인
        team_duplicates = []
        for name, info in self.players.items():
            teammates = defaultdict(int)
            opponents = defaultdict(int)
            
            for match in info['matches']:
                if name in match['team1']:
                    team = match['team1']
                    opp_team = match['team2']
                else:
                    team = match['team2']
                    opp_team = match['team1']
                
                for teammate in team:
                    if teammate != name:
                        teammates[teammate] += 1
                
                for opponent in opp_team:
                    opponents[opponent] += 1
            
            # 같은 팀 3회 이상
            for teammate, count in teammates.items():
                if count >= 3:
                    team_duplicates.append(f"{name}-{teammate} ({count}회)")
            
            # 같은 상대 3회 이상
            for opponent, count in opponents.items():
                if count >= 3:
                    team_duplicates.append(f"{name}vs{opponent} ({count}회)")
        
        if team_duplicates:
            warnings.append(f"⚠️  중복 팀/상대: {', '.join(team_duplicates[:3])}" +
                          (f" 외 {len(team_duplicates)-3}건" if len(team_duplicates) > 3 else ""))
        
        # 결과 저장
        self.validation_results = {
            'violations': violations,
            'warnings': warnings,
            'total_matches': total_matches,
            'match_types': dict(match_types),
            'participation': {
                'min': min(participation_counts) if participation_counts else 0,
                'max': max(participation_counts) if participation_counts else 0,
                'avg': sum(participation_counts) / len(participation_counts) if participation_counts else 0
            },
            'zero_mixed': zero_mixed
        }
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 검증 결과 요약")
        print("="*60)
        
        if violations:
            print(f"\n❌ 제약조건 위반: {len(violations)}건")
            for v in violations:
                print(f"   {v}")
        else:
            print("\n✅ 모든 필수 제약조건 충족")
        
        if warnings:
            print(f"\n⚠️  주의사항: {len(warnings)}건")
            for w in warnings:
                print(f"   {w}")
        else:
            print("\n✅ 주의사항 없음")
        
        print("="*60 + "\n")
        
        return len(violations) == 0
    
    def display_summary(self):
        """매칭 결과 요약 출력"""
        print("\n" + "="*60)
        print("📈 매칭 결과 상세 정보")
        print("="*60)
        
        # 경기 목록
        print(f"\n🎾 경기 목록 ({len(self.matches)}경기)")
        print("-" * 60)
        
        for i, match in enumerate(self.matches, 1):
            team1_str = f"{match['team1'][0]}, {match['team1'][1]}"
            team2_str = f"{match['team2'][0]}, {match['team2'][1]}"
            print(f"{i:2d}. T{match['time']} C{match['court']} [{match['type']:^4}] "
                  f"{team1_str:25} vs {team2_str:25} "
                  f"(실력차: {match['skill_diff']:.2f})")
        
        # 선수별 통계
        print(f"\n👥 선수별 참여 통계 ({len(self.players)}명)")
        print("-" * 60)
        
        # 참여 횟수 순으로 정렬
        sorted_players = sorted(self.players.items(), 
                               key=lambda x: (-x[1]['total_games'], x[0]))
        
        for name, info in sorted_players:
            times = sorted([m['time'] for m in info['matches']])
            times_str = ', '.join([f"T{t}" for t in times])
            print(f"{name:10} [{info['gender']}] "
                  f"총 {info['total_games']}경기 "
                  f"(혼복 {info['mixed_games']}, "
                  f"{'남' if info['gender']=='남' else '여'}복 {info['same_gender_games']}) "
                  f"참여타임: {times_str}")
        
        print("="*60 + "\n")
    
    def generate_pdf(self, output_path=None):
        """PDF 타임표 생성"""
        if not PDF_AVAILABLE:
            print("❌ PDF 생성 불가: reportlab이 설치되지 않았습니다.")
            return None
        
        if output_path is None:
            # 원본 파일명 기반으로 PDF 파일명 생성
            base_name = os.path.splitext(os.path.basename(self.excel_file))[0]
            
            # custom.xlsx인 경우 custom_타임표.pdf로, 아니면 기존 로직 사용
            if base_name == 'custom':
                base_name = 'custom_타임표'
            else:
                base_name = base_name.replace('매칭결과', '타임표')
            
            output_dir = os.path.dirname(self.excel_file)
            output_path = os.path.join(output_dir, f"{base_name}.pdf")
        
        print(f"\n📄 PDF 생성 중: {os.path.basename(output_path)}")
        
        try:
            # 한글 폰트 등록
            font_registered = False
            try:
                font_path = "C:\\Windows\\Fonts\\malgun.ttf"
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('Malgun', font_path))
                    font_registered = True
            except:
                pass
            
            if not font_registered:
                print("⚠️  한글 폰트 등록 실패, 기본 폰트 사용")
            
            # PDF 문서 생성 (A4 가로)
            doc = SimpleDocTemplate(
                output_path,
                pagesize=landscape(A4),
                rightMargin=15*mm,
                leftMargin=15*mm,
                topMargin=15*mm,
                bottomMargin=15*mm
            )
            
            story = []
            font_name = 'Malgun' if font_registered else 'Helvetica'
            
            # 제목
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=getSampleStyleSheet()['Heading1'],
                fontName=font_name,
                fontSize=20,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=10*mm,
                alignment=1  # 중앙 정렬
            )
            
            title = Paragraph(f"🎾 테니스 토너먼트 타임표", title_style)
            story.append(title)
            
            # 요약 정보
            summary_style = ParagraphStyle(
                'Summary',
                parent=getSampleStyleSheet()['Normal'],
                fontName=font_name,
                fontSize=10,
                textColor=colors.HexColor('#34495E'),
                spaceAfter=5*mm
            )
            
            match_types = self.validation_results.get('match_types', {})
            summary_text = (
                f"총 {self.validation_results.get('total_matches', 0)}경기 | "
                f"남복 {match_types.get('남복', 0)} | "
                f"여복 {match_types.get('여복', 0)} | "
                f"혼복 {match_types.get('혼복', 0)} | "
                f"참여 선수: {len(self.players)}명"
            )
            
            summary = Paragraph(summary_text, summary_style)
            story.append(summary)
            story.append(Spacer(1, 5*mm))
            
            # 타임표 테이블 생성
            # 헤더
            table_data = [['타임', '코트1', '코트2', '코트3']]
            
            # 타임별로 그룹화
            time_matches = defaultdict(list)
            for match in self.matches:
                time_matches[match['time']].append(match)
            
            # 각 타임별 데이터
            for time in sorted(time_matches.keys()):
                row = [f"타임 {time}"]
                matches = sorted(time_matches[time], key=lambda x: x['court'])
                
                court_data = [''] * 3  # 3개 코트
                for match in matches:
                    court_idx = match['court'] - 1
                    if 0 <= court_idx < 3:
                        type_icon = {'남복': '👨‍👨', '여복': '👩‍👩', '혼복': '👫'}.get(match['type'], '')
                        team1 = f"{match['team1'][0]}\n{match['team1'][1]}"
                        team2 = f"{match['team2'][0]}\n{match['team2'][1]}"
                        court_data[court_idx] = f"{type_icon} {match['type']}\n{team1}\nvs\n{team2}"
                
                row.extend(court_data)
                table_data.append(row)
            
            # 테이블 스타일
            table = Table(table_data, colWidths=[25*mm, 75*mm, 75*mm, 75*mm])
            
            table_style = TableStyle([
                # 헤더 스타일
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                
                # 타임 컬럼 스타일
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ECF0F1')),
                ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#2C3E50')),
                ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (0, -1), font_name),
                ('FONTSIZE', (0, 1), (0, -1), 11),
                ('FONTWEIGHT', (0, 1), (0, -1), 'BOLD'),
                
                # 경기 셀 스타일
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('VALIGN', (1, 1), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (1, 1), (-1, -1), font_name),
                ('FONTSIZE', (1, 1), (-1, -1), 9),
                
                # 테두리
                ('GRID', (0, 0), (-1, -1), 1.5, colors.HexColor('#BDC3C7')),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2C3E50')),
                
                # 행 높이
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                ('TOPPADDING', (1, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (1, 1), (-1, -1), 8),
            ])
            
            table.setStyle(table_style)
            story.append(table)
            
            # 페이지 나누기
            story.append(PageBreak())
            
            # 선수별 통계 페이지
            story.append(Paragraph("👥 선수별 참여 통계", title_style))
            story.append(Spacer(1, 5*mm))
            
            # 통계 테이블
            stats_data = [['선수명', '성별', '실력', '총참여', '혼복', '남/여복', '참여 타임']]
            
            sorted_players = sorted(self.players.items(), 
                                   key=lambda x: (-x[1]['total_games'], x[0]))
            
            for name, info in sorted_players:
                times = sorted([m['time'] for m in info['matches']])
                times_str = ', '.join([f"T{t}" for t in times])
                
                stats_data.append([
                    name,
                    info['gender'],
                    str(info['skill']),
                    str(info['total_games']),
                    str(info['mixed_games']),
                    str(info['same_gender_games']),
                    times_str
                ])
            
            stats_table = Table(stats_data, colWidths=[40*mm, 15*mm, 15*mm, 20*mm, 20*mm, 25*mm, 60*mm])
            
            stats_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#2C3E50')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ])
            
            stats_table.setStyle(stats_style)
            story.append(stats_table)
            
            # 검증 결과 추가
            if self.validation_results.get('violations') or self.validation_results.get('warnings'):
                story.append(Spacer(1, 10*mm))
                story.append(Paragraph("⚠️ 검증 결과", title_style))
                story.append(Spacer(1, 3*mm))
                
                validation_style = ParagraphStyle(
                    'Validation',
                    parent=getSampleStyleSheet()['Normal'],
                    fontName=font_name,
                    fontSize=9,
                    textColor=colors.HexColor('#E74C3C'),
                    leftIndent=10
                )
                
                for violation in self.validation_results.get('violations', []):
                    story.append(Paragraph(violation, validation_style))
                
                for warning in self.validation_results.get('warnings', []):
                    story.append(Paragraph(warning, validation_style))
            
            # PDF 빌드
            doc.build(story)
            
            print(f"✅ PDF 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self, generate_pdf=True, output_pdf_path=None):
        """전체 프로세스 실행"""
        print("="*60)
        print("🎾 테니스 매칭 결과 시각화 및 검증 도구")
        print("="*60)
        
        # 1. 엑셀 로드
        if not self.load_excel():
            return False
        
        # 2. 데이터 파싱
        if not self.parse_matches():
            return False
        
        # 3. 제약조건 검증
        self.validate_constraints()
        
        # 4. 요약 정보 출력
        self.display_summary()
        
        # 5. PDF 생성
        if generate_pdf:
            self.generate_pdf(output_pdf_path)
        
        return True


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='테니스 매칭 결과 시각화 및 검증')
    parser.add_argument('excel_file', nargs='?', help='매칭 결과 엑셀 파일 경로')
    parser.add_argument('--no-pdf', action='store_true', help='PDF 생성 스킵')
    parser.add_argument('--output', '-o', help='PDF 출력 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 경로 결정
    if args.excel_file:
        excel_file = args.excel_file
    else:
        # custom.xlsx 파일을 기본으로 사용
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_file = os.path.join(script_dir, 'custom.xlsx')
        
        if os.path.exists(excel_file):
            print(f"📂 기본 파일 사용: custom.xlsx\n")
        else:
            # custom.xlsx가 없으면 최신 파일 자동 선택
            results_dir = os.path.join(script_dir, 'results')
            if os.path.exists(results_dir):
                excel_files = [f for f in os.listdir(results_dir) 
                              if f.startswith('테니스_매칭결과_') and f.endswith('.xlsx')]
                if excel_files:
                    excel_files.sort(reverse=True)  # 최신 파일 먼저
                    excel_file = os.path.join(results_dir, excel_files[0])
                    print(f"📂 custom.xlsx가 없어 최신 파일 자동 선택: {excel_files[0]}\n")
                else:
                    print("❌ custom.xlsx와 results 폴더에 매칭 결과 파일이 없습니다.")
                    return
            else:
                print("❌ custom.xlsx 파일이 없습니다.")
                return
    
    # 파일 존재 확인
    if not os.path.exists(excel_file):
        print(f"❌ 파일이 존재하지 않습니다: {excel_file}")
        return
    
    # 시각화 도구 실행
    visualizer = MatchingVisualizer(excel_file)
    visualizer.run(generate_pdf=not args.no_pdf, output_pdf_path=args.output)


if __name__ == '__main__':
    main()
