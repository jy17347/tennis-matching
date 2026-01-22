# -*- coding: utf-8 -*-
"""
테니스 토너먼트 매칭 시스템 v2
- 3코트 × 5타임 = 15경기
- 남복, 여복, 혼복 경기
- 제약조건 기반 최적화
- 모든 남자 혼복 참여 보장
"""

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import random
from datetime import datetime
import os
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
    print("Warning: reportlab not installed. PDF export disabled.")


class Player:
    """선수 클래스"""
    def __init__(self, name, gender, skill, number):
        self.name = name
        self.gender = gender  # 1=남, 2=여
        self.skill = skill if skill not in ['N', 'n'] else 3  # N은 중간값 3으로 처리
        self.number = number
        self.matches_played = 0
        self.mixed_matches = 0  # 혼복 참여 횟수
        self.same_doubles = 0  # 남복/여복 참여 횟수
        self.teammates = defaultdict(int)  # 같은 팀으로 경기한 횟수
        self.opponents = defaultdict(int)  # 상대로 경기한 횟수
        self.last_time_played = -1  # 마지막으로 참여한 타임
        
    def __repr__(self):
        gender_str = "남" if self.gender == 1 else "여"
        return f"{self.name}({gender_str},{self.skill})"


class Match:
    """경기 클래스"""
    def __init__(self, time_slot, court, match_type, team1, team2):
        self.time_slot = time_slot
        self.court = court
        self.match_type = match_type  # '남복', '여복', '혼복'
        self.team1 = team1  # [player1, player2]
        self.team2 = team2  # [player1, player2]
        
    def get_all_players(self):
        return self.team1 + self.team2
    
    def get_team_avg_skill(self, team):
        return sum(p.skill for p in team) / len(team)
    
    def get_skill_diff(self):
        """팀 간 평균 실력 차이"""
        return abs(self.get_team_avg_skill(self.team1) - self.get_team_avg_skill(self.team2))
    
    def get_top_bottom_skill_diff(self):
        """상위/하위 선수 실력 차이"""
        if self.match_type in ['남복', '여복']:
            t1_sorted = sorted(self.team1, key=lambda p: p.skill)
            t2_sorted = sorted(self.team2, key=lambda p: p.skill)
            top_diff = abs(t1_sorted[0].skill - t2_sorted[0].skill)
            bottom_diff = abs(t1_sorted[1].skill - t2_sorted[1].skill)
            return top_diff, bottom_diff
        elif self.match_type == '혼복':
            t1_male = [p for p in self.team1 if p.gender == 1]
            t1_female = [p for p in self.team1 if p.gender == 2]
            t2_male = [p for p in self.team2 if p.gender == 1]
            t2_female = [p for p in self.team2 if p.gender == 2]
            
            if t1_male and t2_male and t1_female and t2_female:
                male_diff = abs(t1_male[0].skill - t2_male[0].skill)
                female_diff = abs(t1_female[0].skill - t2_female[0].skill)
                return male_diff, female_diff
        return 0, 0
    
    def is_one_sided(self):
        """한 팀이 상위/하위 모두 우위인지 확인"""
        if self.match_type in ['남복', '여복']:
            t1_sorted = sorted(self.team1, key=lambda p: p.skill)
            t2_sorted = sorted(self.team2, key=lambda p: p.skill)
            # t1이 상위/하위 모두 우위 (실력 낮을수록 좋음)
            t1_both_better = t1_sorted[0].skill < t2_sorted[0].skill and t1_sorted[1].skill < t2_sorted[1].skill
            # t2가 상위/하위 모두 우위
            t2_both_better = t2_sorted[0].skill < t1_sorted[0].skill and t2_sorted[1].skill < t1_sorted[1].skill
            return t1_both_better or t2_both_better
        elif self.match_type == '혼복':
            t1_male = [p for p in self.team1 if p.gender == 1]
            t1_female = [p for p in self.team1 if p.gender == 2]
            t2_male = [p for p in self.team2 if p.gender == 1]
            t2_female = [p for p in self.team2 if p.gender == 2]
            
            if t1_male and t2_male and t1_female and t2_female:
                t1_both_better = t1_male[0].skill < t2_male[0].skill and t1_female[0].skill < t2_female[0].skill
                t2_both_better = t2_male[0].skill < t1_male[0].skill and t2_female[0].skill < t1_female[0].skill
                return t1_both_better or t2_both_better
        return False
    
    def balance_teams(self):
        """
        한 팀이 상위/하위 모두 우위인 경우 하위선수를 교환하여 밸런스 조정
        팀 중복이나 상대 중복은 무시하고 실력 밸런스만 고려
        """
        if not self.is_one_sided():
            return False
        
        original_diff = self.get_skill_diff()
        
        if self.match_type in ['남복', '여복']:
            # 하위 선수(실력 높은 선수) 교환
            t1_sorted = sorted(self.team1, key=lambda p: p.skill)
            t2_sorted = sorted(self.team2, key=lambda p: p.skill)
            
            # 하위 선수 교환
            new_team1 = [t1_sorted[0], t2_sorted[1]]  # t1 상위 + t2 하위
            new_team2 = [t2_sorted[0], t1_sorted[1]]  # t2 상위 + t1 하위
            
            # 교환 후 실력차 계산
            new_diff = abs((new_team1[0].skill + new_team1[1].skill) / 2 - 
                          (new_team2[0].skill + new_team2[1].skill) / 2)
            
            if new_diff < original_diff:
                self.team1 = new_team1
                self.team2 = new_team2
                return True
                
        elif self.match_type == '혼복':
            t1_male = [p for p in self.team1 if p.gender == 1][0]
            t1_female = [p for p in self.team1 if p.gender == 2][0]
            t2_male = [p for p in self.team2 if p.gender == 1][0]
            t2_female = [p for p in self.team2 if p.gender == 2][0]
            
            # 어느 쪽이 우위인지 판단
            t1_better_male = t1_male.skill < t2_male.skill
            t1_better_female = t1_female.skill < t2_female.skill
            
            if t1_better_male and t1_better_female:
                # t1이 남녀 모두 우위 -> 여자 교환 (하위 역할)
                new_team1 = [t1_male, t2_female]
                new_team2 = [t2_male, t1_female]
            elif not t1_better_male and not t1_better_female:
                # t2가 남녀 모두 우위 -> 여자 교환
                new_team1 = [t1_male, t2_female]
                new_team2 = [t2_male, t1_female]
            else:
                return False
            
            # 교환 후 실력차 계산
            new_diff = abs((new_team1[0].skill + new_team1[1].skill) / 2 - 
                          (new_team2[0].skill + new_team2[1].skill) / 2)
            
            if new_diff < original_diff:
                self.team1 = new_team1
                self.team2 = new_team2
                return True
        
        return False
    
    def __repr__(self):
        t1 = f"{self.team1[0].name} & {self.team1[1].name}"
        t2 = f"{self.team2[0].name} & {self.team2[1].name}"
        return f"T{self.time_slot}C{self.court} [{self.match_type}] {t1} vs {t2}"


class TennisMatchingSystem:
    """테니스 매칭 시스템"""
    
    def __init__(self, roster_path, participation_path):
        self.roster_path = roster_path
        self.participation_path = participation_path
        self.players = []
        self.male_players = []
        self.female_players = []
        self.schedule = []  # 전체 경기 스케줄
        self.time_slots = 5
        self.courts = 3
        self.total_matches = self.time_slots * self.courts
        
        self.load_data()
        
    def load_data(self):
        """데이터 로드"""
        roster = pd.read_excel(self.roster_path, engine='openpyxl')
        participation = pd.read_excel(self.participation_path, engine='openpyxl')
        
        # 참가자 필터링
        participants = participation[participation['참여 (1)'].isin(['O', '1', 1])]['성명'].tolist()
        
        for _, row in roster.iterrows():
            if row['성명'] in participants:
                player = Player(
                    name=row['성명'],
                    gender=row['성별'],
                    skill=row['실력'],
                    number=row['번호']
                )
                self.players.append(player)
                if player.gender == 1:
                    self.male_players.append(player)
                else:
                    self.female_players.append(player)
        
        print(f"총 참가자: {len(self.players)}명 (남: {len(self.male_players)}, 여: {len(self.female_players)})")
        
    def reset_player_stats(self):
        """선수 통계 초기화"""
        for p in self.players:
            p.matches_played = 0
            p.mixed_matches = 0
            p.same_doubles = 0
            p.teammates = defaultdict(int)
            p.opponents = defaultdict(int)
            p.last_time_played = -1
        self.schedule = []
    
    def calculate_match_distribution(self, verbose=True):
        """
        참가자 수에 따른 최적의 경기 타입 분배 계산
        제약조건:
        1. 모든 남자가 혼복 최소 1회 참여 (여자 2명 이상일 때)
        2. 남녀 슬롯이 가용 범위 내
        3. 참여 횟수 균형 (차이 1회 이내)
        """
        num_males = len(self.male_players)
        num_females = len(self.female_players)
        total = num_males + num_females
        
        # 가용 슬롯
        male_slots_available = num_males * self.time_slots
        female_slots_available = num_females * self.time_slots
        
        # 최소 필요 조건 체크
        if total < 4:
            raise ValueError(f"최소 4명 이상 필요합니다. 현재: {total}명")
        
        if num_males < 4:
            raise ValueError(f"남자 최소 4명 필요합니다. 현재: {num_males}명")
        
        # 여자가 2명 미만이면 모두 남복으로 구성
        if num_females < 2:
            max_matches_per_time = min(self.courts, num_males // 4)
            total_matches = max_matches_per_time * self.time_slots
            
            if verbose:
                print(f"\n📊 경기 분배 계산 결과:")
                print(f"   참가자: 남자 {num_males}명, 여자 {num_females}명 (총 {total}명)")
                print(f"   ⚠️ 여자 2명 미만으로 모두 남복 경기로 구성")
                print(f"   경기 분배: 남복 {total_matches}, 여복 0, 혼복 0 (총 {total_matches}경기)")
                avg_male = (total_matches * 4) / num_males if num_males > 0 else 0
                print(f"   예상 참여: 남자 평균 {avg_male:.1f}회")
            
            return total_matches, 0, 0  # 남복, 여복, 혼복
        
        # 혼복 최소 수: 모든 남자가 1회 이상 참여
        min_mixed = (num_males + 1) // 2
        
        # 혼복 최대 수: 여자 슬롯 제한 고려
        # 혼복 m경기 + 여복 f경기 -> 여자 필요 슬롯 = 2m + 4f
        # 여자가 너무 많이 뛰지 않도록 (평균 4회 이하)
        max_mixed_by_female = (female_slots_available - 8) // 2  # 여복 2경기 가정
        max_mixed_by_female = max(max_mixed_by_female, min_mixed)
        
        # 실제 가능한 경기 수 (15경기 또는 슬롯에 맞게 조정)
        # 타임당 최대 경기 수 = min(3, total // 4)
        max_matches_per_time = min(self.courts, total // 4)
        max_total_matches = max_matches_per_time * self.time_slots
        target_matches = min(15, max_total_matches)
        
        best_config = None
        best_score = float('inf')
        
        # 가능한 모든 조합 탐색
        for mixed in range(min_mixed, min(target_matches + 1, max_mixed_by_female + 1)):
            remaining = target_matches - mixed
            
            for female in range(0, min(remaining + 1, 4)):  # 여복 최대 3경기
                male = remaining - female
                
                if male < 0:
                    continue
                
                # 슬롯 계산
                male_slots = male * 4 + mixed * 2
                female_slots = female * 4 + mixed * 2
                
                # 제약 체크
                if male_slots > male_slots_available:
                    continue
                if female_slots > female_slots_available:
                    continue
                if male > 0 and num_males < 4:
                    continue
                if female > 0 and num_females < 4:
                    continue
                
                # 평균 참여 횟수
                avg_male = male_slots / num_males if num_males > 0 else 0
                avg_female = female_slots / num_females if num_females > 0 else 0
                
                # 점수 계산 (낮을수록 좋음)
                score = 0
                
                # 1. 15경기에 가까울수록 좋음
                score += (15 - (male + female + mixed)) * 100
                
                # 2. 남녀 평균 참여 횟수 균형
                score += abs(avg_male - avg_female) * 50
                
                # 3. 참여 횟수가 적당한 범위 (2~4회)
                if avg_male < 2 or avg_male > 5:
                    score += 30
                if avg_female < 2 or avg_female > 5:
                    score += 30
                
                # 4. 혼복이 너무 많으면 페널티
                if mixed > 10:
                    score += (mixed - 10) * 20
                
                if score < best_score:
                    best_score = score
                    best_config = (male, female, mixed)
        
        if best_config is None:
            # 폴백: 최소 구성
            mixed = min_mixed
            male = max(0, target_matches - mixed - 1)
            female = max(0, target_matches - mixed - male)
            best_config = (male, female, mixed)
            print(f"⚠️ 최적 구성을 찾지 못해 기본값 사용: 남복{male}, 여복{female}, 혼복{mixed}")
        
        male_count, female_count, mixed_count = best_config
        
        # 결과 검증 및 출력
        if verbose:
            male_slots = male_count * 4 + mixed_count * 2
            female_slots = female_count * 4 + mixed_count * 2
            avg_male = male_slots / num_males if num_males > 0 else 0
            avg_female = female_slots / num_females if num_females > 0 else 0
            
            print(f"\n📊 경기 분배 계산 결과:")
            print(f"   참가자: 남자 {num_males}명, 여자 {num_females}명 (총 {total}명)")
            print(f"   경기 분배: 남복 {male_count}, 여복 {female_count}, 혼복 {mixed_count} (총 {male_count+female_count+mixed_count}경기)")
            print(f"   예상 참여: 남자 평균 {avg_male:.1f}회, 여자 평균 {avg_female:.1f}회")
        
        return male_count, female_count, mixed_count
    
    def validate_configuration(self):
        """설정 유효성 검증"""
        num_males = len(self.male_players)
        num_females = len(self.female_players)
        total = num_males + num_females
        
        warnings = []
        errors = []
        
        # 최소 인원 체크
        if total < 8:
            warnings.append(f"참가자가 적어 일부 타임에 빈 코트가 발생할 수 있습니다. (현재 {total}명)")
        
        if num_males < 4:
            errors.append(f"남자가 4명 미만이어서 경기 구성이 불가능합니다. (현재 {num_males}명)")
        
        if num_females < 2:
            warnings.append(f"여자가 2명 미만이어서 모두 남복 경기로 구성됩니다. (현재 {num_females}명)")
        elif num_females < 4:
            warnings.append(f"여자가 4명 미만이어서 여복 경기가 불가능합니다. (현재 {num_females}명)")
        
        # 타임당 코트 사용률
        max_per_time = min(total // 4, self.courts)
        if max_per_time < self.courts:
            warnings.append(f"인원 부족으로 타임당 최대 {max_per_time}코트만 사용 가능합니다.")
        
        # 결과 출력
        if errors:
            print("\n❌ 오류:")
            for e in errors:
                print(f"   - {e}")
            raise ValueError("매칭 불가능한 조건입니다.")
        
        if warnings:
            print("\n⚠️ 경고:")
            for w in warnings:
                print(f"   - {w}")
        
        return True
    
    def get_players_in_time(self, time_slot):
        """해당 타임에 참여 중인 선수 이름 집합"""
        players_in_time = set()
        for match in self.schedule:
            if match.time_slot == time_slot:
                for p in match.get_all_players():
                    players_in_time.add(p.name)
        return players_in_time
    
    def get_available_players(self, time_slot, gender=None, exclude=None):
        """해당 타임에 사용 가능한 선수 목록"""
        players_in_time = self.get_players_in_time(time_slot)
        
        available = []
        source = self.players if gender is None else (self.male_players if gender == 1 else self.female_players)
        
        for p in source:
            if p.name not in players_in_time:
                if exclude is None or p.name not in [e.name for e in exclude]:
                    available.append(p)
        
        return available
    
    def update_player_stats(self, match):
        """경기 후 선수 통계 업데이트"""
        for p in match.get_all_players():
            p.matches_played += 1
            p.last_time_played = match.time_slot
            
            if match.match_type == '혼복':
                p.mixed_matches += 1
            else:
                p.same_doubles += 1
        
        # 팀메이트 기록
        for p in match.team1:
            for other in match.team1:
                if p != other:
                    p.teammates[other.name] += 1
        for p in match.team2:
            for other in match.team2:
                if p != other:
                    p.teammates[other.name] += 1
        
        # 상대 기록
        for p1 in match.team1:
            for p2 in match.team2:
                p1.opponents[p2.name] += 1
                p2.opponents[p1.name] += 1

    def evaluate_match_quality(self, team1, team2, match_type):
        """경기 품질 평가 (낮을수록 좋음)"""
        score = 0
        all_players = team1 + team2
        
        # 1. 실력 밸런스 (팀 평균 차이)
        avg1 = sum(p.skill for p in team1) / 2
        avg2 = sum(p.skill for p in team2) / 2
        skill_diff = abs(avg1 - avg2)
        score += skill_diff * 100
        
        # 2. 상위/하위 선수 실력차 체크
        if match_type in ['남복', '여복']:
            t1_sorted = sorted(team1, key=lambda p: p.skill)
            t2_sorted = sorted(team2, key=lambda p: p.skill)
            top_diff = abs(t1_sorted[0].skill - t2_sorted[0].skill)
            bottom_diff = abs(t1_sorted[1].skill - t2_sorted[1].skill)
            
            if top_diff > 1:
                score += (top_diff - 1) * 200
            if bottom_diff > 1:
                score += (bottom_diff - 1) * 200
            
            # 한 팀이 모두 우위면 패널티
            if (t1_sorted[0].skill < t2_sorted[0].skill and t1_sorted[1].skill < t2_sorted[1].skill):
                score += 300
            if (t2_sorted[0].skill < t1_sorted[0].skill and t2_sorted[1].skill < t1_sorted[1].skill):
                score += 300
                
        elif match_type == '혼복':
            t1_male = [p for p in team1 if p.gender == 1]
            t1_female = [p for p in team1 if p.gender == 2]
            t2_male = [p for p in team2 if p.gender == 1]
            t2_female = [p for p in team2 if p.gender == 2]
            
            if t1_male and t2_male and t1_female and t2_female:
                male_diff = abs(t1_male[0].skill - t2_male[0].skill)
                female_diff = abs(t1_female[0].skill - t2_female[0].skill)
                
                if male_diff > 1:
                    score += (male_diff - 1) * 200
                if female_diff > 1:
                    score += (female_diff - 1) * 200
                
                # 한 팀이 남녀 모두 우위면 패널티
                if (t1_male[0].skill < t2_male[0].skill and t1_female[0].skill < t2_female[0].skill):
                    score += 300
                if (t2_male[0].skill < t1_male[0].skill and t2_female[0].skill < t1_female[0].skill):
                    score += 300
        
        # 3. 같은 팀 반복 페널티 (강화)
        for i, p1 in enumerate(team1):
            for p2 in team1[i+1:]:
                if p1.teammates[p2.name] >= 1:
                    score += p1.teammates[p2.name] * 500  # 1회라도 같은 팀이면 페널티
        for i, p1 in enumerate(team2):
            for p2 in team2[i+1:]:
                if p1.teammates[p2.name] >= 1:
                    score += p1.teammates[p2.name] * 500
        
        # 4. 같은 상대 반복 페널티 (강화)
        for p1 in team1:
            for p2 in team2:
                if p1.opponents[p2.name] >= 1:
                    score += p1.opponents[p2.name] * 200  # 1회라도 만났으면 페널티
        
        return score

    def check_skill_diff_limit(self, team1, team2, match_type):
        """
        상위/하위 선수 간 실력차 제한 체크
        - 남자: 상위/하위 모두 실력차 < 2
        - 여자: 상위/하위 모두 실력차 <= 2
        Returns: True if valid, False if exceeds limit
        """
        if match_type == '남복':
            t1_sorted = sorted(team1, key=lambda p: p.skill)
            t2_sorted = sorted(team2, key=lambda p: p.skill)
            top_diff = abs(t1_sorted[0].skill - t2_sorted[0].skill)
            bottom_diff = abs(t1_sorted[1].skill - t2_sorted[1].skill)
            # 남자: 2 미만
            return top_diff < 2 and bottom_diff < 2
        
        elif match_type == '여복':
            t1_sorted = sorted(team1, key=lambda p: p.skill)
            t2_sorted = sorted(team2, key=lambda p: p.skill)
            top_diff = abs(t1_sorted[0].skill - t2_sorted[0].skill)
            bottom_diff = abs(t1_sorted[1].skill - t2_sorted[1].skill)
            # 여자: 2 이하
            return top_diff <= 2 and bottom_diff <= 2
        
        elif match_type == '혼복':
            t1_male = [p for p in team1 if p.gender == 1]
            t1_female = [p for p in team1 if p.gender == 2]
            t2_male = [p for p in team2 if p.gender == 1]
            t2_female = [p for p in team2 if p.gender == 2]
            
            if t1_male and t2_male and t1_female and t2_female:
                male_diff = abs(t1_male[0].skill - t2_male[0].skill)
                female_diff = abs(t1_female[0].skill - t2_female[0].skill)
                # 남자: 2 미만, 여자: 2 이하
                return male_diff < 2 and female_diff <= 2
        
        return True

    def create_match(self, time_slot, court, match_type, players):
        """최적의 팀 구성으로 매치 생성"""
        if match_type == '혼복':
            males = [p for p in players if p.gender == 1]
            females = [p for p in players if p.gender == 2]
            if len(males) < 2 or len(females) < 2:
                return None
            
            best_match = None
            best_score = float('inf')
            
            for m_idx in range(2):
                for f_idx in range(2):
                    team1 = [males[m_idx], females[f_idx]]
                    team2 = [males[1-m_idx], females[1-f_idx]]
                    
                    # 실력차 제한 체크
                    if not self.check_skill_diff_limit(team1, team2, match_type):
                        continue
                    
                    score = self.evaluate_match_quality(team1, team2, match_type)
                    if score < best_score:
                        best_score = score
                        best_match = Match(time_slot, court, match_type, team1, team2)
            
            # 제한 조건 만족하는 조합이 없으면 최선의 조합 선택
            if best_match is None:
                best_score = float('inf')
                for m_idx in range(2):
                    for f_idx in range(2):
                        team1 = [males[m_idx], females[f_idx]]
                        team2 = [males[1-m_idx], females[1-f_idx]]
                        score = self.evaluate_match_quality(team1, team2, match_type)
                        if score < best_score:
                            best_score = score
                            best_match = Match(time_slot, court, match_type, team1, team2)
            
            return best_match
        else:
            # 남복/여복 - 더 넓은 풀에서 최적의 4명 선택
            if len(players) < 4:
                return None
            
            best_match = None
            best_score = float('inf')
            
            # 6명 이상이면 4명 조합을 탐색, 아니면 전체 사용
            if len(players) > 4:
                player_combos = list(combinations(range(len(players)), 4))
            else:
                player_combos = [tuple(range(len(players)))]
            
            for player_combo in player_combos:
                four_players = [players[i] for i in player_combo]
                
                # 4명 중 팀 나누기
                for team1_combo in combinations(range(4), 2):
                    team2_combo = [i for i in range(4) if i not in team1_combo]
                    team1 = [four_players[i] for i in team1_combo]
                    team2 = [four_players[i] for i in team2_combo]
                    
                    # 실력차 제한 체크
                    if not self.check_skill_diff_limit(team1, team2, match_type):
                        continue
                    
                    score = self.evaluate_match_quality(team1, team2, match_type)
                    if score < best_score:
                        best_score = score
                        best_match = Match(time_slot, court, match_type, team1, team2)
            
            # 제한 조건 만족하는 조합이 없으면 최선의 조합 선택
            if best_match is None:
                best_score = float('inf')
                for player_combo in player_combos:
                    four_players = [players[i] for i in player_combo]
                    for team1_combo in combinations(range(4), 2):
                        team2_combo = [i for i in range(4) if i not in team1_combo]
                        team1 = [four_players[i] for i in team1_combo]
                        team2 = [four_players[i] for i in team2_combo]
                        
                        score = self.evaluate_match_quality(team1, team2, match_type)
                        if score < best_score:
                            best_score = score
                            best_match = Match(time_slot, court, match_type, team1, team2)
            
            return best_match

    def generate_schedule(self, seed=None):
        """스케줄 생성 - 혼복 우선 배치로 모든 남자 참여 보장"""
        if seed is not None:
            random.seed(seed)
        
        self.reset_player_stats()
        
        num_males = len(self.male_players)
        num_females = len(self.female_players)
        
        # 동적으로 경기 분배 계산 (반복 출력 방지)
        male_count, female_count, mixed_count = self.calculate_match_distribution(verbose=False)
        total_matches = male_count + female_count + mixed_count
        
        # 스케줄 그리드
        schedule_grid = [[None for _ in range(self.courts)] for _ in range(self.time_slots)]
        
        # 여자가 2명 미만이면 남복만 배치
        if num_females < 2:
            for time_slot in range(self.time_slots):
                for court in range(self.courts):
                    available = self.get_available_players(time_slot + 1, gender=1)
                    if len(available) >= 4:
                        # 참여 횟수 적은 순 + 랜덤성 추가
                        random.shuffle(available)
                        available.sort(key=lambda p: p.matches_played)
                        
                        # 더 넓은 풀에서 최적의 조합 탐색 (최대 8명)
                        pool_size = min(len(available), 8)
                        match = self.create_match(time_slot + 1, court + 1, '남복', available[:pool_size])
                        if match:
                            schedule_grid[time_slot][court] = match
                            self.schedule.append(match)
                            self.update_player_stats(match)
            
            # 팀 밸런스 조정
            for match in self.schedule:
                match.balance_teams()
            
            return self.schedule
        
        # 실제 사용할 슬롯 수 계산
        all_slots = [(t, c) for t in range(self.time_slots) for c in range(self.courts)]
        random.shuffle(all_slots)
        
        # 슬롯이 부족하면 잘라냄
        all_slots = all_slots[:total_matches]
        
        mixed_slots = all_slots[:mixed_count]
        remaining_slots = all_slots[mixed_count:]
        
        # 1단계: 혼복 배치 - 모든 남자가 참여하도록
        males_shuffled = list(self.male_players)
        random.shuffle(males_shuffled)
        
        # 남자를 2명씩 페어로 나누기
        male_pairs = []
        for i in range(0, len(males_shuffled), 2):
            if i + 1 < len(males_shuffled):
                male_pairs.append((males_shuffled[i], males_shuffled[i+1]))
            else:
                male_pairs.append((males_shuffled[i],))
        
        # 여자도 셔플
        females_shuffled = list(self.female_players)
        random.shuffle(females_shuffled)
        
        mixed_placed = 0
        male_pair_idx = 0
        
        for time_slot, court in mixed_slots:
            if mixed_placed >= mixed_count:
                break
            
            # 이 타임에 이미 배정된 선수 체크
            players_in_time = self.get_players_in_time(time_slot + 1)
            
            # 가용 남자 찾기 (혼복 미참여자 우선)
            no_mixed_males = [p for p in self.male_players 
                           if p.mixed_matches == 0 and p.name not in players_in_time]
            other_males = [p for p in self.male_players 
                         if p.mixed_matches > 0 and p.name not in players_in_time]
            
            # 다양성을 위해 셔플
            random.shuffle(no_mixed_males)
            random.shuffle(other_males)
            
            if len(no_mixed_males) >= 2:
                selected_males = no_mixed_males[:2]
            elif len(no_mixed_males) == 1:
                selected_males = [no_mixed_males[0]] + (other_males[:1] if other_males else [])
            else:
                selected_males = other_males[:2] if len(other_males) >= 2 else []
            
            if len(selected_males) < 2:
                continue
            
            # 가용 여자 찾기
            available_females = [p for p in self.female_players 
                               if p.name not in players_in_time]
            if len(available_females) < 2:
                continue
            
            # 참여 횟수 적은 여자 우선 + 랜덤성 추가
            random.shuffle(available_females)
            available_females.sort(key=lambda p: p.matches_played)
            selected_females = available_females[:2]
            
            # 매치 생성
            match = self.create_match(time_slot + 1, court + 1, '혼복', 
                                     selected_males + selected_females)
            if match:
                schedule_grid[time_slot][court] = match
                self.schedule.append(match)
                self.update_player_stats(match)
                mixed_placed += 1
        
        # 2단계: 여복 배치
        female_placed = 0
        for time_slot, court in remaining_slots:
            if female_placed >= female_count:
                break
            if schedule_grid[time_slot][court] is not None:
                continue
            
            available = self.get_available_players(time_slot + 1, gender=2)
            if len(available) >= 4:
                # 참여 횟수 적은 순 + 랜덤성 추가
                random.shuffle(available)
                available.sort(key=lambda p: p.matches_played)
                
                # 더 넓은 풀에서 최적의 조합 탐색
                pool_size = min(len(available), 6)
                match = self.create_match(time_slot + 1, court + 1, '여복', available[:pool_size])
                if match:
                    schedule_grid[time_slot][court] = match
                    self.schedule.append(match)
                    self.update_player_stats(match)
                    female_placed += 1
        
        # 3단계: 남복 배치
        for time_slot in range(self.time_slots):
            for court in range(self.courts):
                if schedule_grid[time_slot][court] is not None:
                    continue
                
                available = self.get_available_players(time_slot + 1, gender=1)
                if len(available) >= 4:
                    # 참여 횟수 적은 순 + 랜덤성 추가
                    random.shuffle(available)
                    available.sort(key=lambda p: p.matches_played)
                    
                    # 더 넓은 풀에서 최적의 조합 탐색 (최대 8명)
                    pool_size = min(len(available), 8)
                    match = self.create_match(time_slot + 1, court + 1, '남복', available[:pool_size])
                    if match:
                        schedule_grid[time_slot][court] = match
                        self.schedule.append(match)
                        self.update_player_stats(match)
        
        # 4단계: 팀 밸런스 조정 (한 팀이 상위/하위 모두 우위인 경우)
        for match in self.schedule:
            match.balance_teams()
        
        # 5단계: 코트 재배치 (여복→코트3, 남복→코트1,2)
        self.rearrange_courts()
        
        return self.schedule
    
    def rearrange_courts(self):
        """
        코트 재배치: 여복은 코트3, 남복은 코트1,2 우선
        같은 타임 내에서 경기 타입에 따라 코트 번호 재배정
        """
        for time_slot in range(1, self.time_slots + 1):
            time_matches = [m for m in self.schedule if m.time_slot == time_slot]
            
            if len(time_matches) == 0:
                continue
            
            # 경기 타입별 분류
            male_matches = [m for m in time_matches if m.match_type == '남복']
            female_matches = [m for m in time_matches if m.match_type == '여복']
            mixed_matches = [m for m in time_matches if m.match_type == '혼복']
            
            # 코트 재배정: 남복(1,2) → 혼복(중간) → 여복(3)
            court = 1
            
            # 남복 먼저 (코트 1, 2)
            for match in male_matches:
                match.court = court
                court += 1
            
            # 혼복 중간
            for match in mixed_matches:
                match.court = court
                court += 1
            
            # 여복 마지막 (코트 3)
            for match in female_matches:
                match.court = court
                court += 1

    def evaluate_schedule(self, target_matches=15):
        """스케줄 전체 평가"""
        if not self.schedule:
            return float('inf')
        
        score = 0
        
        # 1. 혼복 0회 남자 선수 (최우선) - 여자가 2명 이상일 때만
        if len(self.female_players) >= 2:
            males_no_mixed = [p for p in self.male_players if p.mixed_matches == 0]
            score += len(males_no_mixed) * 10000
        
        # 2. 미참여 선수
        no_participation = [p for p in self.players if p.matches_played == 0]
        score += len(no_participation) * 5000
        
        # 3. 참여 횟수 균형
        participations = [p.matches_played for p in self.players if p.matches_played > 0]
        if participations:
            max_diff = max(participations) - min(participations)
            if max_diff > 1:
                score += (max_diff - 1) * 1000
        
        # 4. 실력 밸런스
        for match in self.schedule:
            score += match.get_skill_diff() * 10
        
        # 5. 상위/하위 실력차 제한 위반 (남자<2, 여자≤2)
        for match in self.schedule:
            if not self.check_skill_diff_limit(match.team1, match.team2, match.match_type):
                score += 500  # 제한 위반 시 큰 페널티
            
            # 기존 상위/하위 실력차 페널티도 유지
            top_diff, bottom_diff = match.get_top_bottom_skill_diff()
            if top_diff > 1:
                score += (top_diff - 1) * 100
            if bottom_diff > 1:
                score += (bottom_diff - 1) * 100
        
        # 6. 목표 경기 수 미달
        if len(self.schedule) < target_matches:
            score += (target_matches - len(self.schedule)) * 2000
        
        # 7. 대진 다양성 (같은 파트너/상대 반복 페널티)
        for p in self.players:
            # 같은 파트너와 2회 이상
            for teammate, count in p.teammates.items():
                if count >= 2:
                    score += (count - 1) * 300
            # 같은 상대와 3회 이상
            for opponent, count in p.opponents.items():
                if count >= 3:
                    score += (count - 2) * 150
        
        return score

    def optimize(self, iterations=1000):
        """최적화"""
        best_schedule = None
        best_score = float('inf')
        best_players_state = None
        
        # 목표 경기 수 사전 계산 (최초 1회만 출력)
        male_count, female_count, mixed_count = self.calculate_match_distribution(verbose=True)
        target_matches = male_count + female_count + mixed_count
        
        print(f"\n최적화 시작 ({iterations} iterations)...")
        
        for i in range(iterations):
            self.generate_schedule(seed=i)
            score = self.evaluate_schedule(target_matches)
            
            if score < best_score:
                best_score = score
                best_schedule = list(self.schedule)
                best_players_state = {}
                for p in self.players:
                    best_players_state[p.name] = {
                        'matches_played': p.matches_played,
                        'mixed_matches': p.mixed_matches,
                        'same_doubles': p.same_doubles,
                        'teammates': dict(p.teammates),
                        'opponents': dict(p.opponents),
                        'last_time_played': p.last_time_played
                    }
                
                males_no_mixed = [p for p in self.male_players if p.mixed_matches == 0]
                if i % 100 == 0:
                    print(f"  Iteration {i}: Score={score:.0f}, 혼복0회남자={len(males_no_mixed)}, 경기수={len(self.schedule)}")
                
                if len(males_no_mixed) == 0 and len(self.schedule) >= target_matches and score < 500:
                    print(f"최적 스케줄 발견! (iteration {i})")
                    break
        
        # 최적 스케줄 복원
        self.schedule = best_schedule
        if best_players_state:
            for p in self.players:
                if p.name in best_players_state:
                    state = best_players_state[p.name]
                    p.matches_played = state['matches_played']
                    p.mixed_matches = state['mixed_matches']
                    p.same_doubles = state['same_doubles']
                    p.teammates = defaultdict(int, state['teammates'])
                    p.opponents = defaultdict(int, state['opponents'])
                    p.last_time_played = state['last_time_played']
        
        print(f"\n최종 스코어: {best_score}")
        return best_schedule

    def print_schedule(self):
        """스케줄 출력"""
        print("\n" + "="*60)
        print("                    매칭 결과")
        print("="*60)
        
        for time_slot in range(1, self.time_slots + 1):
            print(f"\n--- 타임 {time_slot} ---")
            time_matches = [m for m in self.schedule if m.time_slot == time_slot]
            for match in sorted(time_matches, key=lambda m: m.court):
                skill_diff = match.get_skill_diff()
                print(f"  코트{match.court} [{match.match_type}] "
                      f"{match.team1[0].name}&{match.team1[1].name} vs "
                      f"{match.team2[0].name}&{match.team2[1].name} "
                      f"(실력차: {skill_diff:.1f})")

    def print_statistics(self):
        """통계 출력"""
        print("\n" + "="*60)
        print("                    참여 통계")
        print("="*60)
        print(f"{'성명':^8} {'성별':^4} {'실력':^4} {'총참여':^6} {'남/여복':^6} {'혼복':^4}")
        print("-" * 50)
        
        active_players = [p for p in self.players if p.matches_played > 0]
        for p in sorted(active_players, key=lambda x: (-x.matches_played, x.gender, x.skill)):
            gender_str = "남" if p.gender == 1 else "여"
            doubles_str = str(p.same_doubles) if p.same_doubles > 0 else '-'
            mixed_str = str(p.mixed_matches) if p.mixed_matches > 0 else '-'
            print(f"{p.name:^8} {gender_str:^4} {p.skill:^4} {p.matches_played:^6} {doubles_str:^6} {mixed_str:^4}")
        
        print("\n" + "="*60)
        print("                    검증 결과")
        print("="*60)
        
        # 여자가 2명 이상일 때만 혼복 체크
        if len(self.female_players) >= 2:
            males_no_mixed = [p for p in self.male_players if p.matches_played > 0 and p.mixed_matches == 0]
            if males_no_mixed:
                print(f"⚠️  혼복 0회 남자: {len(males_no_mixed)}명 - {[p.name for p in males_no_mixed]}")
            else:
                print("✅ 모든 남자 선수가 혼복에 1회 이상 참여")
        else:
            print("ℹ️  여자 2명 미만으로 남복 전용 경기")
        
        participations = [p.matches_played for p in self.players if p.matches_played > 0]
        if participations:
            diff = max(participations) - min(participations)
            print(f"{'✅' if diff <= 1 else '⚠️ '} 참여 횟수: {min(participations)} ~ {max(participations)} (차이: {diff})")
        
        no_participation = [p for p in self.players if p.matches_played == 0]
        if no_participation:
            print(f"⚠️  미참여자: {len(no_participation)}명 - {[p.name for p in no_participation]}")
        else:
            print("✅ 모든 선수 참여")
        
        print(f"{'✅' if len(self.schedule) >= 15 else '⚠️ '} 총 경기 수: {len(self.schedule)}/15")
        
        skill_diffs = [m.get_skill_diff() for m in self.schedule]
        avg_diff = np.mean(skill_diffs) if skill_diffs else 0
        print(f"{'✅' if avg_diff <= 1.0 else '⚠️ '} 평균 팀간 실력차: {avg_diff:.2f}")
        
        # 상위/하위 실력차 제한 위반 체크
        violations = []
        for match in self.schedule:
            if not self.check_skill_diff_limit(match.team1, match.team2, match.match_type):
                top_diff, bottom_diff = match.get_top_bottom_skill_diff()
                violations.append(f"T{match.time_slot}C{match.court}({match.match_type}): 상위{top_diff:.0f}/하위{bottom_diff:.0f}")
        
        if violations:
            print(f"⚠️  실력차 제한 위반: {len(violations)}건")
            for v in violations[:5]:  # 최대 5개만 표시
                print(f"     {v}")
        else:
            print("✅ 상위/하위 실력차 제한 충족 (남자<2, 여자≤2)")
        
        # 대진 다양성 체크
        repeat_partners = 0
        repeat_opponents = 0
        for p in self.players:
            for count in p.teammates.values():
                if count >= 2:
                    repeat_partners += 1
            for count in p.opponents.values():
                if count >= 3:
                    repeat_opponents += 1
        
        if repeat_partners == 0 and repeat_opponents == 0:
            print("✅ 대진 다양성 양호 (파트너/상대 반복 없음)")
        else:
            if repeat_partners > 0:
                print(f"⚠️  같은 파트너 2회 이상: {repeat_partners // 2}쌍")
            if repeat_opponents > 0:
                print(f"⚠️  같은 상대 3회 이상: {repeat_opponents // 2}쌍")
        
        male_matches = len([m for m in self.schedule if m.match_type == '남복'])
        female_matches = len([m for m in self.schedule if m.match_type == '여복'])
        mixed_matches = len([m for m in self.schedule if m.match_type == '혼복'])
        print(f"\n경기 타입: 남복 {male_matches}, 여복 {female_matches}, 혼복 {mixed_matches}")

    def export_to_excel(self, output_path):
        """엑셀 파일로 출력"""
        match_data = []
        for match in sorted(self.schedule, key=lambda m: (m.time_slot, m.court)):
            top_diff, bottom_diff = match.get_top_bottom_skill_diff()
            match_data.append({
                '코트': match.court,
                '타임': match.time_slot,
                '경기타입': match.match_type,
                '팀1_선수1': match.team1[0].name,
                '팀1_선수2': match.team1[1].name,
                '팀1_평균실력': match.get_team_avg_skill(match.team1),
                '팀2_선수1': match.team2[0].name,
                '팀2_선수2': match.team2[1].name,
                '팀2_평균실력': match.get_team_avg_skill(match.team2),
                '팀평균_실력차': match.get_skill_diff(),
                '상위선수_실력차': top_diff,
                '하위선수_실력차': bottom_diff
            })
        df_matches = pd.DataFrame(match_data)
        
        timetable_data = []
        for time_slot in range(1, self.time_slots + 1):
            row = {'타임': time_slot}
            time_matches = [m for m in self.schedule if m.time_slot == time_slot]
            
            # 해당 타임에 경기하는 선수들 수집
            playing_players = set()
            for match in time_matches:
                playing_players.add(match.team1[0].name)
                playing_players.add(match.team1[1].name)
                playing_players.add(match.team2[0].name)
                playing_players.add(match.team2[1].name)
            
            # 쉬는 선수들 찾기 (참여하는 선수 중 경기하지 않는 선수)
            resting_players = [p.name for p in self.players if p.matches_played > 0 and p.name not in playing_players]
            
            for court in range(1, self.courts + 1):
                court_match = next((m for m in time_matches if m.court == court), None)
                if court_match:
                    t1 = f"{court_match.team1[0].name} & {court_match.team1[1].name}"
                    t2 = f"{court_match.team2[0].name} & {court_match.team2[1].name}"
                    row[f'코트{court}'] = f"[{court_match.match_type}]\n{t1}\nvs\n{t2}"
                else:
                    row[f'코트{court}'] = "-"
            
            # 쉬는 사람들 추가
            row['쉬는 사람'] = ', '.join(resting_players) if resting_players else '-'
            timetable_data.append(row)
        df_timetable = pd.DataFrame(timetable_data)
        
        stats_data = []
        for p in sorted(self.players, key=lambda x: (-x.matches_played, x.gender, x.skill)):
            if p.matches_played > 0:
                gender_str = "남" if p.gender == 1 else "여"
                stats_data.append({
                    '성명': p.name,
                    '성별': gender_str,
                    '실력': p.skill,
                    '참여횟수': p.matches_played,
                    '남복' if p.gender == 1 else '여복': p.same_doubles if p.same_doubles > 0 else '-',
                    '혼복': p.mixed_matches if p.mixed_matches > 0 else '-'
                })
        df_stats = pd.DataFrame(stats_data)
        
        participations = [p.matches_played for p in self.players if p.matches_played > 0]
        skill_diffs = [m.get_skill_diff() for m in self.schedule]
        top_diffs = [m.get_top_bottom_skill_diff()[0] for m in self.schedule]
        bottom_diffs = [m.get_top_bottom_skill_diff()[1] for m in self.schedule]
        
        summary_data = [
            {'항목': '총 경기 수', '값': len(self.schedule)},
            {'항목': '남복 경기 수', '값': len([m for m in self.schedule if m.match_type == '남복'])},
            {'항목': '여복 경기 수', '값': len([m for m in self.schedule if m.match_type == '여복'])},
            {'항목': '혼복 경기 수', '값': len([m for m in self.schedule if m.match_type == '혼복'])},
            {'항목': '총 참가자 수', '값': len([p for p in self.players if p.matches_played > 0])},
            {'항목': '남자 참가자', '값': len([p for p in self.male_players if p.matches_played > 0])},
            {'항목': '여자 참가자', '값': len([p for p in self.female_players if p.matches_played > 0])},
            {'항목': '평균 참여 횟수', '값': round(np.mean(participations), 2) if participations else 0},
            {'항목': '최대 참여 횟수', '값': max(participations) if participations else 0},
            {'항목': '최소 참여 횟수', '값': min(participations) if participations else 0},
            {'항목': '평균 팀간 실력차', '값': round(np.mean(skill_diffs), 2) if skill_diffs else 0},
            {'항목': '평균 상위선수 실력차', '값': round(np.mean(top_diffs), 2) if top_diffs else 0},
            {'항목': '평균 하위선수 실력차', '값': round(np.mean(bottom_diffs), 2) if bottom_diffs else 0},
        ]
        df_summary = pd.DataFrame(summary_data)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_matches.to_excel(writer, sheet_name='매칭결과', index=False)
            df_timetable.to_excel(writer, sheet_name='타임표', index=False)
            df_stats.to_excel(writer, sheet_name='참여통계', index=False)
            df_summary.to_excel(writer, sheet_name='전체요약', index=False)
        
        print(f"\n엑셀 저장: {output_path}")
        return output_path

    def export_to_pdf(self, output_path):
        """PDF 파일로 출력"""
        if not PDF_AVAILABLE:
            print("PDF 출력 불가: reportlab 패키지를 설치하세요 (pip install reportlab)")
            return None
        
        font_registered = False
        # Windows 및 Linux 폰트 경로
        font_paths = [
            # Windows
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/NanumGothic.ttf',
            'C:/Windows/Fonts/gulim.ttc',
            # Linux (Ubuntu/Debian)
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            # 프로젝트 로컬 폰트
            './fonts/NanumGothic.ttf',
            '../fonts/NanumGothic.ttf',
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    font_registered = True
                    print(f"폰트 등록 성공: {font_path}")
                    break
                except Exception as e:
                    print(f"폰트 등록 실패 ({font_path}): {e}")
                    continue
        
        if not font_registered:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        
        korean_font = 'Korean' if font_registered else 'Helvetica'
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(A4),
            rightMargin=1*cm, leftMargin=1*cm,
            topMargin=1*cm, bottomMargin=1*cm
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('TitleKorean', parent=styles['Title'],
                                    fontName=korean_font, fontSize=20, alignment=1)
        normal_style = ParagraphStyle('NormalKorean', parent=styles['Normal'],
                                     fontName=korean_font, fontSize=10)
        
        elements.append(Paragraph("테니스 타임표", title_style))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph(f"생성일: {datetime.now().strftime('%Y년 %m월 %d일')}", normal_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # 벤치 컬럼용 작은 폰트 스타일
        bench_style = ParagraphStyle('BenchKorean', parent=styles['Normal'],
                                     fontName=korean_font, fontSize=7, 
                                     alignment=1, leading=9)
        
        table_data = [['타임', '코트 1', '코트 2', '코트 3', '벤치']]
        for time_slot in range(1, self.time_slots + 1):
            row = [f'{time_slot}']
            time_matches = [m for m in self.schedule if m.time_slot == time_slot]
            
            # 해당 타임에 경기하는 선수들 수집
            playing_players = set()
            for match in time_matches:
                playing_players.add(match.team1[0].name)
                playing_players.add(match.team1[1].name)
                playing_players.add(match.team2[0].name)
                playing_players.add(match.team2[1].name)
            
            # 쉬는 선수들 찾기
            resting_players = [p.name for p in self.players if p.matches_played > 0 and p.name not in playing_players]
            
            for court in range(1, self.courts + 1):
                court_match = next((m for m in time_matches if m.court == court), None)
                if court_match:
                    t1 = f"{court_match.team1[0].name} & {court_match.team1[1].name}"
                    t2 = f"{court_match.team2[0].name} & {court_match.team2[1].name}"
                    row.append(f"[{court_match.match_type}]\n{t1}\nvs\n{t2}")
                else:
                    row.append("-")
            
            # 쉬는 사람들 추가 - Paragraph로 감싸서 자동 줄바꿈
            bench_text = ', '.join(resting_players) if resting_players else '-'
            row.append(Paragraph(bench_text, bench_style))
            table_data.append(row)
        
        table = Table(table_data, colWidths=[1.5*cm, 6*cm, 6*cm, 6*cm, 5*cm])
        
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
            ('BACKGROUND', (4, 1), (4, -1), colors.HexColor('#FFF2CC')),
        ])
        
        for i, time_slot in enumerate(range(1, self.time_slots + 1), start=1):
            time_matches = [m for m in self.schedule if m.time_slot == time_slot]
            for court in range(1, self.courts + 1):
                court_match = next((m for m in time_matches if m.court == court), None)
                if court_match:
                    if court_match.match_type == '남복':
                        bg = colors.HexColor('#DDEBF7')
                    elif court_match.match_type == '여복':
                        bg = colors.HexColor('#FCE4D6')
                    else:
                        bg = colors.HexColor('#E2EFDA')
                    table_style.add('BACKGROUND', (court, i), (court, i), bg)
        
        table.setStyle(table_style)
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
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOX', (1, 0), (1, 0), 1, colors.black),
            ('BOX', (2, 0), (2, 0), 1, colors.black),
            ('BOX', (3, 0), (3, 0), 1, colors.black),
        ]))
        elements.append(legend)
        
        # elements.append(Spacer(1, 0.5*cm))
        # male_m = len([m for m in self.schedule if m.match_type == '남복'])
        # female_m = len([m for m in self.schedule if m.match_type == '여복'])
        # mixed_m = len([m for m in self.schedule if m.match_type == '혼복'])
        # parts = [p.matches_played for p in self.players if p.matches_played > 0]
        # diffs = [m.get_skill_diff() for m in self.schedule]
        
        # summary = f"""
        # 총 경기: {len(self.schedule)}경기 (남복 {male_m}, 여복 {female_m}, 혼복 {mixed_m})<br/>
        # 참가자: 남자 {len([p for p in self.male_players if p.matches_played > 0])}명, 
        # 여자 {len([p for p in self.female_players if p.matches_played > 0])}명<br/>
        # 참여 횟수: 최소 {min(parts) if parts else 0}회 ~ 최대 {max(parts) if parts else 0}회<br/>
        # 평균 팀간 실력차: {np.mean(diffs):.2f}
        # """
        # elements.append(Paragraph(summary, normal_style))
        
        doc.build(elements)
        print(f"PDF 저장: {output_path}")
        return output_path


def main():
    """메인 실행"""
    base_path = r'c:\project\matching'
    roster_path = os.path.join(base_path, 'dataset', 'roster.xlsx')
    participation_path = os.path.join(base_path, 'dataset', 'participation.xlsx')
    
    system = TennisMatchingSystem(roster_path, participation_path)
    
    # 유효성 검증
    try:
        system.validate_configuration()
    except ValueError as e:
        print(f"\n❌ 매칭 실행 불가: {e}")
        return None
    
    system.optimize(iterations=1000)
    
    system.print_schedule()
    system.print_statistics()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    system.export_to_excel(os.path.join(base_path, 'results', f'테니스_매칭결과_{timestamp}.xlsx'))
    system.export_to_pdf(os.path.join(base_path, 'results', f'테니스_타임표_{timestamp}.pdf'))
    system.export_to_pdf(f'./테니스_타임표.pdf')

    return system


if __name__ == '__main__':
    main()
