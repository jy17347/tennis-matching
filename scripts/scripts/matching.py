import pandas as pd
import random
from itertools import combinations
from datetime import datetime
import sys, os

# PDF 라이브러리 (선택적)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ reportlab이 설치되지 않아 PDF 생성 기능을 사용할 수 없습니다.")
    print("   설치: pip install reportlab")


def resource_path(relative_path):
    """ 리소스의 절대 경로를 반환함 (빌드 후 임시 폴더 경로 대응) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 엑셀 파일 불러오기
sabang_file = resource_path('./dataset/roster.xlsx')
chamyeo_file = './participation.xlsx' 

df_sabang = pd.read_excel(sabang_file)
player_info = df_sabang[['성명', '성별', '실력']].copy()

df_chamyeo = pd.read_excel(chamyeo_file)
players = pd.merge(player_info, df_chamyeo, on='성명', how='inner')
players = players[(players['비고 O'] == 'O') | (players['비고 O'] == '1')].copy()

print("=== 참가자 목록 ===")
print(players)
print(f"\n총 참가자 수: {len(players)}명")

men = players[players['성별'] == 1].reset_index(drop=True)
women = players[players['성별'] == 2].reset_index(drop=True)
print(f"남자: {len(men)}명, 여자: {len(women)}명\n")

# 매칭 설정
NUM_COURTS = 3
NUM_TIMES = 5
TOTAL_MATCHES = NUM_COURTS * NUM_TIMES

# 추적 변수
participation_count = {name: 0 for name in players['성명']}
men_honbok_count = {name: 0 for name in men['성명']}
players_in_time_slot = {time: set() for time in range(1, NUM_TIMES + 1)}
matches = []

# 매칭 이력 추적
four_players_history = []  # 같은 4명 조합 방지
team_pair_count = {}  # 같은 팀 조합 횟수 추적 (최대 3번)
three_players_count = {}  # 같은 3명 조합 횟수 추적 (최대 2번)
player_tier_combo_count = {}  # 남복: 각 선수의 티어별 조합 횟수 추적 (다양성)

# 이상적 참여 횟수
total_slots = TOTAL_MATCHES * 4
total_players = len(players)
ideal_participation = total_slots / total_players
min_participation = int(ideal_participation)
max_participation = int(ideal_participation) + 1

print(f"총 선수 슬롯: {total_slots}, 총 선수: {total_players}")
print(f"이상적 참여 횟수: {ideal_participation:.2f}")
print(f"허용 범위: {min_participation}~{max_participation}회\n")

# 타임별 경기 타입 스케줄
time_schedule = {
    1: ['남복', '남복', '여복'],
    2: ['남복', '혼복', '혼복'],
    3: ['남복', '혼복', '혼복'],
    4: ['남복', '남복', '여복'],
    5: ['남복', '남복', '혼복']
}

def get_skill(player_name):
    return players[players['성명'] == player_name]['실력'].values[0]

def get_team_skill_avg(team):
    return sum([get_skill(p) for p in team]) / len(team)

def create_match(team1, team2, match_type, court, time_slot):
    return {
        'court': court,
        'time': time_slot,
        'type': match_type,
        'team1': team1,
        'team2': team2,
        'team1_skill': get_team_skill_avg(team1),
        'team2_skill': get_team_skill_avg(team2),
        'skill_diff': abs(get_team_skill_avg(team1) - get_team_skill_avg(team2))
    }

def find_balanced_teams_simple(four_players):
    """
    제약조건 6 적용: 
    - 상위 선수끼리 실력차 ≤1
    - 하위 선수끼리 실력차 ≤1  
    - 최상위 2명 분리
    
    추가 제약:
    - 같은 4명 조합 방지
    - 같은 팀 조합 최대 3번
    - 같은 3명 조합 최대 2번 (다양성 확보)
    """
    # 같은 4명 조합 체크
    four_players_sorted = tuple(sorted(four_players))
    if four_players_sorted in four_players_history:
        return None
    
    # 같은 3명 조합 체크 (다양성 확보)
    for three_combo in combinations(four_players, 3):
        three_key = tuple(sorted(three_combo))
        if three_players_count.get(three_key, 0) >= 2:
            return None
    
    best_match = None
    best_diff = float('inf')
    
    for team1_combo in combinations(four_players, 2):
        team1 = list(team1_combo)
        team2 = [p for p in four_players if p not in team1]
        
        # 같은 팀 조합 횟수 체크 (최대 3번)
        team1_key = tuple(sorted(team1))
        team2_key = tuple(sorted(team2))
        
        if team_pair_count.get(team1_key, 0) >= 3:
            continue
        if team_pair_count.get(team2_key, 0) >= 3:
            continue
        
        # 각 팀의 실력 정렬
        team1_skills = sorted([(p, get_skill(p)) for p in team1], key=lambda x: x[1])
        team2_skills = sorted([(p, get_skill(p)) for p in team2], key=lambda x: x[1])
        
        # 상위 선수끼리 실력차, 하위 선수끼리 실력차
        top_diff = abs(team1_skills[0][1] - team2_skills[0][1])
        bottom_diff = abs(team1_skills[1][1] - team2_skills[1][1])
        
        # 제약조건 6: 실력차 > 1.0이면 제외
        if top_diff > 1.0 or bottom_diff > 1.0:
            continue
        
        # 최상위 2명 확인
        all_skills = sorted([(p, get_skill(p)) for p in four_players], key=lambda x: x[1])
        best_player = all_skills[0][0]
        second_player = all_skills[1][0]
        
        # 제약조건 6: 최상위 2명이 같은 팀이면 제외
        if (best_player in team1 and second_player in team1) or \
           (best_player in team2 and second_player in team2):
            continue
        
        # 팀간 평균 실력차
        team_diff = abs(get_team_skill_avg(team1) - get_team_skill_avg(team2))
        
        if team_diff < best_diff:
            best_diff = team_diff
            best_match = (team1, team2, team_diff)
    
    return best_match

def find_diverse_nambok_teams(four_players):
    """
    남복 전용: 티어 다양성을 고려한 매칭
    - 같은 티어끼리만 계속 붙지 않도록 다양한 조합 선호
    - 1-2티어, 2-3티어, 1-3티어 등 크로스 조합 우선
    """
    # 같은 4명 조합 체크
    four_players_sorted = tuple(sorted(four_players))
    if four_players_sorted in four_players_history:
        return None
    
    # 같은 3명 조합 체크 (다양성 확보)
    for three_combo in combinations(four_players, 3):
        three_key = tuple(sorted(three_combo))
        if three_players_count.get(three_key, 0) >= 2:
            return None
    
    # 각 선수의 티어 정보
    player_tiers = {p: get_skill(p) for p in four_players}
    
    best_match = None
    best_score = -1
    
    for team1_combo in combinations(four_players, 2):
        team1 = list(team1_combo)
        team2 = [p for p in four_players if p not in team1]
        
        # 같은 팀 조합 횟수 체크 (최대 3번)
        team1_key = tuple(sorted(team1))
        team2_key = tuple(sorted(team2))
        
        if team_pair_count.get(team1_key, 0) >= 3:
            continue
        if team_pair_count.get(team2_key, 0) >= 3:
            continue
        
        # 티어 다양성 점수 계산
        diversity_score = 0
        
        # 팀 내 티어 차이 (크로스 조합 선호)
        team1_tier_diff = abs(player_tiers[team1[0]] - player_tiers[team1[1]])
        team2_tier_diff = abs(player_tiers[team2[0]] - player_tiers[team2[1]])
        
        # 티어 차이가 있으면 점수 증가 (크로스 조합)
        diversity_score += (team1_tier_diff + team2_tier_diff) * 10
        
        # 각 선수가 이 티어 조합을 얼마나 했는지 체크
        for player in four_players:
            partner_tier = player_tiers[team1[1]] if team1[0] == player else (
                player_tiers[team1[0]] if team1[1] == player else (
                    player_tiers[team2[1]] if team2[0] == player else player_tiers[team2[0]]
                )
            )
            combo_key = (player, partner_tier)
            combo_count = player_tier_combo_count.get(combo_key, 0)
            
            # 적게 해본 티어 조합이면 점수 증가
            diversity_score += (3 - combo_count) * 5
        
        # 팀 밸런스 (평균 실력차)
        team_diff = abs(get_team_skill_avg(team1) - get_team_skill_avg(team2))
        
        # 밸런스가 나쁘면 점수 감소
        diversity_score -= team_diff * 20
        
        # 실력차가 너무 크면 제외 (2.0 초과)
        if team_diff > 2.0:
            continue
        
        if diversity_score > best_score:
            best_score = diversity_score
            best_match = (team1, team2, team_diff)
    
    return best_match

def find_balanced_mixed_teams_simple(two_men, two_women):
    """
    혼복 매칭 - 제약조건 6 적용:
    - 남자끼리 실력차 ≤1
    - 여자끼리 실력차 ≤1
    - 최상위 남자 + 최상위 여자 분리
    
    추가 제약:
    - 같은 4명 조합 방지
    - 같은 팀 조합 최대 3번
    - 같은 3명 조합 최대 2번 (다양성 확보)
    """
    man1, man2 = two_men
    woman1, woman2 = two_women
    
    # 같은 4명 조합 체크
    four_players = two_men + two_women
    four_players_sorted = tuple(sorted(four_players))
    if four_players_sorted in four_players_history:
        return None
    
    # 같은 3명 조합 체크 (다양성 확보)
    for three_combo in combinations(four_players, 3):
        three_key = tuple(sorted(three_combo))
        if three_players_count.get(three_key, 0) >= 2:
            return None
    
    man1_skill = get_skill(man1)
    man2_skill = get_skill(man2)
    woman1_skill = get_skill(woman1)
    woman2_skill = get_skill(woman2)
    
    # 남자끼리, 여자끼리 실력차
    men_diff = abs(man1_skill - man2_skill)
    women_diff = abs(woman1_skill - woman2_skill)
    
    # 제약조건 6: 실력차 > 1.0이면 None
    if men_diff > 1.0 or women_diff > 1.0:
        return None
    
    # 최상위 남자, 최상위 여자 찾기
    better_man = man1 if man1_skill < man2_skill else man2
    better_woman = woman1 if woman1_skill < woman2_skill else woman2
    
    # 조합 생성
    combos = [
        ([man1, woman1], [man2, woman2]),
        ([man1, woman2], [man2, woman1])
    ]
    
    best_match = None
    best_diff = float('inf')
    
    for team1, team2 in combos:
        # 같은 팀 조합 횟수 체크 (최대 3번)
        team1_key = tuple(sorted(team1))
        team2_key = tuple(sorted(team2))
        
        if team_pair_count.get(team1_key, 0) >= 3:
            continue
        if team_pair_count.get(team2_key, 0) >= 3:
            continue
        
        # 제약조건 6: 최상위 남자 + 최상위 여자 같은 팀이면 제외
        if (better_man in team1 and better_woman in team1) or \
           (better_man in team2 and better_woman in team2):
            continue
        
        team_diff = abs(get_team_skill_avg(team1) - get_team_skill_avg(team2))
        
        if team_diff < best_diff:
            best_diff = team_diff
            best_match = (team1, team2, team_diff)
    
    return best_match

def get_available_players(gender_df, current_time, match_type=None):
    """현재 타임에 참여하지 않은 선수들을 참여 횟수순으로 정렬"""
    available = []
    
    for name in gender_df['성명']:
        if participation_count[name] < max_participation and \
           name not in players_in_time_slot[current_time]:
            available.append(name)
    
    # 남자 선수이고 혼복이면 혼복 0회 선수 우선
    if match_type == '혼복' and len(available) > 0:
        if available[0] in men_honbok_count:
            available.sort(key=lambda x: (men_honbok_count[x], participation_count[x]))
        else:
            available.sort(key=lambda x: participation_count[x])
    else:
        available.sort(key=lambda x: participation_count[x])
    
    return available

# ============================================================================
# 매칭 생성
# ============================================================================
print("=" * 80)
print("=== 매칭 시작 ===")
print("=" * 80)

# 1단계: 혼복 우선 배정 (역순)
honbok_positions = []
for time in range(1, NUM_TIMES + 1):
    for court in range(1, NUM_COURTS + 1):
        if time_schedule[time][court - 1] == '혼복':
            honbok_positions.append((time, court))

honbok_positions.reverse()
print(f"\n혼복 경기: {len(honbok_positions)}개")

for idx, (time, court) in enumerate(honbok_positions):
    print(f"혼복 {idx+1}/{len(honbok_positions)} (타임{time} 코트{court})...")
    
    available_men = get_available_players(men, time, '혼복')
    available_women = get_available_players(women, time)
    
    # 혼복 0회 선수 우선
    zero_honbok = [m for m in available_men if men_honbok_count[m] == 0]
    
    if len(zero_honbok) >= 2:
        selected_men = zero_honbok[:2]
    elif len(zero_honbok) == 1:
        selected_men = [zero_honbok[0], available_men[1] if available_men[1] != zero_honbok[0] else available_men[2]]
    else:
        selected_men = available_men[:2]
    
    selected_women = available_women[:2]
    
    # 매칭
    result = find_balanced_mixed_teams_simple(selected_men, selected_women)
    
    if result:
        team1, team2, _ = result
        match = create_match(team1, team2, '혼복', court, time)
        matches.append(match)
        
        # 이력 기록
        four_players_sorted = tuple(sorted(team1 + team2))
        four_players_history.append(four_players_sorted)
        
        # 3명 조합 카운트 업데이트
        for three_combo in combinations(team1 + team2, 3):
            three_key = tuple(sorted(three_combo))
            three_players_count[three_key] = three_players_count.get(three_key, 0) + 1
        
        team1_key = tuple(sorted(team1))
        team2_key = tuple(sorted(team2))
        team_pair_count[team1_key] = team_pair_count.get(team1_key, 0) + 1
        team_pair_count[team2_key] = team_pair_count.get(team2_key, 0) + 1
        
        for player in team1 + team2:
            participation_count[player] += 1
            players_in_time_slot[time].add(player)
            if player in men_honbok_count:
                men_honbok_count[player] += 1
        
        print(f"  ✓ 배정: {team1[0]} & {team1[1]} vs {team2[0]} & {team2[1]}")
    else:
        # 제약 완화
        team1 = [selected_men[0], selected_women[0]]
        team2 = [selected_men[1], selected_women[1]]
        match = create_match(team1, team2, '혼복', court, time)
        matches.append(match)
        
        # 이력 기록
        four_players_sorted = tuple(sorted(team1 + team2))
        four_players_history.append(four_players_sorted)
        
        # 3명 조합 카운트 업데이트
        for three_combo in combinations(team1 + team2, 3):
            three_key = tuple(sorted(three_combo))
            three_players_count[three_key] = three_players_count.get(three_key, 0) + 1
        
        team1_key = tuple(sorted(team1))
        team2_key = tuple(sorted(team2))
        team_pair_count[team1_key] = team_pair_count.get(team1_key, 0) + 1
        team_pair_count[team2_key] = team_pair_count.get(team2_key, 0) + 1
        
        for player in team1 + team2:
            participation_count[player] += 1
            players_in_time_slot[time].add(player)
            if player in men_honbok_count:
                men_honbok_count[player] += 1
        
        print(f"  ⚠ 제약 완화 배정: {team1[0]} & {team1[1]} vs {team2[0]} & {team2[1]}")

# 2단계: 남복/여복 배정
print(f"\n남복/여복 배정...")

for time in range(1, NUM_TIMES + 1):
    for court in range(1, NUM_COURTS + 1):
        match_type = time_schedule[time][court - 1]
        
        if match_type == '혼복':
            continue
        
        if match_type in ['남복', '여복']:
            gender_df = men if match_type == '남복' else women
            available = get_available_players(gender_df, time)
            
            if len(available) < 4:
                # 제약 완화: max_participation 무시하고 다시 시도
                available_relaxed = []
                for name in gender_df['성명']:
                    if name not in players_in_time_slot[time]:
                        available_relaxed.append(name)
                available_relaxed.sort(key=lambda x: participation_count[x])
                
                if len(available_relaxed) >= 4:
                    available = available_relaxed
                    print(f"  ⚠ 타임{time} 코트{court} {match_type}: 참여 제한 완화")
                else:
                    print(f"  ⚠ 타임{time} 코트{court} {match_type}: 선수 부족")
                    continue
            
            # 남복: 티어 다양성 고려, 여복: 밸런스만 고려
            result = None
            if match_type == '남복':
                # 남복: 다양한 티어 조합 시도
                for combo in combinations(available, 4):
                    four_players = list(combo)
                    four_players_sorted = tuple(sorted(four_players))
                    
                    # 이미 사용된 조합이면 스킵
                    if four_players_sorted in four_players_history:
                        continue
                    
                    result = find_diverse_nambok_teams(four_players)
                    if result:
                        break
            else:
                # 여복: 밸런스만 고려 (중복 조합 허용)
                selected = available[:4]
                result = find_balanced_teams_simple(selected)
            
            if result:
                team1, team2, _ = result
            else:
                # 제약 완화: 첫 4명으로 그냥 2:2로 나눔
                selected = available[:4]
                team1 = selected[:2]
                team2 = selected[2:4]
            
            match = create_match(team1, team2, match_type, court, time)
            matches.append(match)
            
            # 4명 조합 히스토리에 추가 (남복만)
            if match_type == '남복':
                four_players_sorted = tuple(sorted(team1 + team2))
                four_players_history.append(four_players_sorted)
                
                # 3명 조합 카운트 업데이트 (남복만)
                for three_combo in combinations(team1 + team2, 3):
                    three_key = tuple(sorted(three_combo))
                    three_players_count[three_key] = three_players_count.get(three_key, 0) + 1
                
                # 남복: 티어 조합 카운트 업데이트
                for player in team1:
                    partner = team1[1] if team1[0] == player else team1[0]
                    partner_tier = get_skill(partner)
                    combo_key = (player, partner_tier)
                    player_tier_combo_count[combo_key] = player_tier_combo_count.get(combo_key, 0) + 1
                
                for player in team2:
                    partner = team2[1] if team2[0] == player else team2[0]
                    partner_tier = get_skill(partner)
                    combo_key = (player, partner_tier)
                    player_tier_combo_count[combo_key] = player_tier_combo_count.get(combo_key, 0) + 1
            
            # 같은 팀 조합 카운트 증가
            team1_sorted = tuple(sorted(team1))
            team2_sorted = tuple(sorted(team2))
            team_pair_count[team1_sorted] = team_pair_count.get(team1_sorted, 0) + 1
            team_pair_count[team2_sorted] = team_pair_count.get(team2_sorted, 0) + 1
            
            for player in team1 + team2:
                participation_count[player] += 1
                players_in_time_slot[time].add(player)
            
            print(f"  타임{time} 코트{court} {match_type}: {team1[0]}&{team1[1]} vs {team2[0]}&{team2[1]}")

# ============================================================================
# 결과 출력
# ============================================================================
print("\n" + "=" * 80)
print("=== 매칭 결과 ===")
print("=" * 80)

for match in matches:
    print(f"\n코트 {match['court']} | 타임 {match['time']} | {match['type']}")
    print(f"  팀1: {' & '.join(match['team1'])} (평균 실력: {match['team1_skill']:.1f})")
    print(f"  팀2: {' & '.join(match['team2'])} (평균 실력: {match['team2_skill']:.1f})")
    print(f"  실력차: {match['skill_diff']:.2f}")

print("\n" + "=" * 80)
print("=== 참여 통계 ===")
print("=" * 80)

stats = []
for name in players['성명']:
    stats.append({
        '성명': name,
        '참여횟수': participation_count[name],
        '성별': players[players['성명'] == name]['성별'].values[0],
        '실력': players[players['성명'] == name]['실력'].values[0]
    })

stats_df = pd.DataFrame(stats).sort_values('참여횟수', ascending=False)
print(stats_df.to_string(index=False))

print(f"\n평균 참여 횟수: {sum(participation_count.values()) / len(participation_count):.2f}")
print(f"최대 참여 횟수: {max(participation_count.values())}")
print(f"최소 참여 횟수: {min(participation_count.values())}")
print(f"참여 횟수 차이: {max(participation_count.values()) - min(participation_count.values())}")

# 혼복 통계
print("\n" + "=" * 80)
print("=== 혼복 통계 (남자 선수) ===")
print("=" * 80)

honbok_stats = []
for name in men['성명']:
    honbok_stats.append({
        '성명': name,
        '혼복': men_honbok_count[name],
        '총참여': participation_count[name]
    })

honbok_df = pd.DataFrame(honbok_stats).sort_values('혼복', ascending=False)
print(honbok_df.to_string(index=False))

zero_honbok = [name for name in men['성명'] if men_honbok_count[name] == 0]
if len(zero_honbok) > 0:
    print(f"\n⚠ 혼복 0회 선수: {', '.join(zero_honbok)}")
else:
    print(f"\n✅ 모든 남자 선수가 혼복 최소 1회 참여")

# 엑셀 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'./results/테니스_매칭결과_{timestamp}.xlsx'

# 1. 매칭 결과 시트
match_records = []
for match in matches:
    # 각 선수의 실력
    team1_player1_skill = get_skill(match['team1'][0])
    team1_player2_skill = get_skill(match['team1'][1])
    team2_player1_skill = get_skill(match['team2'][0])
    team2_player2_skill = get_skill(match['team2'][1])
    
    # 각 팀의 상위/하위 선수 구분
    team1_skills_sorted = sorted([team1_player1_skill, team1_player2_skill])
    team2_skills_sorted = sorted([team2_player1_skill, team2_player2_skill])
    
    # 상위 선수 간 차이, 하위 선수 간 차이
    top_player_diff = abs(team1_skills_sorted[0] - team2_skills_sorted[0])
    bottom_player_diff = abs(team1_skills_sorted[1] - team2_skills_sorted[1])
    
    match_records.append({
        '코트': match['court'],
        '타임': match['time'],
        '경기타입': match['type'],
        '팀1_선수1': match['team1'][0],
        '팀1_선수2': match['team1'][1],
        '팀1_평균실력': round(match['team1_skill'], 1),
        '팀2_선수1': match['team2'][0],
        '팀2_선수2': match['team2'][1],
        '팀2_평균실력': round(match['team2_skill'], 1),
        '팀평균_실력차': round(match['skill_diff'], 2),
        '상위선수_실력차': round(top_player_diff, 2),
        '하위선수_실력차': round(bottom_player_diff, 2)
    })

matches_df = pd.DataFrame(match_records)

# 2. 참여 통계 시트
participation_records = []
for name, count in participation_count.items():
    player_row = players[players['성명'] == name].iloc[0]
    record = {
        '성명': name,
        '성별': '남' if player_row['성별'] == 1 else '여',
        '실력': player_row['실력'],
        '참여횟수': count
    }
    
    # 남자 선수의 경우 남복/혼복 추가
    if player_row['성별'] == 1:
        # 남복 횟수 계산
        nambok = sum(1 for m in matches if m['type'] == '남복' and name in m['team1'] + m['team2'])
        record['남복'] = nambok
        record['혼복'] = men_honbok_count[name]
    else:
        record['남복'] = '-'
        record['혼복'] = '-'
    
    participation_records.append(record)

participation_df = pd.DataFrame(participation_records).sort_values('참여횟수', ascending=False)

# 3. 타임표 시트
court_schedule = []
for time in range(1, NUM_TIMES + 1):
    row = {'타임': time}
    for court in range(1, NUM_COURTS + 1):
        match = next((m for m in matches if m['court'] == court and m['time'] == time), None)
        if match:
            match_info = f"[{match['type']}]\n{match['team1'][0]} & {match['team1'][1]}\nvs\n{match['team2'][0]} & {match['team2'][1]}"
            row[f'코트{court}'] = match_info
        else:
            row[f'코트{court}'] = '-'
    court_schedule.append(row)

schedule_df = pd.DataFrame(court_schedule)

# 4. 전체요약 시트
# 개인 간 실력차 계산
if len(matches) > 0:
    total_top_diff = 0
    total_bottom_diff = 0
    for match in matches:
        team1_skills = [get_skill(p) for p in match['team1']]
        team2_skills = [get_skill(p) for p in match['team2']]
        team1_sorted = sorted(team1_skills)
        team2_sorted = sorted(team2_skills)
        total_top_diff += abs(team1_sorted[0] - team2_sorted[0])
        total_bottom_diff += abs(team1_sorted[1] - team2_sorted[1])
    
    avg_top_diff = round(total_top_diff / len(matches), 2)
    avg_bottom_diff = round(total_bottom_diff / len(matches), 2)
    avg_skill_diff = round(sum(m['skill_diff'] for m in matches) / len(matches), 2)
else:
    avg_top_diff = 0
    avg_bottom_diff = 0
    avg_skill_diff = 0

summary_data = {
    '항목': ['총 경기 수', '남복 경기 수', '여복 경기 수', '혼복 경기 수', 
            '총 참가자 수', '남자 참가자', '여자 참가자',
            '평균 참여 횟수', '최대 참여 횟수', '최소 참여 횟수', 
            '평균 팀간 실력차', '평균 상위선수 실력차', '평균 하위선수 실력차'],
    '값': [
        len(matches),
        sum(1 for m in matches if m['type'] == '남복'),
        sum(1 for m in matches if m['type'] == '여복'),
        sum(1 for m in matches if m['type'] == '혼복'),
        len(players),
        len(men),
        len(women),
        round(participation_df['참여횟수'].mean(), 2),
        participation_df['참여횟수'].max(),
        participation_df['참여횟수'].min(),
        avg_skill_diff,
        avg_top_diff,
        avg_bottom_diff
    ]
}
summary_df = pd.DataFrame(summary_data)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    matches_df.to_excel(writer, sheet_name='매칭결과', index=False)
    schedule_df.to_excel(writer, sheet_name='타임표', index=False)
    participation_df.to_excel(writer, sheet_name='참여통계', index=False)
    summary_df.to_excel(writer, sheet_name='전체요약', index=False)
    
    # 열 너비 자동 조정
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

print(f"\n✅ 매칭 결과가 '{output_file}' 파일로 저장되었습니다!")
print(f"   - 시트1: 매칭결과 (전체 경기 목록)")
print(f"   - 시트2: 타임표 (코트별 일정)")
print(f"   - 시트3: 참여통계 (선수별 참여 횟수)")
print(f"   - 시트4: 전체요약 (통계 정보)")

# ============================================================================
# PDF 타임표 생성
# ============================================================================
def create_timetable_pdf(matches, output_path='테니스_타임표.pdf'):
    """
    타임표를 PDF로 생성
    """
    try:
        # 한글 폰트 설정 시도 (Windows 기본 폰트)
        try:
            pdfmetrics.registerFont(TTFont('Malgun', 'malgun.ttf'))
            korean_font = 'Malgun'
        except:
            try:
                pdfmetrics.registerFont(TTFont('Gulim', 'gulim.ttf'))
                korean_font = 'Gulim'
            except:
                korean_font = 'Helvetica'  # fallback
        
        # PDF 생성
        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(A4),
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1.5*cm,
            bottomMargin=1*cm
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=korean_font,
            fontSize=20,
            alignment=1,  # 중앙 정렬
            spaceAfter=20
        )
        
        # 제목 추가
        title = Paragraph("테니스 매칭 타임표", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.5*cm))
        
        # 타임별로 정리
        time_courts = {}
        for match in matches:
            time = match['time']
            court = match['court']
            if time not in time_courts:
                time_courts[time] = {}
            time_courts[time][court] = match
        
        # 테이블 데이터 생성
        data = [['타임', '코트 1', '코트 2', '코트 3']]
        
        for time in sorted(time_courts.keys()):
            row = [f'타임 {time}']
            for court in [1, 2, 3]:
                if court in time_courts[time]:
                    match = time_courts[time][court]
                    cell_text = f"[{match['type']}]\n"
                    cell_text += f"{match['team1'][0]} & {match['team1'][1]}\n"
                    cell_text += "vs\n"
                    cell_text += f"{match['team2'][0]} & {match['team2'][1]}"
                else:
                    cell_text = "-"
                row.append(cell_text)
            data.append(row)
        
        # 테이블 생성
        table = Table(data, colWidths=[3*cm, 6*cm, 6*cm, 6*cm])
        
        # 테이블 스타일
        table.setStyle(TableStyle([
            # 헤더
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), korean_font),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # 타임 열
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#D9E1F2')),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (0, -1), korean_font),
            ('FONTSIZE', (0, 1), (0, -1), 11),
            
            # 본문
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (1, 1), (-1, -1), korean_font),
            ('FONTSIZE', (1, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        
        # PDF 생성
        doc.build(elements)
        
        return True
    except Exception as e:
        print(f"\n⚠ PDF 생성 중 오류 발생: {e}")
        return False

# PDF 타임표 생성
pdf_path = './테니스_타임표.pdf'
if create_timetable_pdf(matches, pdf_path):
    print(f"\n✅ 타임표 PDF가 '{pdf_path}' 파일로 저장되었습니다!")
else:
    print(f"\n⚠ PDF 생성에 실패했습니다. Excel 파일을 확인해주세요.")
