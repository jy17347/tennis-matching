import pandas as pd
import random
from itertools import combinations
from datetime import datetime
import sys, os


def resource_path(relative_path):
    """ 리소스의 절대 경로를 반환함 (빌드 후 임시 폴더 경로 대응) """
    try:
        # PyInstaller에 의해 임시폴더가 생성되면 _MEIPASS 변수가 생김
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 엑셀 파일 불러오기

sabang_file = resource_path('./dataset/roster.xlsx')
chamyeo_file = './participation.xlsx' 

# roster.xlsx에서 성명, 성별, 실력 정보 가져오기
df_sabang = pd.read_excel(sabang_file)
player_info = df_sabang[['성명', '성별', '실력']].copy()

# participation.xlsx에서 참여 여부 정보 가져오기
df_chamyeo = pd.read_excel(chamyeo_file)

# 두 데이터프레임을 성명 기준으로 병합
players = pd.merge(player_info, df_chamyeo, on='성명', how='inner')

# 비고가 "O" 또는 "1"인 사람만 필터링
players = players[(players['비고 O'] == 'O') | (players['비고 O'] == '1')].copy()

# 데이터 출력
print("=== 참가자 목록 ===")
print(players)
print(f"\n총 참가자 수: {len(players)}명")

# 성별로 분류 (1=남자, 2=여자)
men = players[players['성별'] == 1].reset_index(drop=True)
women = players[players['성별'] == 2].reset_index(drop=True)

print(f"남자: {len(men)}명, 여자: {len(women)}명\n")

# 매칭 설정
NUM_COURTS = 3  # 코트 수
NUM_TIMES = 5   # 타임 수
TOTAL_MATCHES = NUM_COURTS * NUM_TIMES  # 총 15경기

# 각 선수의 참여 횟수 추적
participation_count = {name: 0 for name in players['성명']}

# 남자 선수들의 경기 타입별 참여 횟수 추적 (혼복 보장용)
men_match_type_count = {name: {'남복': 0, '혼복': 0} for name in men['성명']}

# 매치 결과 저장
matches = []

def get_team_skill_avg(team, players_df):
    """팀의 평균 실력 계산"""
    skills = []
    for player_name in team:
        skill = players_df[players_df['성명'] == player_name]['실력'].values[0]
        skills.append(skill)
    return sum(skills) / len(skills)

def create_match(p1, p2, p3, p4, match_type, court, time_slot):
    """매치 생성"""
    team1 = [p1, p2]
    team2 = [p3, p4]
    
    team1_skill = get_team_skill_avg(team1, players)
    team2_skill = get_team_skill_avg(team2, players)
    
    return {
        'court': court,
        'time': time_slot,
        'type': match_type,
        'team1': team1,
        'team2': team2,
        'team1_skill': team1_skill,
        'team2_skill': team2_skill,
        'skill_diff': abs(team1_skill - team2_skill)
    }

def improve_team_balance(team1, team2, players_df):
    """한쪽 팀이 상위/하위 모두 우월하면 하위 선수를 교환하여 밸런스 개선"""
    # 각 팀의 선수 실력 가져오기
    team1_skills = [(p, players_df[players_df['성명'] == p]['실력'].values[0]) for p in team1]
    team2_skills = [(p, players_df[players_df['성명'] == p]['실력'].values[0]) for p in team2]
    
    # 실력순 정렬 (작은 값 = 상위, 큰 값 = 하위)
    team1_sorted = sorted(team1_skills, key=lambda x: x[1])
    team2_sorted = sorted(team2_skills, key=lambda x: x[1])
    
    # 상위 선수: team1_sorted[0], team2_sorted[0]
    # 하위 선수: team1_sorted[1], team2_sorted[1]
    
    team1_top_skill = team1_sorted[0][1]
    team1_bottom_skill = team1_sorted[1][1]
    team2_top_skill = team2_sorted[0][1]
    team2_bottom_skill = team2_sorted[1][1]
    
    # 팀1이 상위/하위 모두 우월한 경우 (실력 수치가 낮을수록 잘함)
    if team1_top_skill < team2_top_skill and team1_bottom_skill < team2_bottom_skill:
        # 하위 선수 교환
        new_team1 = [team1_sorted[0][0], team2_sorted[1][0]]
        new_team2 = [team2_sorted[0][0], team1_sorted[1][0]]
        return new_team1, new_team2, True
    
    # 팀2가 상위/하위 모두 우월한 경우
    elif team2_top_skill < team1_top_skill and team2_bottom_skill < team1_bottom_skill:
        # 하위 선수 교환
        new_team1 = [team2_sorted[0][0], team1_sorted[1][0]]
        new_team2 = [team1_sorted[0][0], team2_sorted[1][0]]
        return new_team1, new_team2, True
    
    # 교환 불필요
    return team1, team2, False

def improve_mixed_team_balance(team1, team2, players_df):
    """혼복에서 남자끼리, 여자끼리 비교하여 밸런스 개선
    team1 = [남자1, 여자1], team2 = [남자2, 여자2]
    """
    # 각 선수의 실력 가져오기
    man1 = team1[0]
    woman1 = team1[1]
    man2 = team2[0]
    woman2 = team2[1]
    
    man1_skill = players_df[players_df['성명'] == man1]['실력'].values[0]
    woman1_skill = players_df[players_df['성명'] == woman1]['실력'].values[0]
    man2_skill = players_df[players_df['성명'] == man2]['실력'].values[0]
    woman2_skill = players_df[players_df['성명'] == woman2]['실력'].values[0]
    
    improved = False
    
    # 남자끼리 비교: team1의 남자가 더 잘하면 (작은 값)
    if man1_skill < man2_skill:
        men_diff = man2_skill - man1_skill
    else:
        men_diff = man1_skill - man2_skill
    
    # 여자끼리 비교
    if woman1_skill < woman2_skill:
        women_diff = woman2_skill - woman1_skill
    else:
        women_diff = woman1_skill - woman2_skill
    
    # 남자와 여자의 우월한 팀이 같은 경우 (team1 또는 team2가 남녀 모두 우월)
    if man1_skill < man2_skill and woman1_skill < woman2_skill:
        # team1이 남녀 모두 우월 → 남자끼리 또는 여자끼리 교환
        # 실력차가 더 큰 쪽을 교환
        if men_diff > women_diff:
            # 남자 교환
            new_team1 = [man2, woman1]
            new_team2 = [man1, woman2]
            improved = True
        else:
            # 여자 교환
            new_team1 = [man1, woman2]
            new_team2 = [man2, woman1]
            improved = True
    elif man2_skill < man1_skill and woman2_skill < woman1_skill:
        # team2가 남녀 모두 우월 → 남자끼리 또는 여자끼리 교환
        # 실력차가 더 큰 쪽을 교환
        if men_diff > women_diff:
            # 남자 교환
            new_team1 = [man2, woman1]
            new_team2 = [man1, woman2]
            improved = True
        else:
            # 여자 교환
            new_team1 = [man1, woman2]
            new_team2 = [man2, woman1]
            improved = True
    else:
        # 교환 불필요 (남녀가 서로 다른 팀이 우월)
        new_team1 = team1
        new_team2 = team2
    
    if improved:
        return new_team1, new_team2, True
    return team1, team2, False

def get_least_played_players(gender_df, n, participation_count):
    """참여 횟수가 적은 선수들 선택"""
    names = gender_df['성명'].tolist()
    names_sorted = sorted(names, key=lambda x: participation_count[x])
    return names_sorted[:n]

def find_balanced_teams(available_players, players_df, participation_count, max_skill_diff=0.5):
    """실력차가 최소화되고 참여 횟수가 균등한 팀 조합 찾기 (4명 -> 2vs2)"""
    from itertools import combinations
    
    if len(available_players) < 4:
        return None
    
    # 좋은 조합들을 여러 개 수집
    good_combos = []
    all_combos = []  # 모든 조합 저장 (팀 이력 체크만 통과한 조합)
    best_score = float('inf')
    
    # 4명 중 2명씩 조합 생성
    for team1_combo in combinations(available_players, 2):
        team2_combo = tuple(p for p in available_players if p not in team1_combo)
        
        # 팀 이력 체크: 같은 팀을 2번 이상 했는지 확인
        if has_teamed_before(team1_combo[0], team1_combo[1], max_team_count=2):
            continue
        if has_teamed_before(team2_combo[0], team2_combo[1], max_team_count=2):
            continue
        
        # 적 이력 체크: 적으로 3번 이상 만났는지 확인
        opponent_count_exceeded = False
        for p1 in team1_combo:
            for p2 in team2_combo:
                if has_faced_before(p1, p2, max_opponent_count=3):
                    opponent_count_exceeded = True
                    break
            if opponent_count_exceeded:
                break
        if opponent_count_exceeded:
            continue
        
        # 같이 경기한 횟수 체크: 4명 중 어떤 두 선수든 3번 이하
        together_count_exceeded = False
        all_players = list(team1_combo) + list(team2_combo)
        for i, p1 in enumerate(all_players):
            for p2 in all_players[i+1:]:
                if has_played_together_before(p1, p2, max_together_count=3):
                    together_count_exceeded = True
                    break
            if together_count_exceeded:
                break
        if together_count_exceeded:
            continue
        
        # 팀 밸런스 개선: 한쪽 팀이 상위/하위 모두 우월하면 하위 선수 교환
        improved_team1, improved_team2, was_improved = improve_team_balance(list(team1_combo), list(team2_combo), players_df)
        
        # 교환 후 팀 이력 다시 체크
        if was_improved:
            if has_teamed_before(improved_team1[0], improved_team1[1], max_team_count=2):
                continue
            if has_teamed_before(improved_team2[0], improved_team2[1], max_team_count=2):
                continue
            
            # 교환 후 적 이력 다시 체크
            opponent_count_exceeded = False
            for p1 in improved_team1:
                for p2 in improved_team2:
                    if has_faced_before(p1, p2, max_opponent_count=3):
                        opponent_count_exceeded = True
                        break
                if opponent_count_exceeded:
                    break
            if opponent_count_exceeded:
                continue
            
            # 교환 후 같이 경기한 횟수 다시 체크
            together_count_exceeded = False
            all_players = improved_team1 + improved_team2
            for i, p1 in enumerate(all_players):
                for p2 in all_players[i+1:]:
                    if has_played_together_before(p1, p2, max_together_count=3):
                        together_count_exceeded = True
                        break
                if together_count_exceeded:
                    break
            if together_count_exceeded:
                continue
                for p2 in improved_team2:
                    if has_faced_before(p1, p2, max_opponent_count=2):
                        opponent_count_exceeded = True
                        break
                if opponent_count_exceeded:
                    break
            if opponent_count_exceeded:
                continue
        
        # 각 선수의 실력 가져오기 (개선된 팀 기준)
        team1_skills = [players_df[players_df['성명'] == p]['실력'].values[0] for p in improved_team1]
        team2_skills = [players_df[players_df['성명'] == p]['실력'].values[0] for p in improved_team2]
        
        # 팀 평균 실력
        team1_skill = sum(team1_skills) / 2
        team2_skill = sum(team2_skills) / 2
        skill_diff = abs(team1_skill - team2_skill)
        
        # 개인 간 밸런스: 각 팀의 상위/하위 선수 간 차이도 계산
        team1_skills_sorted = sorted(team1_skills)
        team2_skills_sorted = sorted(team2_skills)
        
        # 상위 선수 간 차이, 하위 선수 간 차이
        top_diff = abs(team1_skills_sorted[0] - team2_skills_sorted[0])
        bottom_diff = abs(team1_skills_sorted[1] - team2_skills_sorted[1])
        
        # ★ 새로운 제약조건 6: 상위끼리 실력차 ≤1, 하위끼리 실력차 ≤1
        if top_diff > 1.0 or bottom_diff > 1.0:
            continue
        
        # ★ 새로운 제약조건 6: 우위 선수들이 같은 팀 금지
        # 4명의 실력을 정렬: [best, second, third, worst]
        all_skills_with_names = [(p, players_df[players_df['성명'] == p]['실력'].values[0]) 
                                  for p in improved_team1 + improved_team2]
        all_skills_sorted = sorted(all_skills_with_names, key=lambda x: x[1])
        best_player = all_skills_sorted[0][0]
        second_player = all_skills_sorted[1][0]
        
        # 상위 2명이 같은 팀이면 제외
        if (best_player in improved_team1 and second_player in improved_team1) or \
           (best_player in improved_team2 and second_player in improved_team2):
            continue
        
        individual_diff = top_diff + bottom_diff
        
        # 참여 횟수 편차 계산
        participation_counts = [participation_count[p] for p in available_players]
        max_part = max(participation_counts)
        min_part = min(participation_counts)
        participation_variance = max_part - min_part
        
        # 점수: 팀평균 실력차 + 개인별 실력차 + 참여 횟수 편차
        score = skill_diff + individual_diff * 0.5 + participation_variance * 0.3
        
        # 모든 조합 저장 (참여 횟수 보장을 위해) - 개선된 팀 기준
        all_combos.append((improved_team1, improved_team2, skill_diff, score))
        
        # 실력차 제약을 만족하는 좋은 조합들만 따로 저장 (개선된 팀 기준)
        if skill_diff <= max_skill_diff and individual_diff <= max_skill_diff:
            # 최적 점수 갱신
            if score < best_score:
                best_score = score
                good_combos = [(improved_team1, improved_team2, skill_diff, score)]
            # 최적 점수에서 15% 이내면 좋은 조합으로 추가
            elif score <= best_score * 1.15:
                good_combos.append((improved_team1, improved_team2, skill_diff, score))
    
    # 좋은 조합이 있으면 그 중에서 랜덤 선택
    if good_combos:
        selected = random.choice(good_combos)
        return (selected[0], selected[1], selected[2])
    
    # 좋은 조합이 없으면 모든 조합 중 최선의 조합 선택 (참여 횟수 보장을 위해)
    if all_combos:
        all_combos.sort(key=lambda x: x[3])  # 점수로 정렬
        selected = all_combos[0]
        return (selected[0], selected[1], selected[2])
    
    return None

def find_balanced_mixed_teams(available_men, available_women, players_df, participation_count, max_skill_diff=0.5):
    """혼복에서 실력차가 최소화되고 참여 횟수가 균등한 팀 조합 찾기 (남2, 여2 -> 남1여1 vs 남1여1)"""
    from itertools import combinations
    
    if len(available_men) < 2 or len(available_women) < 2:
        return None
    
    # 좋은 조합들을 여러 개 수집
    good_combos = []
    all_combos = []  # 모든 조합 저장 (팀 이력 체크만 통과한 조합)
    best_score = float('inf')
    
    # 남자 2명 중 1명씩 분배, 여자 2명 중 1명씩 분배
    for men_team1 in combinations(available_men, 1):
        men_team2 = [m for m in available_men if m not in men_team1]
        
        for women_team1 in combinations(available_women, 1):
            women_team2 = [w for w in available_women if w not in women_team1]
            
            team1 = list(men_team1) + list(women_team1)
            team2 = men_team2 + women_team2
            
            # 팀 이력 체크: 같은 팀을 2번 이상 했는지 확인
            if has_teamed_before(team1[0], team1[1], max_team_count=2):
                continue
            if has_teamed_before(team2[0], team2[1], max_team_count=2):
                continue
            
            # 적 이력 체크: 적으로 3번 이상 만났는지 확인
            opponent_count_exceeded = False
            for p1 in team1:
                for p2 in team2:
                    if has_faced_before(p1, p2, max_opponent_count=3):
                        opponent_count_exceeded = True
                        break
                if opponent_count_exceeded:
                    break
            if opponent_count_exceeded:
                continue
            
            # 같이 경기한 횟수 체크: 4명 중 어떤 두 선수든 3번 이하
            together_count_exceeded = False
            all_players = team1 + team2
            for i, p1 in enumerate(all_players):
                for p2 in all_players[i+1:]:
                    if has_played_together_before(p1, p2, max_together_count=3):
                        together_count_exceeded = True
                        break
                if together_count_exceeded:
                    break
            if together_count_exceeded:
                continue
            
            # 혼복 팀 밸런스 개선: 남자끼리, 여자끼리 비교하여 교환
            improved_team1, improved_team2, was_improved = improve_mixed_team_balance(team1, team2, players_df)
            
            # 교환 후 팀 이력 다시 체크
            if was_improved:
                if has_teamed_before(improved_team1[0], improved_team1[1], max_team_count=2):
                    continue
                if has_teamed_before(improved_team2[0], improved_team2[1], max_team_count=2):
                    continue
                
                # 교환 후 적 이력 다시 체크
                opponent_count_exceeded = False
                for p1 in improved_team1:
                    for p2 in improved_team2:
                        if has_faced_before(p1, p2, max_opponent_count=3):
                            opponent_count_exceeded = True
                            break
                    if opponent_count_exceeded:
                        break
                if opponent_count_exceeded:
                    continue
                
                # 교환 후 같이 경기한 횟수 다시 체크
                together_count_exceeded = False
                all_players = improved_team1 + improved_team2
                for i, p1 in enumerate(all_players):
                    for p2 in all_players[i+1:]:
                        if has_played_together_before(p1, p2, max_together_count=3):
                            together_count_exceeded = True
                            break
                    if together_count_exceeded:
                        break
                if together_count_exceeded:
                    continue
            
            # 혼복은 남자끼리, 여자끼리 실력 비교 (개선된 팀 기준)
            # improved_team1 = [남자1, 여자1], improved_team2 = [남자2, 여자2]
            man1_skill = players_df[players_df['성명'] == improved_team1[0]]['실력'].values[0]
            woman1_skill = players_df[players_df['성명'] == improved_team1[1]]['실력'].values[0]
            man2_skill = players_df[players_df['성명'] == improved_team2[0]]['실력'].values[0]
            woman2_skill = players_df[players_df['성명'] == improved_team2[1]]['실력'].values[0]
            
            # 남자끼리 실력차, 여자끼리 실력차 계산
            men_skill_diff = abs(man1_skill - man2_skill)
            women_skill_diff = abs(woman1_skill - woman2_skill)
            
            # ★ 새로운 제약조건 6: 남자끼리 실력차 ≤1, 여자끼리 실력차 ≤1
            if men_skill_diff > 1.0 or women_skill_diff > 1.0:
                continue
            
            # ★ 새로운 제약조건 6: 우위 선수들이 같은 팀 금지 (혼복)
            # 남자 중 더 나은 선수, 여자 중 더 나은 선수 확인
            better_man = improved_team1[0] if man1_skill < man2_skill else improved_team2[0]
            better_woman = improved_team1[1] if woman1_skill < woman2_skill else improved_team2[1]
            
            # 우위 남자와 우위 여자가 같은 팀이면 제외
            if (better_man in improved_team1 and better_woman in improved_team1) or \
               (better_man in improved_team2 and better_woman in improved_team2):
                continue
            
            # 전체 실력차 = 남자 실력차 + 여자 실력차
            total_individual_diff = men_skill_diff + women_skill_diff
            
            # 팀 평균 실력 (출력용)
            team1_skill = (man1_skill + woman1_skill) / 2
            team2_skill = (man2_skill + woman2_skill) / 2
            team_avg_diff = abs(team1_skill - team2_skill)
            
            # 참여 횟수 편차 계산
            all_players = available_men + available_women
            participation_counts = [participation_count[p] for p in all_players]
            max_part = max(participation_counts)
            min_part = min(participation_counts)
            participation_variance = max_part - min_part
            
            # 점수: 팀평균 실력차 + 개인별 실력차 + 참여 횟수 편차
            score = team_avg_diff + total_individual_diff * 0.5 + participation_variance * 0.3
            
            # 모든 조합 저장 (참여 횟수 보장을 위해) - 개선된 팀 기준
            all_combos.append((improved_team1, improved_team2, team_avg_diff, score))
            
            # 실력차 제약을 만족하는 좋은 조합들만 따로 저장 (개선된 팀 기준)
            if men_skill_diff <= max_skill_diff and women_skill_diff <= max_skill_diff and total_individual_diff <= max_skill_diff:
                # 최적 점수 갱신
                if score < best_score:
                    best_score = score
                    good_combos = [(improved_team1, improved_team2, team_avg_diff, score)]
                # 최적 점수에서 15% 이내면 좋은 조합으로 추가
                elif score <= best_score * 1.15:
                    good_combos.append((improved_team1, improved_team2, team_avg_diff, score))
    
    # 좋은 조합이 있으면 그 중에서 랜덤 선택
    if good_combos:
        selected = random.choice(good_combos)
        return (selected[0], selected[1], selected[2])
    
    # 좋은 조합이 없으면 모든 조합 중 최선의 조합 선택 (참여 횟수 보장을 위해)
    if all_combos:
        all_combos.sort(key=lambda x: x[3])  # 점수로 정렬
        selected = all_combos[0]
        return (selected[0], selected[1], selected[2])
    
    return None

# 매칭 알고리즘 - 참여 횟수 밸런스 최우선
# random.seed(42)  # 재현성을 위한 시드 설정 - 주석 처리하여 매번 다른 결과 생성
random.seed()  # 현재 시간 기반 랜덤 시드

# 경기 수 설정
MIN_NAMBOK = 2  # 남복 최소 경기 수
EXACT_YEOBOK = 2  # 여복 정확한 경기 수 (고정)
MIN_HONBOK = 2  # 혼복 최소 경기 수

# 경기 타입별 카운터
match_type_count = {'남복': 0, '여복': 0, '혼복': 0}

# 각 타임별로 참여한 선수 추적 (중요!)
players_in_time_slot = {i: set() for i in range(1, NUM_TIMES + 1)}  # {time: set of player names}

# 타임별 참여 선수 추적
players_in_time_slot = {time: set() for time in range(1, NUM_TIMES + 1)}

# 경기 타입별 카운트
match_type_count = {'남복': 0, '여복': 0, '혼복': 0}

print("=== 매칭 생성 중... ===\n")

def can_participate(player_name, max_participation):
    """선수가 참여 가능한지 확인 (최대 참여 횟수 제한)"""
    return participation_count[player_name] < max_participation

def can_participate_in_time(player_name, time_slot):
    """해당 타임에 이미 참여했는지 확인"""
    return player_name not in players_in_time_slot.get(time_slot, set())

def get_balanced_players(gender_df, needed, participation_count, max_participation, current_time, match_type=None, min_participation=None):
    """참여 횟수가 적고, 현재 타임에 참여하지 않은 선수들 선택"""
    available = []
    low_participation_players = []  # 최소 참여 미달 선수
    
    for name in gender_df['성명']:
        if can_participate(name, max_participation) and can_participate_in_time(name, current_time):
            available.append(name)
            # 최소 참여 횟수 미달 선수 체크
            if min_participation and participation_count[name] < min_participation:
                low_participation_players.append(name)
    
    # 남자 선수이고 경기 타입이 지정된 경우, 남복/혼복 밸런스 고려
    if match_type in ['남복', '혼복'] and len(available) > 0:
        # 남자 선수인지 확인
        first_player = available[0]
        is_men = first_player in men_match_type_count
        
        if is_men:
            # 남복/혼복 밸런스를 고려한 정렬
            def get_priority_score(player_name):
                part_count = participation_count[player_name]
                nambok_count = men_match_type_count[player_name]['남복']
                honbok_count = men_match_type_count[player_name]['혼복']
                
                # ★ 최고우선: 최소 참여 미달 선수 (타임 4-5)
                if min_participation and part_count < min_participation:
                    return -10000000 + part_count * 100
                
                # 최최우선: 혼복 경기를 한 번도 못한 선수 (혼복 경기일 때만)
                if match_type == '혼복' and honbok_count == 0:
                    # 전체 참여 횟수가 적을수록 더 우선
                    return -1000000 + part_count * 100
                
                # 최우선: 해당 타입 경기를 한 번도 못한 선수
                if match_type == '남복' and nambok_count == 0:
                    return -100000 + part_count * 100
                
                # 총 참여 횟수를 최우선으로 (가장 큰 가중치)
                # 참여 횟수가 1 차이나면 점수가 5000 차이남
                participation_score = part_count * 5000
                
                # 경기 타입별 점수 (해당 타입 참여가 적을수록 우선, 중간 가중치)
                # 타입 횟수가 1 차이나면 점수가 1000 차이남
                if match_type == '남복':
                    type_score = nambok_count * 1000
                elif match_type == '혼복':
                    type_score = honbok_count * 1000
                else:
                    type_score = 0
                
                # 최종 점수: 총 참여 횟수 > 타입별 참여 횟수
                # 예: 총3회/남2회 = 15000 + 2000 = 17000
                #     총4회/남1회 = 20000 + 1000 = 21000 (총 참여가 많아서 후순위)
                return participation_score + type_score
            
            available.sort(key=get_priority_score)
            
            # 디버깅: 혼복일 때 상위 선수 정보 출력
            if match_type == '혼복' and len(available) >= 6:
                debug_info = []
                for i, name in enumerate(available[:6]):
                    score = get_priority_score(name)
                    nambok = men_match_type_count[name]['남복']
                    honbok = men_match_type_count[name]['혼복']
                    total = participation_count[name]
                    debug_info.append(f"{i+1}.{name}(남{nambok}/혼{honbok}/총{total}/점수{score})")
                print(f"    [디버그] 혼복 선수 정렬: {', '.join(debug_info)}")
        else:
            # 여자 선수는 참여 횟수로만 정렬 (최소 참여 미달 선수 최우선)
            def get_priority_score_women(player_name):
                part_count = participation_count[player_name]
                # 최소 참여 미달 선수 최우선
                if min_participation and part_count < min_participation:
                    return -10000000 + part_count * 100
                return part_count * 1000
            available.sort(key=get_priority_score_women)
    else:
        # 참여 횟수로 정렬 (최소 참여 미달 선수 최우선)
        def get_priority_score_general(player_name):
            part_count = participation_count[player_name]
            if min_participation and part_count < min_participation:
                return -10000000 + part_count * 100
            return part_count * 1000
        available.sort(key=get_priority_score_general)
    
    # 마지막 타임(4, 5)에서는 최소 참여 미달 선수를 맨 앞에 배치
    if min_participation and len(low_participation_players) > 0:
        # 미달 선수를 참여 횟수순으로 정렬
        low_participation_players.sort(key=lambda x: participation_count[x])
        # 미달 선수를 제외한 나머지
        other_players = [p for p in available if p not in low_participation_players]
        # 미달 선수를 앞에 배치
        available = low_participation_players + other_players
        if current_time >= 4:
            print(f"    [타임{current_time}] 최소 참여 미달 선수 우선: {', '.join(low_participation_players[:3])}")
    
    return available

def try_match_same_gender(available, players_df, participation_count, match_type, time, court, min_participation_threshold=None):
    """남복 또는 여복 매칭 시도 (공통 로직)"""
    # ★ 타임 4-5: 최소 참여 미달 선수 강제 포함
    forced_include_players = []
    if min_participation_threshold and time >= 4:
        forced_include_players = [p for p in available if participation_count[p] < min_participation_threshold]
        if len(forced_include_players) > 0:
            print(f"    → 최소 참여 미달 선수 강제 포함: {', '.join(forced_include_players[:3])}")
    
    # 0회 선수 확인 (남복만)
    zero_count_players = []
    if match_type == '남복':
        zero_count_players = [m for m in available if men_match_type_count[m]['남복'] == 0]
    
    # 참여 횟수가 적은 선수 확인
    low_participation_players = []
    if len(available) >= 4:
        min_participation = min(participation_count[p] for p in available[:6])
        low_participation_players = [p for p in available[:6] if participation_count[p] == min_participation]
    
    # 참여 횟수 차이 확인
    participation_gap = 0
    if len(available) >= 4:
        current_max = max(participation_count[p] for p in available[:4])
        current_min = min(participation_count[p] for p in available[:4])
        participation_gap = current_max - current_min
    
    # 실력차 제약 설정
    skill_diff_limit = 1.0 if participation_gap > 0 else 0.5
    
    best_match = None
    
    # 전략 0: 강제 포함 (타임 4-5, 최소 참여 미달 선수)
    if len(forced_include_players) >= 1:
        priority_set = forced_include_players[:min(3, len(forced_include_players))]
        remaining_available = [p for p in available[:10] if p not in priority_set]
        needed = 4 - len(priority_set)
        
        for other_players in combinations(remaining_available[:10], needed):
            four_players = priority_set + list(other_players)
            balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=skill_diff_limit)
            if balanced_teams:
                best_match = (four_players, balanced_teams)
                print(f"    ✓ 최소 참여 미달 선수 포함")
                break
        
        # 완화 1: 제약을 1.5로 완화
        if not best_match:
            for other_players in combinations(remaining_available[:10], needed):
                four_players = priority_set + list(other_players)
                balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=1.5)
                if balanced_teams:
                    best_match = (four_players, balanced_teams)
                    print(f"    ✓ 최소 참여 미달 선수 포함 (완화 1.5)")
                    break
        
        # 완화 2: 제약을 2.5로 완화
        if not best_match:
            for other_players in combinations(remaining_available[:10], needed):
                four_players = priority_set + list(other_players)
                balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=2.5)
                if balanced_teams:
                    best_match = (four_players, balanced_teams)
                    print(f"    ✓ 최소 참여 미달 선수 포함 (완화 2.5)")
                    break
        
        # 완화 3: 제약을 거의 무시 (5.0)
        if not best_match:
            for other_players in combinations(remaining_available[:10], needed):
                four_players = priority_set + list(other_players)
                balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=5.0)
                if balanced_teams:
                    best_match = (four_players, balanced_teams)
                    print(f"    ✓ 최소 참여 미달 선수 포함 (완화 5.0)")
                    break
    
    # 전략 1: 0회 선수 우선 (남복만)
    if not best_match and match_type == '남복' and len(zero_count_players) >= 1:
        print(f"    → 남복 0회 선수 우선: {', '.join(zero_count_players[:2])}")
        priority_set = zero_count_players[:min(2, len(zero_count_players))]
        remaining_available = [p for p in available[:10] if p not in priority_set]
        needed = 4 - len(priority_set)
        
        for other_players in combinations(remaining_available[:8], needed):
            four_players = priority_set + list(other_players)
            balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=skill_diff_limit)
            if balanced_teams:
                best_match = (four_players, balanced_teams)
                print(f"    ✓ 남복 0회 선수 포함")
                break
        
        # 완화된 제약으로 재시도
        if not best_match:
            for other_players in combinations(remaining_available[:8], needed):
                four_players = priority_set + list(other_players)
                balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=1.5)
                if balanced_teams:
                    best_match = (four_players, balanced_teams)
                    print(f"    ✓ 남복 0회 선수 포함 (완화)")
                    break
    
    # 전략 2: 참여 적은 선수 우선
    if not best_match and len(low_participation_players) >= 2:
        print(f"    → 참여 적은 선수 우선: {', '.join(low_participation_players[:3])}")
        priority_set = low_participation_players[:min(3, len(low_participation_players))]
        remaining_available = [p for p in available[:10] if p not in priority_set]
        needed = 4 - len(priority_set)
        for other_players in combinations(remaining_available[:8], needed):
            four_players = priority_set + list(other_players)
            balanced_teams = find_balanced_teams(four_players, players_df, participation_count, max_skill_diff=skill_diff_limit)
            if balanced_teams:
                best_match = (four_players, balanced_teams)
                print(f"    ✓ 참여 적은 선수 포함")
                break
    
    # 전략 3: 일반 매칭
    if not best_match:
        for candidate_group in combinations(available[:min(10, len(available))], 4):
            balanced_teams = find_balanced_teams(list(candidate_group), players_df, participation_count, max_skill_diff=skill_diff_limit)
            if balanced_teams:
                best_match = (list(candidate_group), balanced_teams)
                break
    
    # 전략 4: 완화된 제약
    if not best_match:
        print(f"    → 제약 완화 ({'1.5' if match_type == '남복' else '1.0'}) 매칭")
        relax_limit = 1.5 if match_type == '남복' else 1.0
        for candidate_group in combinations(available[:min(12, len(available))], 4):
            balanced_teams = find_balanced_teams(list(candidate_group), players_df, participation_count, max_skill_diff=relax_limit)
            if balanced_teams:
                best_match = (list(candidate_group), balanced_teams)
                break
    
    # 전략 5: 강제 매칭
    if not best_match and len(available) >= 4:
        print(f"    ⚠ 제약 무시 강제 매칭")
        four_players = available[:4]
        team1 = [four_players[0], four_players[1]]
        team2 = [four_players[2], four_players[3]]
        team1_skill = (players_df.loc[players_df['성명'] == team1[0], '실력'].values[0] + 
                     players_df.loc[players_df['성명'] == team1[1], '실력'].values[0]) / 2
        team2_skill = (players_df.loc[players_df['성명'] == team2[0], '실력'].values[0] + 
                     players_df.loc[players_df['성명'] == team2[1], '실력'].values[0]) / 2
        skill_diff = abs(team1_skill - team2_skill)
        best_match = (four_players, (team1, team2, skill_diff))
    
    return best_match

# 이상적인 참여 횟수 계산
total_player_slots = TOTAL_MATCHES * 4  # 15경기 * 4명
total_players = len(players)
ideal_participation = total_player_slots / total_players
max_participation = int(ideal_participation) + 1  # 천장값 + 1
min_participation = int(ideal_participation)      # 바닥값 (최소 참여 횟수)

print(f"총 선수 슬롯: {total_player_slots}, 총 선수: {total_players}")
print(f"이상적 참여 횟수: {ideal_participation:.2f}")
print(f"허용 범위: {min_participation}~{max_participation}회 (차이 최대 1)\n")

# 타임별 경기 타입 스케줄 정의
# 타임 1, 4: 남복 2경기 + 여복 1경기
# 타임 2, 3: 남복 1경기 + 혼복 2경기
# 타임 5: 남복 2경기 + 혼복 1경기
time_schedule = {
    1: ['남복', '남복', '여복'],  # 코트 1, 2, 3
    2: ['남복', '혼복', '혼복'],  # 코트 1, 2, 3
    3: ['남복', '혼복', '혼복'],  # 코트 1, 2, 3
    4: ['남복', '남복', '여복'],  # 코트 1, 2, 3
    5: ['남복', '남복', '혼복'],  # 코트 1, 2, 3
}

print("=" * 80)
print("=== 타임별 경기 타입 스케줄 ===")
print("=" * 80)
for time, types in time_schedule.items():
    print(f"타임 {time}: {' / '.join([f'코트{i+1}={t}' for i, t in enumerate(types)])}")
print(f"\n총 경기 수 - 남복: 8경기, 여복: 2경기, 혼복: 5경기")
print()

# ============================================================================
# 1단계: 모든 혼복 경기를 먼저 배정 (타임 무관하게 전체 혼복 경기 구성)
# ============================================================================
print("=" * 80)
print("=== 1단계: 혼복 경기 우선 배정 (총 5경기) ===")
print("=" * 80)

# 혼복 경기 위치 수집
honbok_matches_plan = []  # (time, court) 튜플 저장
for time, types in time_schedule.items():
    for court_idx, match_type in enumerate(types):
        if match_type == '혼복':
            honbok_matches_plan.append((time, court_idx + 1))

# 혼복 경기를 역순으로 배정 (나중 타임부터)
# 이렇게 하면 참여 적은 선수들이 먼저 혼복에 배정되고,
# 참여 많은 선수들이 나중에 혼복에 배정되어 전체 밸런스가 개선됨
honbok_matches_plan.sort(reverse=True)  # (5,3), (3,3), (3,2), (2,3), (2,2) 순서

print(f"혼복 경기 위치 (역순 배정): {honbok_matches_plan}\n")

# 혼복 경기 배정
honbok_assigned = []  # (time, court, match_data) 저장

for match_idx, (time, court) in enumerate(honbok_matches_plan):
    print(f"혼복 경기 {match_idx + 1}/5 배정 중 (타임{time} 코트{court})...")
    
    # 해당 타임에 이미 배정된 선수는 제외
    # 마지막 혼복 경기(타임 5)에서는 min_participation 적용
    apply_min_participation = min_participation if time >= 4 else None
    available_men = get_balanced_players(men, 10, participation_count, max_participation, time, '혼복', apply_min_participation)
    available_women = get_balanced_players(women, 6, participation_count, max_participation, time, None, apply_min_participation)
    
    # 혼복 0회 선수 확인
    zero_honbok_players = [m for m in available_men if men_match_type_count[m]['혼복'] == 0]
    
    print(f"  Available: 남자 {len(available_men)}명 (혼복0회: {len(zero_honbok_players)}명), 여자 {len(available_women)}명")
    
    # 참여 횟수 차이가 크면 실력차 제약을 완화
    all_available = (available_men[:2] if len(available_men) >= 2 else []) + \
                  (available_women[:2] if len(available_women) >= 2 else [])
    if len(all_available) >= 4:
        current_max = max(participation_count[p] for p in all_available)
        current_min = min(participation_count[p] for p in all_available)
        participation_gap = current_max - current_min
    else:
        participation_gap = 0
    
    # 참여 횟수 차이가 크면 실력차 제약을 완화
    if participation_gap > 0:
        skill_diff_limit = 1.0
    else:
        skill_diff_limit = 0.5
    
    # 최적의 조합 찾기
    best_match = None
    relaxed_skill_diff = 1.5
    
    # 전략 1: 혼복 0회 선수가 있으면 우선 포함
    if len(zero_honbok_players) >= 1:
        print(f"  → 혼복 0회 선수 우선: {', '.join(zero_honbok_players[:3])}")
        
        # 1-1: 혼복 0회 선수 2명 동시 배정 시도
        if len(zero_honbok_players) >= 2:
            men_combo = zero_honbok_players[:2]
            for women_combo in combinations(available_women[:min(6, len(available_women))], 2):
                balanced_teams = find_balanced_mixed_teams(men_combo, list(women_combo), players, participation_count, max_skill_diff=skill_diff_limit)
                if not balanced_teams:
                    balanced_teams = find_balanced_mixed_teams(men_combo, list(women_combo), players, participation_count, max_skill_diff=relaxed_skill_diff)
                if balanced_teams:
                    best_match = (men_combo, list(women_combo), balanced_teams)
                    print(f"  ✓ 혼복 0회 2명 포함: {', '.join(men_combo)}")
                    break
        
        # 1-2: 개별 시도
        if not best_match:
            for priority_player in zero_honbok_players[:3]:
                other_men = [m for m in available_men[:10] if m != priority_player]
                for other_man in other_men[:8]:
                    men_combo = [priority_player, other_man]
                    for women_combo in combinations(available_women[:min(6, len(available_women))], 2):
                        balanced_teams = find_balanced_mixed_teams(men_combo, list(women_combo), players, participation_count, max_skill_diff=skill_diff_limit)
                        if not balanced_teams:
                            balanced_teams = find_balanced_mixed_teams(men_combo, list(women_combo), players, participation_count, max_skill_diff=relaxed_skill_diff)
                        if balanced_teams:
                            best_match = (men_combo, list(women_combo), balanced_teams)
                            print(f"  ✓ 혼복 0회 1명 포함: {priority_player}")
                            break
                    if best_match:
                        break
                if best_match:
                    break
        
        # 1-3: 강제 배정
        if not best_match and len(zero_honbok_players) >= 1:
            priority_player = zero_honbok_players[0]
            other_men = [m for m in available_men[:10] if m != priority_player]
            if len(other_men) >= 1 and len(available_women) >= 2:
                selected_men = [priority_player, other_men[0]]
                selected_women = available_women[:2]
                team1 = [selected_men[0], selected_women[0]]
                team2 = [selected_men[1], selected_women[1]]
                team1_skill = (players.loc[players['성명'] == team1[0], '실력'].values[0] + 
                             players.loc[players['성명'] == team1[1], '실력'].values[0]) / 2
                team2_skill = (players.loc[players['성명'] == team2[0], '실력'].values[0] + 
                             players.loc[players['성명'] == team2[1], '실력'].values[0]) / 2
                skill_diff = abs(team1_skill - team2_skill)
                best_match = (selected_men, selected_women, (team1, team2, skill_diff))
                print(f"  ✓ 혼복 0회 강제 배정: {priority_player}")
    
    # 전략 2: 일반 매칭
    if not best_match:
        for men_combo in combinations(available_men[:min(10, len(available_men))], 2):
            for women_combo in combinations(available_women[:min(6, len(available_women))], 2):
                balanced_teams = find_balanced_mixed_teams(list(men_combo), list(women_combo), players, participation_count, max_skill_diff=skill_diff_limit)
                if balanced_teams:
                    best_match = (list(men_combo), list(women_combo), balanced_teams)
                    break
            if best_match:
                break
    
    # Fallback
    if not best_match:
        if len(available_men) >= 2 and len(available_women) >= 2:
            zero_honbok_in_available = [m for m in available_men[:6] if men_match_type_count[m]['혼복'] == 0]
            if len(zero_honbok_in_available) >= 2:
                selected_men = zero_honbok_in_available[:2]
            elif len(zero_honbok_in_available) == 1:
                selected_men = [zero_honbok_in_available[0], available_men[0] if available_men[0] != zero_honbok_in_available[0] else available_men[1]]
            else:
                selected_men = available_men[:2]
            
            selected_women = available_women[:2]
            team1 = [selected_men[0], selected_women[0]]
            team2 = [selected_men[1], selected_women[1]]
            match = create_match(team1[0], team1[1], team2[0], team2[1], '혼복', court, time)
            best_match = (selected_men, selected_women, (team1, team2, 0))
            print(f"  ✓ Fallback 배정")
    
    if best_match:
        selected_men, selected_women, (team1, team2, skill_diff) = best_match
        match = create_match(team1[0], team1[1], team2[0], team2[1], '혼복', court, time)
        
        # matches 리스트에 추가 (중요!)
        matches.append(match)
        match_type_count['혼복'] += 1
        
        honbok_assigned.append((time, court, match))
        
        add_team_history(team1[0], team1[1])
        add_team_history(team2[0], team2[1])
        add_opponent_history(team1, team2)
        add_match_together_history(team1 + team2)
        
        for player in selected_men + selected_women:
            participation_count[player] += 1
            players_in_time_slot[time].add(player)
        
        for player in selected_men:
            men_match_type_count[player]['혼복'] += 1
        
        print(f"  ✓ 배정: {' & '.join(team1)} vs {' & '.join(team2)}\n")
    else:
        print(f"  ✗ 혼복 경기 생성 실패! (타임{time} 코트{court})\n")

# 혼복 경기 수 체크
expected_honbok = sum(1 for t in range(1, NUM_TIMES + 1) for match_type in time_schedule[t] if match_type == '혼복')
actual_honbok = len(honbok_assigned)
print(f"혼복 경기 배정 완료: {actual_honbok}/{expected_honbok}개\n")

if actual_honbok < expected_honbok:
    print(f"⚠️  경고: 혼복 경기가 {expected_honbok - actual_honbok}개 부족합니다!")
    print(f"    → 제약을 완화하여 재시도가 필요할 수 있습니다.\n")

# ============================================================================
# 2단계: 각 타임별로 남복/여복 경기 배정
# ============================================================================
print("=" * 80)
print("=== 2단계: 남복/여복 경기 배정 ===")
print("=" * 80)

for time in range(1, NUM_TIMES + 1):
    print(f"\n[타임 {time}] 남복/여복 경기 배정")
    
    # 타임 4-5에서는 min_participation 적용
    apply_min_participation = min_participation if time >= 4 else None
    
    for court in range(1, NUM_COURTS + 1):
        match_type = time_schedule[time][court - 1]
        
        # 혼복은 이미 1단계에서 배정되었으므로 스킵
        if match_type == '혼복':
            continue
        if match_type == '남복':
            # 남복: 남자 4명
            available = get_balanced_players(men, 10, participation_count, max_participation, time, '남복', apply_min_participation)
            
            print(f"  타임{time} 코트{court}: {match_type} 경기 생성 중...")
            best_match = try_match_same_gender(available, players, participation_count, '남복', time, court, apply_min_participation)
            
            # 매칭 성공 시 기록
            if best_match:
                selected, (team1, team2, skill_diff) = best_match
                match = create_match(team1[0], team1[1], team2[0], team2[1], '남복', court, time)
                matches.append(match)
                match_type_count['남복'] += 1
                add_team_history(team1[0], team1[1])
                add_team_history(team2[0], team2[1])
                add_opponent_history(team1, team2)
                add_match_together_history(team1 + team2)
                for player in selected:
                    participation_count[player] += 1
                    players_in_time_slot[time].add(player)
                    men_match_type_count[player]['남복'] += 1
                print(f"    ✓ 남복 경기 생성 완료")
            else:
                print(f"    ✗ 남복 경기 생성 실패")
        
        elif match_type == '여복':
            # 여복: 여자 4명
            available = get_balanced_players(women, 10, participation_count, max_participation, time, None, apply_min_participation)
            
            print(f"  타임{time} 코트{court}: {match_type} 경기 생성 중...")
            best_match = try_match_same_gender(available, players, participation_count, '여복', time, court, apply_min_participation)
            
            # 매칭 성공 시 기록
            if best_match:
                selected, (team1, team2, skill_diff) = best_match
                match = create_match(team1[0], team1[1], team2[0], team2[1], '여복', court, time)
                matches.append(match)
                match_type_count['여복'] += 1
                add_team_history(team1[0], team1[1])
                add_team_history(team2[0], team2[1])
                add_opponent_history(team1, team2)
                add_match_together_history(team1 + team2)
                for player in selected:
                    participation_count[player] += 1
                    players_in_time_slot[time].add(player)
                print(f"    ✓ 여복 경기 생성 완료")
            else:
                print(f"    ✗ 여복 경기 생성 실패 (타임{time} 코트{court})")

# ============================================================================
# 최종 경기 수 검증
# ============================================================================
print("\n" + "=" * 80)
print("=== 경기 생성 결과 검증 ===")
print("=" * 80)

expected_matches = {
    '남복': sum(1 for t in range(1, NUM_TIMES + 1) for mt in time_schedule[t] if mt == '남복'),
    '여복': sum(1 for t in range(1, NUM_TIMES + 1) for mt in time_schedule[t] if mt == '여복'),
    '혼복': sum(1 for t in range(1, NUM_TIMES + 1) for mt in time_schedule[t] if mt == '혼복')
}

actual_matches = {
    '남복': match_type_count['남복'],
    '여복': match_type_count['여복'],
    '혼복': match_type_count['혼복']
}

total_expected = sum(expected_matches.values())
total_actual = len(matches)

print(f"예상 경기 수: {total_expected}개 (남복 {expected_matches['남복']}, 여복 {expected_matches['여복']}, 혼복 {expected_matches['혼복']})")
print(f"실제 생성 수: {total_actual}개 (남복 {actual_matches['남복']}, 여복 {actual_matches['여복']}, 혼복 {actual_matches['혼복']})")

if total_actual < total_expected:
    print(f"\n⚠️  경고: 경기가 {total_expected - total_actual}개 부족합니다!")
    for match_type in ['남복', '여복', '혼복']:
        if actual_matches[match_type] < expected_matches[match_type]:
            shortage = expected_matches[match_type] - actual_matches[match_type]
            print(f"  - {match_type}: {shortage}개 부족")
    print("\n해결 방법:")
    print("  1. 제약 조건을 완화하여 재실행")
    print("  2. 참가자 수를 늘리기")
    print("  3. 경기 타입 구성을 조정")
elif total_actual == total_expected:
    print(f"\n✅ 모든 경기가 정상적으로 생성되었습니다!")
else:
    print(f"\n⚠️  경고: 예상보다 {total_actual - total_expected}개 더 많이 생성되었습니다!")



# 코트 교환 로직: 남복은 코트1로, 여복은 코트3으로 최대한 배치
print("=" * 80)
print("=== 코트 교환 (남복→코트1, 여복→코트3) ===")
print("=" * 80)

for time in range(1, NUM_TIMES + 1):
    # 해당 타임의 모든 코트 매치 찾기
    court_matches = {}
    
    for idx, match in enumerate(matches):
        if match['time'] == time:
            court_matches[match['court']] = (idx, match)
    
    # 코트1에 여복이 있으면 다른 코트와 교환
    if 1 in court_matches and court_matches[1][1]['type'] == '여복':
        # 코트3에 남복이나 혼복이 있으면 우선 교환
        if 3 in court_matches and court_matches[3][1]['type'] in ['남복', '혼복']:
            matches[court_matches[1][0]]['court'] = 3
            matches[court_matches[3][0]]['court'] = 1
            print(f"타임{time}: 코트1(여복) ↔ 코트3({court_matches[3][1]['type']}) 교환")
        # 코트2에 남복이나 혼복이 있으면 교환
        elif 2 in court_matches and court_matches[2][1]['type'] in ['남복', '혼복']:
            matches[court_matches[1][0]]['court'] = 2
            matches[court_matches[2][0]]['court'] = 1
            print(f"타임{time}: 코트1(여복) ↔ 코트2({court_matches[2][1]['type']}) 교환")
    
    # 재확인: 코트2에 여복이 있고 코트3에 남복이나 혼복이 있으면 교환
    court_matches = {}
    for idx, match in enumerate(matches):
        if match['time'] == time:
            court_matches[match['court']] = (idx, match)
    
    if 2 in court_matches and court_matches[2][1]['type'] == '여복':
        if 3 in court_matches and court_matches[3][1]['type'] in ['남복', '혼복']:
            matches[court_matches[2][0]]['court'] = 3
            matches[court_matches[3][0]]['court'] = 2
            print(f"타임{time}: 코트2(여복) ↔ 코트3({court_matches[3][1]['type']}) 교환")
    
    # 추가: 코트3에 남복이 있고 코트1에 혼복이 있으면 교환 (남복을 1로)
    court_matches = {}
    for idx, match in enumerate(matches):
        if match['time'] == time:
            court_matches[match['court']] = (idx, match)
    
    if 1 in court_matches and 3 in court_matches:
        if court_matches[1][1]['type'] == '혼복' and court_matches[3][1]['type'] == '남복':
            matches[court_matches[1][0]]['court'] = 3
            matches[court_matches[3][0]]['court'] = 1
            print(f"타임{time}: 코트1(혼복) ↔ 코트3(남복) 교환")

print("\n코트 교환 완료!\n")

# 결과 출력 (타임-코트 순으로 정렬)
print("=" * 80)
print("=== 매칭 결과 ===")
print("=" * 80)

sorted_matches = sorted(matches, key=lambda x: (x['time'], x['court']))
for match in sorted_matches:
    print(f"\n코트 {match['court']} | 타임 {match['time']} | {match['type']}")
    print(f"  팀1: {match['team1'][0]} & {match['team1'][1]} (평균 실력: {match['team1_skill']:.1f})")
    print(f"  팀2: {match['team2'][0]} & {match['team2'][1]} (평균 실력: {match['team2_skill']:.1f})")
    print(f"  실력차: {match['skill_diff']:.2f}")

# 통계 출력
print("\n" + "=" * 80)
print("=== 참여 통계 ===")
print("=" * 80)

participation_df = pd.DataFrame([
    {'성명': name, '참여횟수': count, 
     '성별': players[players['성명'] == name]['성별'].values[0],
     '실력': players[players['성명'] == name]['실력'].values[0]}
    for name, count in participation_count.items()
]).sort_values('참여횟수', ascending=False)

print(participation_df.to_string(index=False))

print(f"\n평균 참여 횟수: {participation_df['참여횟수'].mean():.2f}")
print(f"최대 참여 횟수: {participation_df['참여횟수'].max()}")
print(f"최소 참여 횟수: {participation_df['참여횟수'].min()}")

# 남자 선수들의 남복/혼복 참여 통계
print("\n" + "=" * 80)
print("=== 남자 선수 경기 타입별 참여 통계 ===")
print("=" * 80)
men_type_stats = []
for name in men['성명']:
    men_type_stats.append({
        '성명': name,
        '남복': men_match_type_count[name]['남복'],
        '혼복': men_match_type_count[name]['혼복'],
        '총참여': men_match_type_count[name]['남복'] + men_match_type_count[name]['혼복']
    })
men_type_df = pd.DataFrame(men_type_stats).sort_values('총참여', ascending=False)
print(men_type_df.to_string(index=False))
print(f"\n평균 남복 참여: {men_type_df['남복'].mean():.2f}회")
print(f"평균 혼복 참여: {men_type_df['혼복'].mean():.2f}회")

# 남복/혼복 밸런스 체크
print("\n[남복/혼복 밸런스 분석]")
zero_count_players = []
imbalanced_players = []
for name in men['성명']:
    nambok = men_match_type_count[name]['남복']
    honbok = men_match_type_count[name]['혼복']
    diff = abs(nambok - honbok)
    total = nambok + honbok
    
    # 한 타입도 참여하지 못한 경우
    if nambok == 0 or honbok == 0:
        zero_count_players.append((name, nambok, honbok))
    # 차이가 1보다 크면 불균형
    elif total > 0 and diff > 1:
        imbalanced_players.append((name, nambok, honbok, diff))

if zero_count_players:
    print(f"🚨 한 타입 경기에 전혀 참여하지 못한 선수 ({len(zero_count_players)}명):")
    for name, nambok, honbok in zero_count_players:
        if nambok == 0:
            print(f"   {name}: 남복 0회, 혼복 {honbok}회")
        else:
            print(f"   {name}: 남복 {nambok}회, 혼복 0회")

if imbalanced_players:
    print(f"⚠ 밸런스가 맞지 않는 선수 ({len(imbalanced_players)}명):")
    for name, nambok, honbok, diff in sorted(imbalanced_players, key=lambda x: x[3], reverse=True):
        print(f"   {name}: 남복 {nambok}회, 혼복 {honbok}회 (차이: {diff})")

if not zero_count_players and not imbalanced_players:
    print("✅ 모든 남자 선수가 남복/혼복 밸런스를 유지하고 있습니다.")

# 매치 타입별 통계
print("\n" + "=" * 80)
print("=== 경기 타입별 통계 ===")
print("=" * 80)
if len(matches) > 0:
    type_counts = pd.DataFrame(matches)['type'].value_counts()
    print(type_counts)
    
    # 평균 실력차 계산
    avg_skill_diff = sum(m['skill_diff'] for m in matches) / len(matches)
    
    # 개인 간 실력차 계산
    total_top_diff = 0
    total_bottom_diff = 0
    for match in matches:
        team1_skills = [players[players['성명'] == p]['실력'].values[0] for p in match['team1']]
        team2_skills = [players[players['성명'] == p]['실력'].values[0] for p in match['team2']]
        team1_sorted = sorted(team1_skills, reverse=True)
        team2_sorted = sorted(team2_skills, reverse=True)
        total_top_diff += abs(team1_sorted[0] - team2_sorted[0])
        total_bottom_diff += abs(team1_sorted[1] - team2_sorted[1])
    
    avg_top_diff = total_top_diff / len(matches)
    avg_bottom_diff = total_bottom_diff / len(matches)
    
    print(f"\n평균 팀간 실력차: {avg_skill_diff:.2f}")
    print(f"평균 상위선수 실력차: {avg_top_diff:.2f}")
    print(f"평균 하위선수 실력차: {avg_bottom_diff:.2f}")
else:
    print("생성된 매치가 없습니다.")

# 같이 경기한 횟수 통계
print("\n" + "=" * 80)
print("=== 같이 경기한 횟수 검증 (최대 3번) ===")
print("=" * 80)
max_together_violations = []
for p1 in match_together_history:
    for p2, count in match_together_history[p1].items():
        if count > 3:
            max_together_violations.append((p1, p2, count))

if max_together_violations:
    print(f"⚠ 경고: {len(max_together_violations)}개의 위반 발견!")
    for p1, p2, count in sorted(max_together_violations, key=lambda x: x[2], reverse=True)[:10]:
        print(f"   {p1} ↔ {p2}: {count}회")
else:
    print("✅ 모든 선수 쌍이 최대 3번 이하로 같이 경기했습니다.")

# 같은 팀 횟수 통계
print("\n" + "=" * 80)
print("=== 같은 팀 횟수 검증 (최대 2번) ===")
print("=" * 80)
max_team_violations = []
for p1 in team_history:
    for p2, count in team_history[p1].items():
        if count > 2:
            max_team_violations.append((p1, p2, count))

if max_team_violations:
    print(f"⚠ 경고: {len(max_team_violations)}개의 위반 발견!")
    for p1, p2, count in sorted(max_team_violations, key=lambda x: x[2], reverse=True)[:10]:
        print(f"   {p1} - {p2}: {count}회")
else:
    print("✅ 모든 팀 쌍이 최대 2번 이하로 같은 팀이 되었습니다.")

# 적으로 만난 횟수 통계
print("\n" + "=" * 80)
print("=== 적으로 만난 횟수 검증 (최대 3번) ===")
print("=" * 80)
max_opponent_violations = []
for p1 in opponent_history:
    for p2, count in opponent_history[p1].items():
        if count > 3:
            max_opponent_violations.append((p1, p2, count))

if max_opponent_violations:
    print(f"⚠ 경고: {len(max_opponent_violations)}개의 위반 발견!")
    for p1, p2, count in sorted(max_opponent_violations, key=lambda x: x[2], reverse=True)[:10]:
        print(f"   {p1} vs {p2}: {count}회")
else:
    print("✅ 모든 선수 쌍이 최대 3번 이하로 적으로 만났습니다.")

# 엑셀 파일로 저장
print("\n" + "=" * 80)
print("=== 엑셀 파일 저장 중... ===")
print("=" * 80)

# 1. 매칭 결과 시트 데이터 생성
match_records = []
for match in matches:
    # 각 팀의 선수 실력 가져오기
    team1_player1_skill = players[players['성명'] == match['team1'][0]]['실력'].values[0]
    team1_player2_skill = players[players['성명'] == match['team1'][1]]['실력'].values[0]
    team2_player1_skill = players[players['성명'] == match['team2'][0]]['실력'].values[0]
    team2_player2_skill = players[players['성명'] == match['team2'][1]]['실력'].values[0]
    
    # 각 팀의 상위/하위 선수 구분
    team1_skills_sorted = sorted([team1_player1_skill, team1_player2_skill], reverse=True)
    team2_skills_sorted = sorted([team2_player1_skill, team2_player2_skill], reverse=True)
    
    # 상위 선수 간 차이, 하위 선수 간 차이 계산
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

# 2. 참여 통계 시트 데이터
participation_records = []
for name, count in participation_count.items():
    player_info = players[players['성명'] == name].iloc[0]
    record = {
        '성명': name,
        '성별': '남' if player_info['성별'] == 1 else '여',
        '실력': player_info['실력'],
        '참여횟수': count
    }
    
    # 남자 선수의 경우 남복/혼복 참여 횟수 추가
    if player_info['성별'] == 1:
        record['남복'] = men_match_type_count[name]['남복']
        record['혼복'] = men_match_type_count[name]['혼복']
    else:
        record['남복'] = '-'
        record['혼복'] = '-'
    
    participation_records.append(record)

participation_df = pd.DataFrame(participation_records).sort_values('참여횟수', ascending=False)

# 3. 코트별 타임표 시트 생성 (보기 좋은 형태)
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

# 4. 통계 요약 시트
# 개인 간 실력차 계산
if len(matches) > 0:
    total_top_diff_summary = 0
    total_bottom_diff_summary = 0
    for match in matches:
        team1_skills = [players[players['성명'] == p]['실력'].values[0] for p in match['team1']]
        team2_skills = [players[players['성명'] == p]['실력'].values[0] for p in match['team2']]
        team1_sorted = sorted(team1_skills, reverse=True)
        team2_sorted = sorted(team2_skills, reverse=True)
        total_top_diff_summary += abs(team1_sorted[0] - team2_sorted[0])
        total_bottom_diff_summary += abs(team1_sorted[1] - team2_sorted[1])
    
    avg_top_diff_summary = round(total_top_diff_summary / len(matches), 2)
    avg_bottom_diff_summary = round(total_bottom_diff_summary / len(matches), 2)
    avg_skill_diff_summary = round(sum(m['skill_diff'] for m in matches) / len(matches), 2)
else:
    avg_top_diff_summary = 0
    avg_bottom_diff_summary = 0
    avg_skill_diff_summary = 0

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
        avg_skill_diff_summary,
        avg_top_diff_summary,
        avg_bottom_diff_summary
    ]
}
summary_df = pd.DataFrame(summary_data)

# 엑셀 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'./results/테니스_매칭결과_{timestamp}.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 각 시트 저장
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

# PDF 생성 (reportlab 사용)
print("\n" + "=" * 80)
print("=== PDF 타임표 생성 중... ===")
print("=" * 80)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 한글 폰트 등록 (맑은 고딕)
    try:
        pdfmetrics.registerFont(TTFont('MalgunGothic', 'C:/Windows/Fonts/malgun.ttf'))
        font_name = 'MalgunGothic'
    except:
        font_name = 'Helvetica'  # 폴백 폰트
    
    pdf_file = f'테니스_타임표.pdf'
    doc = SimpleDocTemplate(pdf_file, pagesize=landscape(A4),
                           rightMargin=30, leftMargin=30,
                           topMargin=30, bottomMargin=18)
    
    # 스타일 설정
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=20,
        alignment=1,  # 중앙 정렬
        spaceAfter=30
    )
    
    # PDF 내용 구성
    story = []
    
    # 제목
    title = Paragraph(f"테니스 대회 타임표", title_style)
    story.append(title)
    
    # 타임표 데이터 준비
    table_data = []
    
    # 헤더
    header = ['타임']
    for court in range(1, NUM_COURTS + 1):
        header.append(f'코트 {court}')
    table_data.append(header)
    
    # 각 타임별 경기 정보
    for time in range(1, NUM_TIMES + 1):
        row = [f'타임 {time}']
        for court in range(1, NUM_COURTS + 1):
            match = next((m for m in matches if m['court'] == court and m['time'] == time), None)
            if match:
                match_info = f"[{match['type']}]\n{match['team1'][0]} & {match['team1'][1]}\nvs\n{match['team2'][0]} & {match['team2'][1]}"
                row.append(match_info)
            else:
                row.append('-')
        table_data.append(row)
    
    # 테이블 생성
    table = Table(table_data, colWidths=[1*inch] + [2.5*inch]*NUM_COURTS)
    
    # 테이블 스타일
    table.setStyle(TableStyle([
        # 헤더 스타일
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        # 테두리
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        # 타임 열 배경색
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
    ]))
    
    story.append(table)
    
    # 통계 정보 추가
    story.append(Spacer(1, 0.3*inch))
    
    stats_style = ParagraphStyle(
        'Stats',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        alignment=0
    )
    
    stats_text = f"""
    <b>대회 정보</b><br/>
    총 참가자: {len(players)}명 (남자: {len(men)}명, 여자: {len(women)}명)<br/>
    총 경기 수: {len(matches)}경기 (남복: {sum(1 for m in matches if m['type'] == '남복')}경기, 
    여복: {sum(1 for m in matches if m['type'] == '여복')}경기, 
    혼복: {sum(1 for m in matches if m['type'] == '혼복')}경기)<br/>
    평균 참여 횟수: {participation_df['참여횟수'].mean():.2f}회<br/>
    생성일시: {datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")}
    """
    
    stats_para = Paragraph(stats_text, stats_style)
    story.append(stats_para)
    
    # PDF 빌드
    doc.build(story)
    
    print(f"\n✅ 타임표 PDF가 '{pdf_file}' 파일로 저장되었습니다!")

except ImportError:
    print("\n⚠ PDF 생성을 위해 reportlab 라이브러리가 필요합니다.")
    print("   다음 명령어로 설치하세요: pip install reportlab")
except Exception as e:
    print(f"\n❌ PDF 생성 중 오류 발생: {e}")

