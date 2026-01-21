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

# 남자 선수들의 경기 타입별 참여 횟수 추적 (남복 vs 혼복 밸런스)
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
        
        # 팀 이력 체크: 같은 팀을 3번 이상 했는지 확인
        if has_teamed_before(team1_combo[0], team1_combo[1], max_team_count=3):
            continue
        if has_teamed_before(team2_combo[0], team2_combo[1], max_team_count=3):
            continue
        
        # 적 이력 체크: 적으로 4번 이상 만났는지 확인
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
        
        # 팀 밸런스 개선: 한쪽 팀이 상위/하위 모두 우월하면 하위 선수 교환
        improved_team1, improved_team2, was_improved = improve_team_balance(list(team1_combo), list(team2_combo), players_df)
        
        # 교환 후 팀 이력 다시 체크
        if was_improved:
            if has_teamed_before(improved_team1[0], improved_team1[1], max_team_count=3):
                continue
            if has_teamed_before(improved_team2[0], improved_team2[1], max_team_count=3):
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
        
        individual_diff = top_diff + bottom_diff
        
        # 개인 간 실력차 합이 2.0 이상이면 제외 (강제 조건)
        # 예: top_diff=0.5, bottom_diff=1.4 → 합=1.9 (OK)
        #     top_diff=1.0, bottom_diff=1.0 → 합=2.0 (제외)
        if individual_diff >= 2.0:
            continue
        
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
            
            # 팀 이력 체크: 같은 팀을 3번 이상 했는지 확인
            if has_teamed_before(team1[0], team1[1], max_team_count=3):
                continue
            if has_teamed_before(team2[0], team2[1], max_team_count=3):
                continue
            
            # 적 이력 체크: 적으로 4번 이상 만났는지 확인
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
            
            # 혼복 팀 밸런스 개선: 남자끼리, 여자끼리 비교하여 교환
            improved_team1, improved_team2, was_improved = improve_mixed_team_balance(team1, team2, players_df)
            
            # 교환 후 팀 이력 다시 체크
            if was_improved:
                if has_teamed_before(improved_team1[0], improved_team1[1], max_team_count=3):
                    continue
                if has_teamed_before(improved_team2[0], improved_team2[1], max_team_count=3):
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
            
            # 혼복은 남자끼리, 여자끼리 실력 비교 (개선된 팀 기준)
            # improved_team1 = [남자1, 여자1], improved_team2 = [남자2, 여자2]
            man1_skill = players_df[players_df['성명'] == improved_team1[0]]['실력'].values[0]
            woman1_skill = players_df[players_df['성명'] == improved_team1[1]]['실력'].values[0]
            man2_skill = players_df[players_df['성명'] == improved_team2[0]]['실력'].values[0]
            woman2_skill = players_df[players_df['성명'] == improved_team2[1]]['실력'].values[0]
            
            # 남자끼리 실력차, 여자끼리 실력차 계산
            men_skill_diff = abs(man1_skill - man2_skill)
            women_skill_diff = abs(woman1_skill - woman2_skill)
            
            # 전체 실력차 = 남자 실력차 + 여자 실력차
            total_individual_diff = men_skill_diff + women_skill_diff
            
            # 개인 간 실력차 합이 2.0 이상이면 제외 (강제 조건)
            # 예: men_diff=0.5, women_diff=1.4 → 합=1.9 (OK)
            #     men_diff=1.0, women_diff=1.0 → 합=2.0 (제외)
            if total_individual_diff >= 2.0:
                continue
            
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
players_in_time_slot = {}  # {time: set of player names}

# 팀 매칭 이력 추적 (같은 팀 2번까지 허용)
team_history = {}  # {player_name: {teammate: count}}
for name in players['성명']:
    team_history[name] = {}

# 적 매칭 이력 추적 (적으로 만난 횟수 2번까지 허용)
opponent_history = {}  # {player_name: {opponent: count}}
for name in players['성명']:
    opponent_history[name] = {}

print("=== 매칭 생성 중... ===\n")

def can_participate(player_name, max_participation):
    """선수가 참여 가능한지 확인 (최대 참여 횟수 제한)"""
    return participation_count[player_name] < max_participation

def has_teamed_before(player1, player2, max_team_count=3):
    """두 선수가 이전에 같은 팀을 max_team_count번 이상 했는지 확인"""
    return team_history[player1].get(player2, 0) >= max_team_count

def has_faced_before(player1, player2, max_opponent_count=3):
    """두 선수가 이전에 적으로 max_opponent_count번 이상 만났는지 확인"""
    return opponent_history[player1].get(player2, 0) >= max_opponent_count

def add_team_history(player1, player2):
    """팀 이력에 추가"""
    team_history[player1][player2] = team_history[player1].get(player2, 0) + 1
    team_history[player2][player1] = team_history[player2].get(player1, 0) + 1

def add_opponent_history(team1, team2):
    """적 이력에 추가 (team1의 각 선수 vs team2의 각 선수)"""
    for p1 in team1:
        for p2 in team2:
            opponent_history[p1][p2] = opponent_history[p1].get(p2, 0) + 1
            opponent_history[p2][p1] = opponent_history[p2].get(p1, 0) + 1

def can_participate_in_time(player_name, time_slot):
    """해당 타임에 이미 참여했는지 확인"""
    return player_name not in players_in_time_slot.get(time_slot, set())

def get_balanced_players(gender_df, needed, participation_count, max_participation, current_time, match_type=None):
    """참여 횟수가 적고, 현재 타임에 참여하지 않은 선수들 선택"""
    available = []
    for name in gender_df['성명']:
        if can_participate(name, max_participation) and can_participate_in_time(name, current_time):
            available.append(name)
    
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
                
                # 기본 점수: 참여 횟수 (적을수록 우선)
                score = part_count * 10
                
                # 경기 타입별 가중치 추가
                if match_type == '남복':
                    # 남복이 적을수록 우선순위 높음
                    score += nambok_count * 5 - honbok_count * 2
                elif match_type == '혼복':
                    # 혼복이 적을수록 우선순위 높음
                    score += honbok_count * 5 - nambok_count * 2
                
                return score
            
            available.sort(key=get_priority_score)
        else:
            # 여자 선수는 참여 횟수로만 정렬
            available.sort(key=lambda x: participation_count[x])
    else:
        # 참여 횟수로 정렬
        available.sort(key=lambda x: participation_count[x])
    
    return available

# 이상적인 참여 횟수 계산
total_player_slots = TOTAL_MATCHES * 4  # 15경기 * 4명
total_players = len(players)
ideal_participation = total_player_slots / total_players
max_participation = int(ideal_participation) + 1  # 천장값 + 1

print(f"총 선수 슬롯: {total_player_slots}, 총 선수: {total_players}")
print(f"이상적 참여 횟수: {ideal_participation:.2f}, 최대 허용: {max_participation}\n")

for time in range(1, NUM_TIMES + 1):
    # 각 타임 시작 시 해당 타임의 참여자 집합 초기화
    players_in_time_slot[time] = set()
    
    for court in range(1, NUM_COURTS + 1):
        # 현재 최대/최소 참여 횟수 확인
        max_current = max(participation_count.values())
        min_current = min(participation_count.values())
        
        # 코트별 선호 경기 타입 설정
        # 1코트: 남복 우선, 3코트: 여복 우선, 2코트: 중립
        preferred_type = None
        if court == 1:
            preferred_type = '남복'
        elif court == 3:
            preferred_type = '여복'
        
        # 가능한 매치 타입 결정 (현재 타임에 참여하지 않은 선수 중에서)
        possible_matches = []
        
        # 남복 가능 여부
        available_men = get_balanced_players(men, 4, participation_count, max_participation, time, '남복')
        if len(available_men) >= 4:
            possible_matches.append('남복')
        
        # 여복 가능 여부
        available_women = get_balanced_players(women, 4, participation_count, max_participation, time)
        if len(available_women) >= 4:
            possible_matches.append('여복')
        
        # 혼복 가능 여부
        available_men_honbok = get_balanced_players(men, 2, participation_count, max_participation, time, '혼복')
        available_women_honbok = get_balanced_players(women, 2, participation_count, max_participation, time)
        if len(available_men_honbok) >= 2 and len(available_women_honbok) >= 2:
            possible_matches.append('혼복')
        
        if not possible_matches:
            # 최대 참여 횟수를 늘려서 재시도
            max_participation += 1
            print(f"⚠️ 타임{time} 코트{court}: 참여 가능한 선수 부족. 최대 허용 횟수를 {max_participation}로 증가\n")
            
            # 다시 확인
            available_men = get_balanced_players(men, 4, participation_count, max_participation, time, '남복')
            if len(available_men) >= 4:
                possible_matches.append('남복')
            
            available_women = get_balanced_players(women, 4, participation_count, max_participation, time)
            if len(available_women) >= 4:
                possible_matches.append('여복')
            
            available_men_honbok = get_balanced_players(men, 2, participation_count, max_participation, time, '혼복')
            available_women_honbok = get_balanced_players(women, 2, participation_count, max_participation, time)
            if len(available_men_honbok) >= 2 and len(available_women_honbok) >= 2:
                possible_matches.append('혼복')
            
            if not possible_matches:
                print(f"❌ 타임{time} 코트{court}: 매치 생성 불가능\n")
                continue
        
        # 매치 타입 선택 로직
        match_type = None
        
        # 1순위: 여복을 2타임과 4타임에 고정 배정
        if time in [2, 4] and match_type_count['여복'] < EXACT_YEOBOK and '여복' in possible_matches:
            match_type = '여복'
        # 여복이 아닌 타임(1,3,5)에서는 여복 배정 안함
        elif time not in [2, 4] and '여복' in possible_matches:
            possible_matches.remove('여복')
        
        # 2순위: 남복 최소 경기 수
        if match_type is None and match_type_count['남복'] < MIN_NAMBOK and '남복' in possible_matches:
            match_type = '남복'
        # 3순위: 혼복 최소 경기 수
        elif match_type is None and match_type_count['혼복'] < MIN_HONBOK and '혼복' in possible_matches:
            match_type = '혼복'
        
        # 4순위: 코트별 선호 타입 (1코트=남복, 3코트=여복) - 단, 여복은 이미 2경기 완료된 경우만
        if match_type is None and preferred_type and preferred_type in possible_matches:
            # 여복 선호이지만 이미 2경기를 채웠다면 선택 안함
            if preferred_type == '여복' and match_type_count['여복'] >= EXACT_YEOBOK:
                pass  # 여복은 건너뜀
            else:
                match_type = preferred_type
        
        # 5순위: 참여 횟수가 가장 적은 그룹 우선 (여복 제외)
        if match_type is None:
            # 각 타입별로 참여 가능한 선수들의 평균 참여 횟수 계산
            type_scores = {}
            
            if '남복' in possible_matches:
                available = get_balanced_players(men, 4, participation_count, max_participation, time, '남복')[:4]
                if len(available) >= 4:
                    # 남자 선수들의 남복/혼복 밸런스 고려
                    nambok_count = sum(men_match_type_count[p]['남복'] for p in available)
                    honbok_count = sum(men_match_type_count[p]['혼복'] for p in available)
                    # 남복 참여가 적을수록 점수가 낮음 (우선순위 높음)
                    type_scores['남복'] = sum(participation_count[p] for p in available) / 4 + (nambok_count - honbok_count) * 0.5
            
            # 여복은 이미 2경기를 채운 경우 제외
            if '여복' in possible_matches and match_type_count['여복'] < EXACT_YEOBOK:
                available = get_balanced_players(women, 4, participation_count, max_participation, time)[:4]
                if len(available) >= 4:
                    type_scores['여복'] = sum(participation_count[p] for p in available) / 4
            
            if '혼복' in possible_matches:
                available_m = get_balanced_players(men, 2, participation_count, max_participation, time, '혼복')[:2]
                available_w = get_balanced_players(women, 2, participation_count, max_participation, time)[:2]
                if len(available_m) >= 2 and len(available_w) >= 2:
                    # 남자 선수들의 남복/혼복 밸런스 고려
                    nambok_count = sum(men_match_type_count[p]['남복'] for p in available_m)
                    honbok_count = sum(men_match_type_count[p]['혼복'] for p in available_m)
                    # 혼복 참여가 적을수록 점수가 낮음 (우선순위 높음)
                    type_scores['혼복'] = (sum(participation_count[p] for p in available_m) + 
                                          sum(participation_count[p] for p in available_w)) / 4 + (honbok_count - nambok_count) * 0.5
            
            if type_scores:
                match_type = min(type_scores, key=type_scores.get)
        
        if match_type == '남복':
            # 남복: 남자 4명 - 실력차를 고려한 팀 구성
            available = get_balanced_players(men, 8, participation_count, max_participation, time, '남복')
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화
            current_max = max(participation_count[p] for p in available[:4]) if len(available) >= 4 else 0
            current_min = min(participation_count[p] for p in available[:4]) if len(available) >= 4 else 0
            participation_gap = current_max - current_min
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화 (참여 횟수 보장이 최우선)
            if participation_gap > 0:
                skill_diff_limit = 1.0  # 제약 완화
            else:
                skill_diff_limit = 0.5  # 기본 제약
            
            # 최대 8명 중에서 최적의 4명 조합 찾기
            best_match = None
            for candidate_group in combinations(available[:min(8, len(available))], 4):
                balanced_teams = find_balanced_teams(list(candidate_group), players, participation_count, max_skill_diff=skill_diff_limit)
                if balanced_teams:
                    best_match = (list(candidate_group), balanced_teams)
                    break
            
            if best_match:
                selected, (team1, team2, skill_diff) = best_match
                match = create_match(team1[0], team1[1], team2[0], team2[1], '남복', court, time)
                matches.append(match)
                match_type_count['남복'] += 1
                # 팀 이력 기록
                add_team_history(team1[0], team1[1])
                add_team_history(team2[0], team2[1])
                # 적 이력 기록
                add_opponent_history(team1, team2)
                for player in selected:
                    participation_count[player] += 1
                    players_in_time_slot[time].add(player)
                    men_match_type_count[player]['남복'] += 1  # 남복 참여 기록
            else:
                # 밸런스를 맞출 수 없으면 그냥 순서대로 (참여 횟수 보장이 최우선)
                selected = available[:4]
                if len(selected) >= 4:
                    team1 = [selected[0], selected[1]]
                    team2 = [selected[2], selected[3]]
                    match = create_match(team1[0], team1[1], team2[0], team2[1], '남복', court, time)
                    matches.append(match)
                    match_type_count['남복'] += 1
                    # 팀 이력 기록
                    add_team_history(team1[0], team1[1])
                    add_team_history(team2[0], team2[1])
                    # 적 이력 기록
                    add_opponent_history(team1, team2)
                    for player in selected:
                        participation_count[player] += 1
                        players_in_time_slot[time].add(player)
                        men_match_type_count[player]['남복'] += 1  # 남복 참여 기록
                    
        elif match_type == '여복':
            # 여복: 여자 4명 - 실력차를 고려한 팀 구성
            available = get_balanced_players(women, 8, participation_count, max_participation, time)
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화
            current_max = max(participation_count[p] for p in available[:4]) if len(available) >= 4 else 0
            current_min = min(participation_count[p] for p in available[:4]) if len(available) >= 4 else 0
            participation_gap = current_max - current_min
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화 (참여 횟수 보장이 최우선)
            if participation_gap > 0:
                skill_diff_limit = 1.0  # 제약 완화
            else:
                skill_diff_limit = 0.5  # 기본 제약
            
            # 최대 8명 중에서 최적의 4명 조합 찾기
            best_match = None
            for candidate_group in combinations(available[:min(8, len(available))], 4):
                balanced_teams = find_balanced_teams(list(candidate_group), players, participation_count, max_skill_diff=skill_diff_limit)
                if balanced_teams:
                    best_match = (list(candidate_group), balanced_teams)
                    break
            
            if best_match:
                selected, (team1, team2, skill_diff) = best_match
                match = create_match(team1[0], team1[1], team2[0], team2[1], '여복', court, time)
                matches.append(match)
                match_type_count['여복'] += 1
                # 팀 이력 기록
                add_team_history(team1[0], team1[1])
                add_team_history(team2[0], team2[1])
                # 적 이력 기록
                add_opponent_history(team1, team2)
                for player in selected:
                    participation_count[player] += 1
                    players_in_time_slot[time].add(player)
            else:
                # 밸런스를 맞출 수 없으면 그냥 순서대로 (참여 횟수 보장이 최우선)
                selected = available[:4]
                if len(selected) >= 4:
                    team1 = [selected[0], selected[1]]
                    team2 = [selected[2], selected[3]]
                    match = create_match(team1[0], team1[1], team2[0], team2[1], '여복', court, time)
                    matches.append(match)
                    match_type_count['여복'] += 1
                    # 팀 이력 기록
                    add_team_history(team1[0], team1[1])
                    add_team_history(team2[0], team2[1])
                    # 적 이력 기록
                    add_opponent_history(team1, team2)
                    for player in selected:
                        participation_count[player] += 1
                        players_in_time_slot[time].add(player)
                    
        elif match_type == '혼복':
            # 혼복: 남자 2명, 여자 2명 - 실력차를 고려한 팀 구성
            available_men = get_balanced_players(men, 4, participation_count, max_participation, time, '혼복')
            available_women = get_balanced_players(women, 4, participation_count, max_participation, time)
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화
            all_available = (available_men[:2] if len(available_men) >= 2 else []) + \
                          (available_women[:2] if len(available_women) >= 2 else [])
            if len(all_available) >= 4:
                current_max = max(participation_count[p] for p in all_available)
                current_min = min(participation_count[p] for p in all_available)
                participation_gap = current_max - current_min
            else:
                participation_gap = 0
            
            # 참여 횟수 차이가 크면 실력차 제약을 완화 (참여 횟수 보장이 최우선)
            if participation_gap > 0:
                skill_diff_limit = 1.0  # 제약 완화
            else:
                skill_diff_limit = 0.5  # 기본 제약
            
            # 최적의 조합 찾기
            best_match = None
            for men_combo in combinations(available_men[:min(4, len(available_men))], 2):
                for women_combo in combinations(available_women[:min(4, len(available_women))], 2):
                    balanced_teams = find_balanced_mixed_teams(list(men_combo), list(women_combo), players, participation_count, max_skill_diff=skill_diff_limit)
                    if balanced_teams:
                        best_match = (list(men_combo), list(women_combo), balanced_teams)
                        break
                if best_match:
                    break
            
            if best_match:
                selected_men, selected_women, (team1, team2, skill_diff) = best_match
                match = create_match(team1[0], team1[1], team2[0], team2[1], '혼복', court, time)
                matches.append(match)
                match_type_count['혼복'] += 1
                # 팀 이력 기록
                add_team_history(team1[0], team1[1])
                add_team_history(team2[0], team2[1])
                # 적 이력 기록
                add_opponent_history(team1, team2)
                for player in selected_men + selected_women:
                    participation_count[player] += 1
                    players_in_time_slot[time].add(player)
                # 남자 선수들의 혼복 참여 기록
                for player in selected_men:
                    men_match_type_count[player]['혼복'] += 1
            else:
                # 밸런스를 맞출 수 없으면 그냥 순서대로 (참여 횟수 보장이 최우선)
                if len(available_men) >= 2 and len(available_women) >= 2:
                    selected_men = available_men[:2]
                    selected_women = available_women[:2]
                    team1 = [selected_men[0], selected_women[0]]
                    team2 = [selected_men[1], selected_women[1]]
                    match = create_match(team1[0], team1[1], team2[0], team2[1], '혼복', court, time)
                    matches.append(match)
                    match_type_count['혼복'] += 1
                    # 팀 이력 기록
                    add_team_history(team1[0], team1[1])
                    add_team_history(team2[0], team2[1])
                    # 적 이력 기록
                    add_opponent_history(team1, team2)
                    for player in selected_men + selected_women:
                        participation_count[player] += 1
                        players_in_time_slot[time].add(player)
                    # 남자 선수들의 혼복 참여 기록
                    for player in selected_men:
                        men_match_type_count[player]['혼복'] += 1

print(f"총 {len(matches)}개 매치 생성 완료!\n")

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
    print("\n⚠️ PDF 생성을 위해 reportlab 라이브러리가 필요합니다.")
    print("   다음 명령어로 설치하세요: pip install reportlab")
except Exception as e:
    print(f"\n❌ PDF 생성 중 오류 발생: {e}")

