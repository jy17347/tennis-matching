import pandas as pd
import random
from itertools import combinations
from datetime import datetime
import sys, os


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
players = players[(players['비고'] == 'O') | (players['비고'] == '1')].copy()

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
    """
    best_match = None
    best_diff = float('inf')
    
    for team1_combo in combinations(four_players, 2):
        team1 = list(team1_combo)
        team2 = [p for p in four_players if p not in team1]
        
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

def find_balanced_mixed_teams_simple(two_men, two_women):
    """
    혼복 매칭 - 제약조건 6 적용:
    - 남자끼리 실력차 ≤1
    - 여자끼리 실력차 ≤1
    - 최상위 남자 + 최상위 여자 분리
    """
    man1, man2 = two_men
    woman1, woman2 = two_women
    
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
                print(f"  ⚠ 타임{time} 코트{court} {match_type}: 선수 부족")
                continue
            
            selected = available[:4]
            result = find_balanced_teams_simple(selected)
            
            if result:
                team1, team2, _ = result
            else:
                # 제약 완화: 그냥 2:2로 나눔
                team1 = selected[:2]
                team2 = selected[2:4]
            
            match = create_match(team1, team2, match_type, court, time)
            matches.append(match)
            
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
