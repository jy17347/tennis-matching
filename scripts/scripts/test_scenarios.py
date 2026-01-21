"""다양한 참가자 조합에 대한 매칭 시스템 테스트"""

import sys
sys.path.insert(0, r'c:\project\matching')
from tennis_matching import TennisMatchingSystem
import pandas as pd
import tempfile
import os


def test_scenario(name, male_count, female_count):
    """시나리오 테스트"""
    print(f"\n{'='*60}")
    print(f"테스트: {name} (남자 {male_count}명, 여자 {female_count}명)")
    print(f"{'='*60}")
    
    # 로스터 데이터 생성
    roster_data = []
    for i in range(male_count):
        roster_data.append({'번호': i+1, '성별': 1, '성명': f'남{i+1}', '실력': (i % 5) + 1})
    for i in range(female_count):
        roster_data.append({'번호': male_count+i+1, '성별': 2, '성명': f'여{i+1}', '실력': (i % 5) + 1})
    
    df_roster = pd.DataFrame(roster_data)
    df_part = pd.DataFrame({'성명': [r['성명'] for r in roster_data], '비고': ['O'] * len(roster_data)})
    
    # 임시 파일로 저장
    with tempfile.TemporaryDirectory() as tmpdir:
        roster_path = os.path.join(tmpdir, 'roster.xlsx')
        part_path = os.path.join(tmpdir, 'participation.xlsx')
        df_roster.to_excel(roster_path, index=False)
        df_part.to_excel(part_path, index=False)
        
        try:
            system = TennisMatchingSystem(roster_path, part_path)
            system.validate_configuration()
            male_m, female_m, mixed_m = system.calculate_match_distribution()
            
            total = male_m + female_m + mixed_m
            male_slots = male_m * 4 + mixed_m * 2
            female_slots = female_m * 4 + mixed_m * 2
            
            print(f"\n✅ 검증 통과!")
            print(f"   총 {total}경기: 남복 {male_m}, 여복 {female_m}, 혼복 {mixed_m}")
            print(f"   남자 슬롯: {male_slots}/{male_count * 5} ({male_slots/(male_count*5)*100:.0f}%)")
            print(f"   여자 슬롯: {female_slots}/{female_count * 5} ({female_slots/(female_count*5)*100:.0f}%)")
            
            return True
            
        except ValueError as e:
            print(f"\n❌ 실패: {e}")
            return False
        except Exception as e:
            print(f"\n❌ 에러: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    # 다양한 시나리오 테스트
    scenarios = [
        ("현재 상황", 16, 8),
        ("남자 감소", 12, 8),
        ("여자 감소", 16, 4),
        ("전체 감소", 10, 6),
        ("전체 증가", 20, 10),
        ("남녀 동수", 8, 8),
        ("중간 케이스", 14, 6),
        ("최소 가능 인원", 6, 4),
    ]

    results = {}
    for name, m, f in scenarios:
        results[name] = test_scenario(name, m, f)

    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    for name, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"  {name}: {status}")
