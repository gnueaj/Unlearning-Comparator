import requests
import json
import pandas as pd
from datetime import datetime

def get_webtoon_comments(webtoon_id, episode_id, max_pages=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Referer': 'https://comic.naver.com',
        'Host': 'apis.naver.com'
    }
    
    all_comments = []
    
    for page in range(1, max_pages + 1):
        url = f'https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json?ticket=comic&templateId=webtoon&pool=cbox3&_callback=jQuery112403824265166960604_1633940001123&lang=ko&country=KR&objectId={webtoon_id}_{episode_id}&pageSize=15&indexSize=10&listType=OBJECT&pageType=default&page={page}&initialize=true&useAltSort=true&replyPageSize=30&sort=new'

        response = requests.get(url, headers=headers)
        #print(page)

        if response.status_code != 200:
            print(f"요청 실패: {response.status_code}")
            print(response.text)
            break

        json_string = response.text[response.text.find('(') + 1:-2]

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            print("JSON 디코딩 에러:", e)
            print("응답 내용:", response.text)
            break

        if 'result' not in data or 'commentList' not in data['result']:
            print("댓글 리스트가 응답에 없습니다.")
            break
        
        for comment in data['result']['commentList']:
            # 날짜 변환
            raw_date = comment.get('regTime', 'N/A')
            try:
                # 변환할 수 없는 경우 기본값 'N/A' 사용
                date = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%S+0900").strftime("%Y%m%d") if raw_date != 'N/A' else 'N/A'
            except ValueError:
                date = 'N/A'
            
            main_comment = {
                '작성자 id': comment.get('userName', 'N/A'),
                '날짜': date,
                '댓글': comment.get('contents', 'N/A'),
                '좋아요 수': comment.get('sympathyCount', 0),
                '싫어요 수': comment.get('antipathyCount', 0),
                '대댓글': []
            }
            
            # 각 댓글의 대댓글 가져오기
            if comment.get('commentNo'):
                reply_url = f'https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json?ticket=comic&templateId=webtoon&pool=cbox3&_callback=jQuery112403824265166960604_1633940001123&lang=ko&country=KR&objectId={webtoon_id}_{episode_id}&parentCommentNo={comment["commentNo"]}&pageSize=10&indexSize=10&listType=OBJECT&pageType=default&page=1&sort=favorite'
                
                reply_response = requests.get(reply_url, headers=headers)
                
                if reply_response.status_code == 200:
                    reply_json_string = reply_response.text[reply_response.text.find('(') + 1:-2]
                    
                    try:
                        reply_data = json.loads(reply_json_string)
                        
                        if 'result' in reply_data and 'commentList' in reply_data['result']:
                            for reply in reply_data['result']['commentList']:
                                raw_reply_date = reply.get('regTime', 'N/A')
                                try:
                                    reply_date = datetime.strptime(raw_reply_date, "%Y-%m-%dT%H:%M:%S+0900").strftime("%Y%m%d") if raw_reply_date != 'N/A' else 'N/A'
                                except ValueError:
                                    reply_date = 'N/A'
                                
                                main_comment['대댓글'].append({
                                    '작성자 id': reply.get('userName', 'N/A'),
                                    '날짜': reply_date,
                                    '댓글': reply.get('contents', 'N/A'),
                                    '좋아요 수': reply.get('sympathyCount', 0),
                                    '싫어요 수': reply.get('antipathyCount', 0)
                                })
                    except json.JSONDecodeError as e:
                        print("대댓글 JSON 디코딩 에러:", e)
                        print("응답 내용:", reply_response.text)

            all_comments.append(main_comment)
            #print(main_comment)

    return all_comments


def save_comments_to_excel(comments, filename='webtoon_comments_crawl.xlsx'):
    rows = []
    for comment in comments:
        # 메인 댓글이 좋아요나 싫어요가 1 이상인 경우만 추가
        if comment['좋아요 수'] >= 1 or comment['싫어요 수'] >= 1:
            rows.append({
                '회차': int(episode_id),
                '작성자 id': comment['작성자 id'],
                '날짜': comment['날짜'],
                '좋아요 수': comment['좋아요 수'],
                '싫어요 수': comment['싫어요 수'],
                '대댓글 여부': 0,
                '댓글 내용': comment['댓글'],
                'label': '',  # 빈 label 컬럼 추가
                'about': ''   # 빈 about 컬럼 추가
            })
        
        # 대댓글이 좋아요나 싫어요가 1 이상인 경우만 추가
        for reply in comment['대댓글']:
            if reply['좋아요 수'] >= 1 or reply['싫어요 수'] >= 1:
                rows.append({
                    '회차': int(episode_id),
                    '작성자 id': reply['작성자 id'],
                    '날짜': reply['날짜'],
                    '좋아요 수': reply['좋아요 수'],
                    '싫어요 수': reply['싫어요 수'],
                    '대댓글 여부': 1,
                    '댓글 내용': reply['댓글'],
                    'label': '',  # 빈 label 컬럼 추가
                    'about': ''   # 빈 about 컬럼 추가
                })
    
    # DataFrame으로 변환 후 엑셀 파일로 저장
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"{filename} 파일에 저장되었습니다.")

# 사용 예시
webtoon_id = '737628'
episode_id = '9'
comments = get_webtoon_comments(webtoon_id, episode_id, max_pages=50)
save_comments_to_excel(comments, 'webtoon_comments_crawl_'+episode_id+'.xlsx')
