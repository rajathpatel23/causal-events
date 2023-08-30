import json
JSON1 = "/home/jovyan/work/causal-events/src/report/classification/causal-news-256-32-2e-5-3-2023-07-06-True-roberta-base/predict_results_causal_news.json"
JSON2 = "/home/jovyan/work/causal-events/src/report/classification/date-08-25-2023-opps/causal-news-128-8-2e-5-10-2023-08-25-8-only-train-FRESH-1.0-eval_recall-roberta-base/predict_results_causal_news.json"
def get_json_data(JSON):
    with open(JSON, "r") as json1:
        data = json1.readlines()
        data = [json.loads(res) for res in data]
    return data
count_match_1 = 0
data1 = get_json_data(JSON1)
data2 = get_json_data(JSON2)
for index, dict_data in enumerate(data1):
    if dict_data['prediction'] == 1.0:
        if data2[index]['prediction'] == 1.0:
            count_match_1 +=1

print(count_match_1)