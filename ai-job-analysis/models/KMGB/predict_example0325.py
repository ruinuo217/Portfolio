"""
=============================================================
  predict_example.py  ← 交給後端成員 D
  職缺推薦 Top 3 + 技能缺口分析
  流程：技能字串 → encoder → 技能分群 → 分類器 → 職類
=============================================================
"""

import json
import joblib
import numpy as np

# ─────────────────────────────────────────────
# 載入模型
# ─────────────────────────────────────────────
mlb = joblib.load("models/encoder.pkl")
kmeans_skill = joblib.load("models/KMGB/skill_clusters.pkl")
clf = joblib.load("models/KMGB/job_classifier.pkl")
le = joblib.load("models/KMGB/label_encoder.pkl")

with open("models/cluster_skills_mapping.json", "r", encoding="utf-8") as f:
    cluster_skills_mapping = json.load(f)

print("✅ 模型載入成功！\n")


# ─────────────────────────────────────────────
# 預測函式：回傳 Top N 職缺推薦
# ─────────────────────────────────────────────
def predict(user_skills_list, top_n=3):
    """
    參數：
      user_skills_list: 二維陣列，例如 [["Python", "SQL", "Docker"]]
      top_n: 回傳前幾名推薦，預設 3

    回傳：
      list of dict，包含職缺名稱、信心度、技能缺口等資訊
    """
    user_skills = user_skills_list[0]

    # 第一關：技能字串 → 0/1 矩陣
    X = mlb.transform(user_skills_list)

    # 第二關：0/1 矩陣 → 技能群距離向量
    skill_features = kmeans_skill.transform(X)

    # 第三關：技能群距離向量 → 職類機率
    probas = clf.predict_proba(skill_features)[0]
    top_indices = probas.argsort()[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        job_title = le.classes_[idx]
        probability = probas[idx]
        required_skills = cluster_skills_mapping.get(job_title, [])
        user_skill_set = set(user_skills)
        required_set = set(required_skills)
        matched = list(required_set & user_skill_set)
        skill_gap = list(required_set - user_skill_set)
        match_pct = round(len(matched) / len(required_skills)
                          * 100, 1) if required_skills else 0

        results.append({
            "rank":                rank,
            "predicted_job_title": job_title,
            "probability":         round(float(probability) * 100, 1),
            "matched_skills":      matched,
            "skill_gap":           skill_gap[:5],
            "match_percentage":    match_pct
        })

    return results


# ─────────────────────────────────────────────
# 列印結果（方便測試用）
# ─────────────────────────────────────────────
def print_results(user_skills, results):
    print(f"輸入技能：{user_skills}")
    print("=" * 50)
    for r in results:
        print(
            f"  第 {r['rank']} 名：{r['predicted_job_title']}（信心度 {r['probability']}%）")
        print(f"    ✅ 已具備：{r['matched_skills']}")
        print(f"    ❌ 技能缺口：{r['skill_gap']}")
        print(f"    📊 符合度：{r['match_percentage']}%")
        print()


# ─────────────────────────────────────────────
# 測試區（改這裡就好）
# ─────────────────────────────────────────────
test_cases = [
    ["JavaScript", "React", "CSS", "TypeScript"],
    ["Python", "PyTorch", "TensorFlow", "Git"],
    ["AutoCAD", "SolidWorks", "Excel"],
    ["Docker", "Kubernetes", "AWS", "CI/CD"],
    ["Python", "SQL", "Docker"],
]

for skills in test_cases:
    results = predict([skills], top_n=3)
    print_results(skills, results)
    print("-" * 50)


# ─────────────────────────────────────────────
# 後端 API 回傳格式範例（給成員 D 參考）
# ─────────────────────────────────────────────
print("\n📦 後端回傳前端的 JSON 格式範例：")
print(json.dumps(predict([["Python", "SQL", "Docker"]],
      top_n=3), ensure_ascii=False, indent=2))
