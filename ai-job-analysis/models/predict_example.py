import joblib

# 載入模型
mlb = joblib.load("models/encoder.pkl")
clf = joblib.load("models/job_clusters.pkl")
le = joblib.load("models/skill_clusters.pkl")


def predict(user_skills):
    # 1️⃣ 技能 → 向量
    X = mlb.transform(user_skills)

    # 2️⃣ 預測（數字 label）
    pred = clf.predict(X)

    # 3️⃣ 轉回職類名稱
    job_name = le.inverse_transform(pred)

    return job_name


# =========================
# 測試區
# =========================

# 測試 1：前端
# user_skills = [["JavaScript", "React", "CSS", "TypeScript"]]

# 測試 2：AI
# user_skills = [["Python", "PyTorch", "TensorFlow", "Git"]]

# 測試 3：製造
user_skills = [["AutoCAD", "SolidWorks", "Excel"]]

# 測試 4：DevOps
# user_skills = [["Docker", "Kubernetes", "AWS", "CI/CD"]]

# user_skills = [["Excel", "AutoCAD", "Docker"]]


result = predict(user_skills)

print("輸入技能：", user_skills)
print("預測職類：", result[0])
