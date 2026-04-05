from flask import Flask, jsonify, render_template, request
from flasgger import Swagger
import os
from dotenv import load_dotenv
import pymysql
import numpy as np
import joblib
import json
import schedule
import threading
import time

load_dotenv()

try:
    connection = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        ssl={"ssl": {}},
        cursorclass=pymysql.cursors.DictCursor,
    )
    print("Connected to primary DB.")
except pymysql.MySQLError as e:
    print(f"Failed to connect to primary DB, trying backup DB: {e}")
    connection = pymysql.connect(
        host=os.getenv("GCP_DB_HOST"),
        port=int(os.getenv("GCP_DB_PORT", 3306)) if os.getenv("GCP_DB_PORT") else 3306,
        user=os.getenv("GCP_DB_USER"),
        password=os.getenv("GCP_DB_PASS"),
        database=os.getenv("GCP_DB_NAME"),
        charset="utf8mb4",
        ssl={"ssl": {}},
        cursorclass=pymysql.cursors.DictCursor,
    )
    print("Connected to backup DB.")


def backup_database():
    print("Starting daily backup task from primary DB to backup DB...")

    primary_conn = None
    backup_conn = None

    try:
        primary_conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME"),
            charset="utf8mb4",
            ssl={"ssl": {}},
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        print(f"Cannot connect to primary DB for backup: {e}")
        return

    try:
        backup_conn = pymysql.connect(
            host=os.getenv("GCP_DB_HOST"),
            port=(
                int(os.getenv("GCP_DB_PORT", 3306))
                if os.getenv("GCP_DB_PORT")
                else 3306
            ),
            user=os.getenv("GCP_DB_USER"),
            password=os.getenv("GCP_DB_PASS"),
            database=os.getenv("GCP_DB_NAME"),
            charset="utf8mb4",
            ssl={"ssl": {}},
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        print(f"Cannot connect to backup DB for backup: {e}")
        if primary_conn:
            primary_conn.close()
        return

    try:
        with primary_conn.cursor() as p_cur, backup_conn.cursor() as b_cur:
            p_cur.execute("SHOW TABLES")
            tables = [list(t.values())[0] for t in p_cur.fetchall()]

            for table in tables:
                p_cur.execute(f"CHECKSUM TABLE `{table}`")
                p_checksum_row = p_cur.fetchone()
                p_checksum = p_checksum_row["Checksum"] if p_checksum_row else None

                try:
                    b_cur.execute(f"CHECKSUM TABLE `{table}`")
                    b_checksum_row = b_cur.fetchone()
                    b_checksum = b_checksum_row["Checksum"] if b_checksum_row else None
                except Exception as b_err:
                    print(
                        f"Warning: Failed to checksum table `{table}` in backup DB: {b_err}"
                    )
                    b_checksum = None

                if p_checksum != b_checksum or p_checksum is None:
                    print(f"Changes detected in table `{table}`, backing up...")
                    p_cur.execute(f"SELECT * FROM `{table}`")
                    rows = p_cur.fetchall()

                    try:
                        b_cur.execute("SET FOREIGN_KEY_CHECKS = 0;")
                        b_cur.execute(f"TRUNCATE TABLE `{table}`")

                        if rows:
                            cols = rows[0].keys()
                            col_str = ", ".join([f"`{c}`" for c in cols])
                            val_str = ", ".join(["%s"] * len(cols))
                            insert_query = (
                                f"INSERT INTO `{table}` ({col_str}) VALUES ({val_str})"
                            )
                            data = [[row[c] for c in cols] for row in rows]
                            b_cur.executemany(insert_query, data)
                        backup_conn.commit()
                        b_cur.execute("SET FOREIGN_KEY_CHECKS = 1;")
                    except Exception as in_e:
                        print(f"Error backing up table `{table}`: {in_e}")
                        backup_conn.rollback()
                        try:
                            b_cur.execute("SET FOREIGN_KEY_CHECKS = 1;")
                        except:
                            pass
                else:
                    print(f"No changes in table `{table}`, skipping backup.")
    except Exception as e:
        print(f"Backup task failed during syncing: {e}")
    finally:
        if backup_conn and backup_conn.open:
            backup_conn.close()
        if primary_conn and primary_conn.open:
            primary_conn.close()


def run_schedule():
    schedule.every().day.at("03:00").do(backup_database)
    while True:
        schedule.run_pending()
        time.sleep(60)


threading.Thread(target=run_schedule, daemon=True).start()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 解決中文亂碼問題
swagger = Swagger(app)  # 初始化 Swagger API 文件

# 載入模型與技能對應表
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")

try:
    mlb = joblib.load(os.path.join(models_dir, "encoder.pkl"))
    kmeans_skill = joblib.load(os.path.join(models_dir, "KMGB", "skill_clusters.pkl"))
    clf = joblib.load(os.path.join(models_dir, "KMGB", "job_classifier.pkl"))
    le = joblib.load(os.path.join(models_dir, "KMGB", "label_encoder.pkl"))
    with open(
        os.path.join(models_dir, "cluster_skills_mapping.json"), "r", encoding="utf-8"
    ) as f:
        cluster_skills_mapping = json.load(f)
    salary_model = joblib.load(os.path.join(models_dir, "salary_prediction_xgb.pkl"))
    title_cols = joblib.load(os.path.join(models_dir, "title_columns.pkl"))
except Exception as e:
    print(f"Warning: Models could not be loaded: {e}")
    mlb = clf = le = kmeans_skill = cluster_skills_mapping = salary_model = (
        title_cols
    ) = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/job/top-skills", methods=["GET"])
def get_top_skills():
    """
    市場熱門觀測站 API (取得主流技能、含金量與熱門職缺)
    ---
    tags:
      - AI 職缺分析
    responses:
      200:
        description: 成功回傳 Top 5 需求技能、高薪技能與熱門職稱
    """

    connection.ping(reconnect=True)
    with connection.cursor() as cursor:
        # 需求量最高技能 Top 5 (市場主流)
        cursor.execute(
            """
            SELECT s.skill_name, COUNT(m.job_id) AS demand_count
            FROM skills s
            JOIN job_skills_mapping m ON s.skill_id = m.skill_id
            JOIN jobs j ON m.job_id = j.job_id
            WHERE j.job_title IN (
                'AI工程師',
                'APP工程師',
                'MIS/IT工程師',
                'QA/品管工程師',
                '全端工程師',
                '前端工程師',
                '後端工程師',
                '營建/水電工程',
                '硬體工程師',
                '系統工程師',
                '製造/設備工程',
                '資料工程師',
                '雲端/DevOps工程師',
                '電子/電機工程',
                '韌體工程師'
            )
            GROUP BY s.skill_id, s.skill_name
            ORDER BY demand_count DESC
            LIMIT 5;
        """
        )
        top_demand_skills = cursor.fetchall()

        # 含金量最高技能 Top 5 (談薪武器)
        cursor.execute(
            """
            SELECT s.skill_name, ROUND(AVG((j.min_salary + j.max_salary) / 2)) AS avg_salary
            FROM skills s
            JOIN job_skills_mapping m ON s.skill_id = m.skill_id
            JOIN jobs j ON m.job_id = j.job_id
            WHERE j.min_salary IS NOT NULL AND j.max_salary IS NOT NULL
            AND j.job_title IN (
                'AI工程師',
                'APP工程師',
                'MIS/IT工程師',
                'QA/品管工程師',
                '全端工程師',
                '前端工程師',
                '後端工程師',
                '營建/水電工程',
                '硬體工程師',
                '系統工程師',
                '製造/設備工程',
                '資料工程師',
                '雲端/DevOps工程師',
                '電子/電機工程',
                '韌體工程師'
            )
            AND j.is_negotiable = 0
            GROUP BY s.skill_id, s.skill_name
            HAVING COUNT(j.job_id) >= 10
            ORDER BY avg_salary DESC
            LIMIT 5;
        """
        )
        top_salary_skills = cursor.fetchall()

        # 開缺最多的熱門職稱 Top 5
        cursor.execute(
            """
            SELECT job_title, COUNT(job_id) AS opening_count
            FROM jobs
            WHERE job_title IS NOT NULL
            AND job_title IN (
                'AI工程師',
                'APP工程師',
                'MIS/IT工程師',
                'QA/品管工程師',
                '全端工程師',
                '前端工程師',
                '後端工程師',
                '營建/水電工程',
                '硬體工程師',
                '系統工程師',
                '製造/設備工程',
                '資料工程師',
                '雲端/DevOps工程師',
                '電子/電機工程',
                '韌體工程師'
            )
            AND is_negotiable = 0
            GROUP BY job_title
            ORDER BY opening_count DESC
            LIMIT 5;
        """
        )
        top_job_titles = cursor.fetchall()

    return jsonify(
        {
            "top_demand_skills": top_demand_skills,
            "top_salary_skills": top_salary_skills,
            "top_job_titles": top_job_titles,
        }
    )


@app.route("/job/recommend", methods=["POST"])
def recommend_job():
    """
    職缺推薦 API
    ---
    tags:
      - AI 職缺分析
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            skills:
              type: array
              items:
                type: string
              example: ["Python", "SQL", "Docker"]
            target_job_title:
              type: string
              example: "後端工程師"
    responses:
      200:
        description: 推薦結果
    """
    data = request.get_json(silent=True)
    if not data or "skills" not in data:
        return jsonify({"error": "Missing 'skills' in request body"}), 400

    user_skills = data["skills"]
    if not isinstance(user_skills, list):
        return jsonify({"error": "'skills' must be a list"}), 400

    target_job_title = data.get("target_job_title")
    user_skills_lower = set([str(s).lower() for s in user_skills])

    if target_job_title:
        # Target mode
        required_skills = (
            cluster_skills_mapping.get(target_job_title, [])
            if cluster_skills_mapping
            else []
        )

        user_has = [s for s in required_skills if s.lower() in user_skills_lower]
        skill_gap = [s for s in required_skills if s.lower() not in user_skills_lower]

        return jsonify(
            {
                "mode": "target",
                "target_job_title": target_job_title,
                "required_skills": required_skills,
                "user_has": user_has,
                "skill_gap": skill_gap,
            }
        )

    else:
        # Explore mode
        if not mlb or not clf or not le or not kmeans_skill:
            return jsonify({"error": "Models are not loaded."}), 500

        try:
            X = mlb.transform([user_skills])
            skill_features = kmeans_skill.transform(X)
            probs = clf.predict_proba(skill_features)[0]
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        n = 3
        top_n_indices = np.argsort(probs)[::-1][:n]
        top_names = le.inverse_transform(top_n_indices)
        top_probs = probs[top_n_indices]

        results = []
        for name, prob in zip(top_names, top_probs):
            name_key = (
                name.replace("群", "")
                if cluster_skills_mapping and name not in cluster_skills_mapping
                else name
            )
            required_skills = (
                cluster_skills_mapping.get(name_key, [])
                if cluster_skills_mapping
                else []
            )

            matched_skills = [
                s for s in required_skills if s.lower() in user_skills_lower
            ]

            match_pct = (
                round(len(matched_skills) / len(required_skills) * 100, 1)
                if required_skills
                else 0
            )

            results.append(
                {
                    "mode": "explore",
                    "recommended_cluster": name,
                    "matched_skills": matched_skills,
                    "confidence": match_pct,
                }
            )

        return jsonify(results)


@app.route("/job/salary", methods=["POST"])
def predict_salary_api():
    """
    薪資預測 API
    ---
    tags:
      - AI 職缺分析
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            job_title:
              type: string
              example: "後端工程師"
            experience_years:
              type: integer
              example: 3
            skills:
              type: array
              items:
                type: string
              example: ["Python", "Docker", "SQL"]
    responses:
      200:
        description: 預測薪資結果
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    job_title = data.get("job_title")
    experience_years = data.get("experience_years")
    skills = data.get("skills")

    if job_title is None or experience_years is None or skills is None:
        return jsonify({"error": "Missing required fields"}), 400

    if not isinstance(skills, list):
        return jsonify({"error": "'skills' must be a list"}), 400

    if not salary_model or not title_cols or not mlb:
        return jsonify({"error": "Models are not loaded."}), 500

    try:
        X_skills = mlb.transform([skills])
        X_title = np.array(
            [[1 if c == f"title_{job_title}" else 0 for c in title_cols]]
        )
        X_exp = np.array([[experience_years]])
        X = np.hstack([X_title, X_exp, X_skills])
        predicted_salary = int(salary_model.predict(X)[0])

        return jsonify(
            {
                "job_title": job_title,
                "experience_years": experience_years,
                "skills": skills,
                "predicted_salary": predicted_salary,
                "unit": "TWD/月",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=3939, debug=True)