import optuna
import json

# 1. Veritabanı bilgilerini tanımla
DB_PATH = "sqlite:///fast_kiln.db" # Veritabanı dosya adın
STUDY_NAME = "kiln_v9_8_unleashed"       # Optuna çalışmandaki isim

try:
    # 2. Mevcut çalışmayı veritabanından yükle
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    
    # 3. En iyi parametreleri al
    best_params = study.best_params
    best_score = study.best_value
    est_mae = best_score / 5.0

    output_data = {
        "study_name": STUDY_NAME,
        "best_score_weighted": round(best_score, 4),
        "estimated_mae": round(est_mae, 4),
        "params": best_params
    }

    # 4. JSON olarak kaydet
    with open("best_mpc_params.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print("✅ JSON başarıyla oluşturuldu!")
    print(f"🏆 En İyi Skor: {best_score:.2f}")

except Exception as e:
    print(f"❌ Hata: {e}")