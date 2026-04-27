import optuna
import json
import os

def export_best_params(db_name="kiln_opt.db", output_file="best_params.json"):
    # 1. Veritabanı yolunu belirle
    # Eğer dosya optimization klasöründeyse bir üst dizine bak
    if not os.path.exists(db_name):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", db_name)
    else:
        db_path = db_name

    try:
        # 2. Çalışmayı (Study) yükle
        study = optuna.load_study(
            study_name="kiln_optimization_v2", 
            storage=f"sqlite:///{db_path}"
        )

        # 3. En iyi parametreleri ve skoru al
        best_trial = study.best_trial
        
        output_data = {
            "trial_number": best_trial.number,
            "best_value": -best_trial.value,  # Minimize ettiğimiz için eksi ile çarpıyoruz
            "params": best_trial.params,
            "datetime": best_trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if best_trial.datetime_start else "N/A"
        }

        # 4. JSON olarak kaydet
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"--- En İyi Parametreler Başarıyla Kaydedildi ---")
        print(f"Dosya: {output_file}")
        print(f"Deneme No: {output_data['trial_number']}")
        print(f"En İyi C3S Skoru: {output_data['best_value']:.6f}")
        
        # Terminale de parametreleri yazdır
        print("\nParametre Detayları:")
        for k, v in best_trial.params.items():
            print(f"  {k}: {v}")

    except KeyError:
        print(f"Hata: '{db_name}' içinde 'kiln_optimization' isimli bir çalışma bulunamadı.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    export_best_params()