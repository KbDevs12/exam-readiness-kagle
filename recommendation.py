def status_from_score(predicted_score, risk_probability):
    if predicted_score >= 75 and risk_probability < 0.45:
        return "Low Risk", "Prediksi performa aman. Pertahankan ritme belajar dan review rutin."
    if predicted_score >= 60:
        return "Medium Risk", "Performa masih cukup, tetapi perlu memperbaiki kebiasaan belajar dan mengurangi distraksi."
    return "High Risk", "Risiko performa rendah. Fokus pada jam belajar, kehadiran, tidur, dan pengurangan screen time."


def make_recommendations(values, predicted_score):
    recs = []
    if values["study_hours_per_day"] < 2.5:
        recs.append("Tambahkan durasi belajar harian minimal 2.5 sampai 3 jam dengan sesi pendek yang konsisten.")
    if values["attendance_percentage"] < 75:
        recs.append("Tingkatkan kehadiran karena attendance menjadi salah satu sinyal penting performa akademik.")
    if values["sleep_hours"] < 6:
        recs.append("Perbaiki jam tidur. Targetkan minimal 6 jam agar konsentrasi saat belajar meningkat.")
    if values["social_media_hours"] + values["netflix_hours"] > 5:
        recs.append("Kurangi konsumsi hiburan digital. Batasi social media dan streaming sebelum ujian.")
    if values["mental_health_rating"] < 5:
        recs.append("Kondisi mental health rendah. Gunakan jadwal belajar realistis dan beri jeda istirahat.")
    if predicted_score < 60:
        recs.append("Gunakan mode prioritas: latihan soal, ringkasan bab inti, dan evaluasi materi yang paling sering keluar.")
    else:
        recs.append("Lanjutkan review, latihan soal, dan pertahankan pola belajar yang sudah berjalan.")
    return recs
