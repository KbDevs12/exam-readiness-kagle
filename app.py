import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from models import Adaline, BackpropANN
from pipeline import load_and_prepare
from preprocessing import denormalize_exam_score, prepare_user_input
from recommendation import make_recommendations, status_from_score
from training_utils import compare_learning_rates


@st.cache_resource
def train_models():
    df, X, y_score, y_risk, meta, source_message, is_preview = load_and_prepare()

    ann = BackpropANN(input_size=X.shape[1], hidden_size=10, seed=7)
    ann_loss = ann.train(X, y_score, epochs=1800, lr=0.08)

    adaline = Adaline(input_size=X.shape[1], seed=11)
    adaline_loss = adaline.train(X, y_risk, epochs=1200, lr=0.02)
    lr_comparison = compare_learning_rates(X, y_score, input_size=X.shape[1])

    return df, meta, source_message, is_preview, ann, adaline, ann_loss, adaline_loss, lr_comparison


def plot_loss(loss, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def main():
    st.set_page_config(page_title="Exam Readiness Predictor", layout="wide")

    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Prediction", "Dataset", "Training", "About"],
            icons=["clipboard-data", "table", "graph-up", "info-circle"],
            menu_icon="list-task",
            default_index=0,
        )

    df, meta, source_message, is_preview, ann, adaline, ann_loss, adaline_loss, lr_comparison = train_models()

    st.title("Exam Readiness Predictor")
    st.caption("Backpropagation ANN + Adaline using Kaggle Student Habits dataset")

    if is_preview:
        st.warning(
            "Aplikasi sedang memakai preview CSV lokal karena dataset Kaggle belum berhasil diunduh. "
            "Jalankan `python download_dataset.py` untuk memakai dataset Kaggle asli."
        )
    else:
        st.success(source_message)

    if selected == "Prediction":
        st.header("Prediction Dashboard")
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Input Student Habit")
            values = {
                "study_hours_per_day": st.slider("Study hours per day", 0.0, 8.0, 3.0, step=0.25),
                "attendance_percentage": st.slider("Attendance percentage", 0.0, 100.0, 80.0, step=1.0),
                "sleep_hours": st.slider("Sleep hours", 0.0, 10.0, 7.0, step=0.25),
                "mental_health_rating": st.slider("Mental health rating", 1.0, 10.0, 7.0, step=0.5),
                "exercise_frequency": st.slider("Exercise frequency per week", 0.0, 7.0, 2.0, step=1.0),
                "social_media_hours": st.slider("Social media hours", 0.0, 8.0, 2.0, step=0.25),
                "netflix_hours": st.slider("Netflix/streaming hours", 0.0, 8.0, 1.0, step=0.25),
                "part_time_job": st.selectbox("Part-time job", ["No", "Yes"]),
                "diet_quality": st.selectbox("Diet quality", ["Poor", "Fair", "Good"]),
                "internet_quality": st.selectbox("Internet quality", ["Poor", "Average", "Good"]),
                "extracurricular_participation": st.selectbox("Extracurricular participation", ["No", "Yes"]),
                "parental_education_level": st.selectbox("Parental education level", ["High School", "Bachelor", "Master", "PhD"]),
            }
            predict_button = st.button("Run Prediction")

        with right:
            st.subheader("Prediction Result")
            if predict_button:
                X_new = prepare_user_input(values, meta)
                predicted_score = float(denormalize_exam_score(ann.predict(X_new))[0][0])
                risk_probability = float(adaline.predict_score(X_new)[0])
                status, explanation = status_from_score(predicted_score, risk_probability)
                
                status_label = {
                    "High Risk": "High",
                    "Medium Risk": "Medium",
                    "Low Risk": "Low"
                }.get(status, status)

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="Exam Score",
                    value=f"{predicted_score:.1f}",
                    help="Predicted Exam Score: estimasi nilai ujian berdasarkan input kebiasaan belajar dan gaya hidup.",
                    label_visibility="visible",
                )
                col2.metric(
                    label="Risk Prob.",
                    value=f"{risk_probability:.1%}",
                    help="Predicted Risk Probability: estimasi probabilitas performa rendah berdasarkan model Adaline.",
                    label_visibility="visible",
                )
                col3.metric(
                    label="Status",
                    value=status_label,
                    help=f"Risk Status: {status}. Status ditentukan dari predicted exam score dan probabilitas risiko.",
                    label_visibility="visible",
                )

                st.progress(min(max(predicted_score / 100, 0), 1))
                if status == "High Risk":
                    st.error(explanation)
                elif status == "Medium Risk":
                    st.warning(explanation)
                else:
                    st.success(explanation)

                st.subheader("Study Recommendation")
                for rec in make_recommendations(values, predicted_score):
                    st.write(f"- {rec}")
            else:
                st.info("Isi input di sisi kiri lalu klik Run Prediction.")

    elif selected == "Dataset":
        st.header("Dataset")
        st.write(source_message)
        st.write(f"Jumlah data: {len(df)} baris")
        st.dataframe(df.head(30), use_container_width=True)

        st.subheader("Exam Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["exam_score"], bins=20)
        ax.set_xlabel("Exam Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribusi Exam Score")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    elif selected == "Training":
        st.header("Training Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Backpropagation ANN")
            st.pyplot(plot_loss(ann_loss, "MSE Backpropagation ANN"))
            st.write(f"MSE awal: {ann_loss[0]:.6f}")
            st.write(f"MSE akhir: {ann_loss[-1]:.6f}")
        with c2:
            st.subheader("Adaline")
            st.pyplot(plot_loss(adaline_loss, "MSE Adaline"))
            st.write(f"MSE awal: {adaline_loss[0]:.6f}")
            st.write(f"MSE akhir: {adaline_loss[-1]:.6f}")

        st.subheader("Learning Rate Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        for lr, loss in lr_comparison.items():
            ax.plot(loss, label=f"lr={lr}")
        ax.set_title("Perbandingan Learning Rate Backpropagation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        st.write("Learning rate kecil cenderung stabil tetapi lebih lambat, sedangkan learning rate besar dapat mempercepat penurunan error tetapi berisiko tidak stabil.")

    else:
        st.header("About")
        st.markdown(
            """
            Project ini menggunakan dataset Kaggle **Student Habits vs Academic Performance**.

            Model pertama adalah Backpropagation ANN untuk prediksi nilai ujian. Model kedua adalah Adaline untuk klasifikasi risiko performa rendah.

            Package tambahan:
            - streamlit-option-menu untuk navigasi menu.
            - kagglehub untuk mengunduh dataset Kaggle.
            """
        )


if __name__ == "__main__":
    main()
