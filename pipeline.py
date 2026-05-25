from data_loader import load_raw_dataset
from preprocessing import prepare_training_data


def load_and_prepare():
    raw_df, source_message, is_preview = load_raw_dataset(allow_preview=True)
    clean_df, X, y_score, y_risk, meta = prepare_training_data(raw_df)
    return clean_df, X, y_score, y_risk, meta, source_message, is_preview
