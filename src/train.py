from __future__ import annotations

from src.data import build_dataset_from_raw, make_train_test_split, split_features_target
from src.modeling import (
    evaluate_models,
    prepare_output_dirs,
    save_all_figures,
    save_best_model,
    save_predictions,
    select_best_model,
    train_models,
    write_analysis,
)


def run_pipeline():
    prepare_output_dirs()
    clean_df = build_dataset_from_raw()
    X, y = split_features_target(clean_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    trained_models, tuning_df = train_models(X_train, X_test, y_train, y_test)
    metrics_df = evaluate_models(trained_models, y_test)
    save_predictions(trained_models, y_test)
    figure_paths = save_all_figures(trained_models, metrics_df, tuning_df, y_test)
    best_model = select_best_model(trained_models, metrics_df)
    metadata = save_best_model(best_model, metrics_df, tuning_df, figure_paths)
    analysis = write_analysis(metrics_df, metadata)
    return metrics_df, metadata, analysis


def main() -> None:
    metrics_df, metadata, analysis = run_pipeline()
    print(metrics_df.to_string(index=False))
    print()
    print(analysis)
    print()
    print(f"Best model: {metadata['best_model']}")


if __name__ == "__main__":
    main()
