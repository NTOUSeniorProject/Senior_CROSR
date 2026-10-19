"""十折實驗專用設定；既有超參數仍以 constants.TRAINING_CONFIG 為主。"""

from constants import TRAINING_CONFIG, project_path

SUBJECT_GROUPS = (
    (7,), (8,), (2, 15), (3, 17), (19, 28), (16, 27), (1, 25),
    (4, 13, 18, 37),
    (5, 6, 9, 10, 11, 12, 20, 21, 39),
    (14, 22, 23, 24, 26, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40),
)

# 僅複製訓練需要的欄位，不把推論端設定或連線資訊寫進實驗結果。
CV_CONFIG = {
    key: TRAINING_CONFIG[key] for key in (
        "data_root", "known_actions", "max_frames", "num_nodes",
        "center_joint_idx", "feat_dim", "num_epochs", "batch_size",
        "lr_model", "weight_decay", "lambda_center", "lr_center",
        "max_lambda_recon", "warmup_epochs",
    )
}
CV_CONFIG.update(
    lambda_entropy=TRAINING_CONFIG.get("lambda_entropy", 0.0002),
    seed=42, num_workers=0, known_fpr=0.05, cosine_weight=0.6,
    output_root=str(project_path("output/cross_validation")),
)
