_base_ = ["./seg-minknet34-syn.py"]

weight = "exp/stu/minknet34-syn/model/model_best.pth"
data = dict(test=dict(split="test"))
test = dict(
    _delete_=True,
    type="AnomSegTester",
    verbose=True,
    save_conf=False,
    save_prediction=True,
    save_anomaly_probability=True,
    anomaly_class=19,
)
