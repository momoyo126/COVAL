_base_ = ["./seg-coval.py"]

weight = "exp/stu/coval/model/model_best.pth"
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
