import json
import pickle
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

GOAL_TYPES = {"var":0, "and":1, "or":2, "imp":3, "bot":4}

def state_to_vector(state):

    return [state["gamma_size"],
        GOAL_TYPES[state["goal_type"]],
        state["formula_size"],
        state["max_depth"],
        state["num_vars"],
        state["num_and"],
        state["num_or"],
        state["num_imp"],
        state["num_bot"]
    ]

X = []
y = []
groups = []

with open("branch_dataset.jsonl",encoding="utf8") as f:
    for line in f:
        row=json.loads(line)
        if row["result"] != "proof":
            continue
        X.append(state_to_vector(row["state"]))
        y.append(row["choice"])
        groups.append(row["formula"])

encoder=LabelEncoder()

y=encoder.fit_transform(y)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

splitter = GroupShuffleSplit(
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

model=GradientBoostingClassifier()

model.fit(X_train,y_train)

pred=model.predict(X_test)

print("accuracy:",accuracy_score(y_test,pred))

with open("rule_model.pkl", "wb") as f:pickle.dump(model, f)

with open("rule_encoder.pkl", "wb") as f:pickle.dump(encoder, f)

pickle.dump(encoder,open("rule_encoder.pkl","wb"))

print("Всего примеров:", len(X))
print("Обучение:", len(X_train))
print("Тест:", len(X_test))