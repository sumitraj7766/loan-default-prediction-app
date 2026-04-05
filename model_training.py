from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def train_model(df):

    X = df.drop("Default", axis=1)
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=62
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.2,
        scale_pos_weight=4   
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model