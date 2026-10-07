import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(42)  # makes the random data the same every run

def make_transaction():
    amount = round(random.expovariate(1 / 1500), 2)   # mostly small, some very large
    hour = random.randint(0, 23)
    transfers_today = random.choices([1, 2, 3, 4, 5, 6, 8], weights=[40, 25, 15, 8, 5, 4, 3])[0]
    new_customer = 1 if random.random() < 0.2 else 0
    return [amount, hour, transfers_today, new_customer]


def hidden_label(t):
    amount, hour, transfers_today, new_customer = t
    unusual = (
        amount > 8000
        or (hour < 6 and amount > 2000)
        or transfers_today >= 5
        or (new_customer == 1 and amount > 5000)
    )
    if random.random() < 0.03:   # 3% "human mistakes" to make it realistic
        unusual = not unusual
    return 1 if unusual else 0


X = [make_transaction() for _ in range(2000)]
y = [hidden_label(t) for t in X]

print(f"Created {len(X)} transactions. Unusual: {sum(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

baseline = y_test.count(0) / len(y_test)
print(f"\nLazy model (always 'normal') accuracy: {baseline:.0%}")
print(f"Our model accuracy: {accuracy_score(y_test, predictions):.0%}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["normal", "unusual"]))

FEATURES = ["amount", "hour", "transfers_today", "new_customer"]
print("Rules the model learned:")
print(export_text(model, feature_names=FEATURES))

while True:
    answer = input("\nCheck a transaction? (y/n): ")
    if answer.lower() != "y":
        break
    try:
        amount = float(input("Amount (USD): "))
        hour = int(input("Hour (0-23): "))
        transfers_today = int(input("Transfers by this sender today: "))
        new_customer = int(input("New customer? (1 = yes, 0 = no): "))
    except ValueError:
        print("Please type numbers only.")
        continue

    t = [[amount, hour, transfers_today, new_customer]]
    result = model.predict(t)[0]
    confidence = model.predict_proba(t)[0][result]

    if result == 1:
        print(f"⚠ UNUSUAL ({confidence:.0%} sure). Please double-check this transaction.")
    else:
        print(f"✅ Normal ({confidence:.0%} sure).")

    