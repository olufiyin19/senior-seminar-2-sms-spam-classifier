import pandas as pd
import matplotlib.pyplot as plt

messages = pd.read_csv(
    "data/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print(messages.head())
print("\nNumber of messages:", len(messages))

# Count ham v spam messages
print("\nLabel counts:")
print(messages["label"].value_counts())

# ham v spam percentages
print("\nLabel percentages:")
print(messages["label"].value_counts(normalize=True) * 100)

label_counts = messages["label"].value_counts()

label_counts.plot(kind="bar")

plt.title("SMS Spam Class Distribution")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.tight_layout()
plt.savefig("results/class_distribution.png")
plt.close()