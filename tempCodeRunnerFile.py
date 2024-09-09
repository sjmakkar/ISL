import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Check for inconsistent sample lengths
max_length = max(len(sample) for sample in data)
print(f"Max sample length: {max_length}")

# Option 1: Pad all samples with zeros to the maximum length
data_padded = np.array([np.pad(sample, (0, max_length - len(sample))) for sample in data])

# Alternatively, Option 2: Truncate all samples to the minimum length
# min_length = min(len(sample) for sample in data)
# data_truncated = np.array([sample[:min_length] for sample in data])

# Convert labels to numerical values
labels = np.asarray(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)

# Check shapes before training
print(f"Data shape: {data_padded.shape}")
print(f"Labels shape: {labels.shape}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
