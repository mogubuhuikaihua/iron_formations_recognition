from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
import matplotlib.ticker as plticker
from sklearn.svm import SVC
from utils.normlization import X_train_subset, Y_train_subset, X_test_subset, Y_test_subset, all_X, all_Y

# Initialize Random Forest model and GridSearchCV
svc = SVC(random_state=42)
param_grid = {'C': [3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3], 'kernel': ['linear','rbf']}
svc_grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10)
svc_grid_search.fit(X_train_subset, Y_train_subset)

# Use best parameters for Random Forest
best_svc_model = svc_grid_search.best_estimator_
best_svc_model.fit(all_X, all_Y)

# Calculate the accuracy of the training set
train_accuracy = best_svc_model.score(X_train_subset, Y_train_subset)
val_accuracy = cross_val_score(best_svc_model, X_train_subset, Y_train_subset, cv=10, scoring='accuracy').mean()
print("Best parameters for Random Forest:", svc_grid_search.best_params_)
print(
    f"Training accuracy: {train_accuracy:.4f}, Verification of accuracy: {val_accuracy:.4f}, Differences in accuracy: {train_accuracy - val_accuracy:.4f}")

# Mapping the learning curve
f, ax = plt.subplots(figsize=(4, 3))
train_sizes = np.linspace(0.1, 1.0, 5),
train_sizes, train_scores, test_scores = learning_curve(
    best_svc_model, all_X, all_Y, cv=5, train_sizes=train_sizes, random_state=2)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, marker='o', color="r",
         label="Training accuracy", alpha=0.6, markersize='3')

plt.plot(train_sizes, test_scores_mean, marker='o', color="#F97306",
         label="Cross-validation accuracy", alpha=0.6, markersize='3')

plt.xlabel("The number of samples in training set", fontsize=8)
plt.ylabel("Accuracy", fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=50))
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.xlim(50, 879)
plt.ylim(0.4, 1.1)
plt.tight_layout()

# plt.plot([0, 160], [0.9, 0.9], color="k", linestyle='--',lw=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.draw()
plt.show()

# Define feature selection function
def perform_feature_selection(model, X_train_subset, X_test_subset, Y_train_subset, Y_test_subset, model_name):
    sfs = SequentialFeatureSelector(model, n_features_to_select=1, direction='forward', cv=10)
    selected_features = []
    min_diff = float('inf')
    best_model = None

    for i in range(X_train_subset.shape[1]):
        sfs.n_features_to_select = i + 1
        sfs.fit(X_train_subset, Y_train_subset)

        current_features = all_X.columns[sfs.get_support()]
        X_train_subset_selected = X_train_subset[current_features]
        X_test_subset_selected = X_test_subset[current_features]

        model.fit(X_train_subset_selected, Y_train_subset)
        train_accuracy = accuracy_score(Y_train_subset, model.predict(X_train_subset_selected))
        test_accuracy = accuracy_score(Y_test_subset, model.predict(X_test_subset_selected))
        diff = abs(train_accuracy - test_accuracy)

        print(f"{model_name} - Number of features: {i + 1}, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}, Difference: {diff:.4f}")
        print(f"{model_name} - Current features:", list(current_features))

        if diff < min_diff:
            min_diff = diff
            selected_features = current_features
            best_model = model

    return selected_features, best_model

# Perform feature selection
print("\nFor Random Forest:")
svc_selected_features, best_svc_model = perform_feature_selection(best_svc_model, X_train_subset, X_test_subset, Y_train_subset, Y_test_subset, "Random Forest")

# Output final selected features
print("Final selected features for Random Forest:", list(svc_selected_features))

# Save the final RF model
joblib.dump(best_svc_model, 'final_svc_model.pkl')

# Compute and print final model's accuracy on training and test sets
final_train_accuracy = accuracy_score(Y_train_subset, best_svc_model.predict(X_train_subset))
final_test_accuracy = accuracy_score(Y_test_subset, best_svc_model.predict(X_test_subset))
print(f"\nFinal Random Forest Model Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Random Forest Model Testing Accuracy: {final_test_accuracy:.4f}")

# Plot feature importances
importances = best_svc_model.feature_importances_
feature_names = all_X.columns
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances - Random Forest')
plt.bar(range(all_X.shape[1]), importances[indices], align='center')
plt.xticks(range(all_X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()