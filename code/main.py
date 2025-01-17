"""
Created on Sun Jan 17 8:00:00 2025

@author: Mengmeng Shen

Please contact for relevant datasets
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import  accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils.normlization import X_train_subset, Y_train_subset, X_test_subset, Y_test_subset, X_external_testing, Y_external_testing, X_predict, all_X, all_Y, final_data, selected_features,df
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import warnings
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import cartopy.feature as cfeature
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 7})
plt.rcParams['lines.linewidth'] = 1.5

# Define models with random_state for reproducibility where applicable
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(probability=True, random_state=42))
    ], voting='soft'),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
}

# Parameters for tuning
param_grids = {
    'K-Nearest Neighbors': {'n_neighbors': [2,3,4, 5, 6,7,8]},'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'Random Forest': {'n_estimators': [73,76,85],
                      'max_depth': [5,7,9,10,12]},
    'Decision Tree': {'max_depth': [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]},
    'Support Vector Machine': {'C': [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6], 'kernel': ['linear', 'rbf']},
    'Gradient Boosting': {'n_estimators': [50,51,52,53,54,55,56,57,59], 'learning_rate': [0.1,0.2,0.5]},
    'Voting': {},
    'AdaBoost': {'n_estimators': [310,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340], 'learning_rate': [ 0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]},
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform Grid Search and Train Models
best_models = {}
for name, model in models.items():
    print(f"Training {name}...")

    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grids[name], cv=kf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_subset, Y_train_subset)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Train on the full dataset (all_X, all_Y)
    best_model.fit(all_X, all_Y)
    best_models[name] = {
        'Model': best_model,
        'Best Parameters': best_params,
    }

    # Print the best parameters for each model
    print(f"Best Parameters for {name}: {best_params}")

# Evaluate models on both training and testing subsets
results = {}
for name, model_info in best_models.items():
    model = model_info['Model']

    # Predict on training set
    y_train_subset_pred = model.predict(X_train_subset)
    train_accuracy = accuracy_score(Y_train_subset, y_train_subset_pred)
    train_f1 = f1_score(Y_train_subset, y_train_subset_pred, average='weighted')

    # Predict on test subset
    y_test_subset_pred = model.predict(X_test_subset)
    test_accuracy = accuracy_score(Y_test_subset, y_test_subset_pred)
    test_f1 = f1_score(Y_test_subset, y_test_subset_pred, average='weighted')

    # Store the results for both training and testing
    results[name] = {
        'Train Accuracy': train_accuracy,
        'Train F1': train_f1,
        'Test Accuracy': test_accuracy,
        'Test F1': test_f1
    }

    # Print the results for each model
    print(f"\n{name} - Training Set:")
    print(f"Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}")
    print(f"{name} - Testing Set:")
    print(f"Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")

# Sorting Models by Test Accuracy
sorted_models = sorted(results.items(), key=lambda x: x[1]['Test Accuracy'], reverse=True)

# Get the best model based on test accuracy
best_one = sorted_models[:1]
print("\nBest Model on Test Set:")
for model_name, metrics in best_one:
    print(f"{model_name} - Train Accuracy: {metrics['Train Accuracy']:.4f}, Train F1: {metrics['Train F1']:.4f}")
    print(f"{model_name} - Test Accuracy: {metrics['Test Accuracy']:.4f}, Test F1: {metrics['Test F1']:.4f}")

# Print classification reports for all 7 models
for model_name, model_info in best_models.items():
    model = model_info['Model']

    # Predict on the test subset
    y_test_subset_pred = model.predict(X_test_subset)

    # Print the classification report for each model
    print(f"\n{model_name} Classification Report:")
    print(classification_report(Y_test_subset, y_test_subset_pred, target_names=['0', '1']))\

# Access to the best models
best_rf_model = best_models['Random Forest']['Model']
y_test_subset_pred = best_rf_model.predict(X_test_subset)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test_subset, y_test_subset_pred)

# Assuming cm, best_rf_model, selected_features, final_data, and filtered_data are already defined
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(4.83, 2.42))

# Confusion matrix plotting (ax2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Algoma', 'Superior'])
disp.plot(ax=ax2, cmap=plt.cm.Blues, colorbar=False)
ax2.set_ylabel('True label', fontsize=7)
ax2.set_xlabel('Predicted label', fontsize=7)
for text in disp.ax_.texts:
    text.set_fontsize(7)

# Add subgraph label “(A)”
ax2.text(0.02, 0.92, 'A', transform=ax2.transAxes, fontsize=7, verticalalignment='bottom', horizontalalignment='left')

# Rotate y-axis labels (Algoma and Superior) in ax2 to -90 degrees
ax2.set_yticklabels(['Algoma', 'Superior'], rotation=90, fontsize=7)

# Assuming best_rf_model and selected_features are already defined
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [selected_features[i] for i in indices]
sorted_importances = importances[indices]

bars = ax1.bar(sorted_features, sorted_importances, color='skyblue')
ax1.set_ylabel(r'Relative feature importance', fontsize=7)
ax1.set_xticklabels(sorted_features, rotation=45, fontsize=7)
ax1.set_ylim(0, 0.2)

# Update for input feature label: Convert 'SN' to subscript in x-axis labels
formatted_features = [feature.replace("SN", r"$\mathrm{_{SN}}$") for feature in sorted_features]
ax1.set_xticklabels(formatted_features, rotation=90, fontsize=7, fontname='Arial')
ax1.tick_params(axis='y', labelsize=7)

# Add subgraph label “(B)”
ax1.text(0.02, 0.92, 'B', transform=ax1.transAxes, fontsize=7, verticalalignment='bottom', horizontalalignment='left')

# Adjust font size of confusion matrix axis labels
ax2.tick_params(axis='x', labelsize=7)
ax2.tick_params(axis='y', labelsize=7)

# Increase space between subplots and adjust layout for better label visibility
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3)

# Save the plot
plt.tight_layout()
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 1.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 1.tiff', format='tiff', dpi=400)

# Drawing a violin
normalized_feature_names = [f"{feature}_normd" for feature in sorted_features]
filtered_data = final_data[final_data['Label'].isin([0, 1])]

plt.figure(figsize=(4.83, 4.29))
for i, feature in enumerate(normalized_feature_names):
    plt.subplot(3, 3, i + 1)
    sns.violinplot(x='Label', y=feature, data=filtered_data, inner='quartile')
    plt.xlabel('')

    # Apply subscript for SN in y-axis label without changing the font
    # Decreased labelpad to shorten the distance between y-axis and label
    plt.ylabel(sorted_features[i].replace("SN", r"$_{\text{SN}}$"), fontsize=7, fontname='Arial', labelpad=1)

    plt.xticks(ticks=[0, 1], labels=['Algoma', 'Superior'], fontsize=7, fontname='Arial')

    # Apply subscript to x-axis label if necessary (though labels for x-axis are already in desired format)
    plt.gca().set_xticklabels([r"$\text{Algoma}$", r"$\text{Superior}$"], fontsize=7)

# Tight layout to minimize space and adjust bottom margin
plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 2.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 2.tiff', format='tiff', dpi=400)

# Filter out different types of data
algoma_data = df[df['Type'] == 'Algoma']['Age(Ga)']
superior_data = df[df['Type'] == 'Superior']['Age(Ga)']

# Creating Histograms
plt.figure(figsize=(4.83, 2.90))

# Plotting Algoma-type histograms (red)
plt.hist(algoma_data, bins=20, color='red', alpha=0.7, label='Algoma', edgecolor='black', density=True)

# Plotting Superior-type histograms (blue)
plt.hist(superior_data, bins=20, color='blue', alpha=0.7, label='Superior', edgecolor='black', density=True)

# Use set_xlim to set the range of the X-axis (make sure the range is decreasing)
plt.gca().set_xlim(4.0, 1.8)

# Kernel Density Estimation (KDE) curves on histograms
sns.kdeplot(algoma_data, color='red', linestyle='-', label='Algoma KDE', linewidth=2)
sns.kdeplot(superior_data, color='blue', linestyle='-', label='Superior KDE', linewidth=2)

# Add title and tags
plt.xlabel('Age(Ga)', fontsize=7)
plt.ylabel('Density', fontsize=7)

# Show legend
plt.legend()

# Adjust layout to ensure all labels are visible
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

# Save the figure
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 3.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 3.tiff', format='tiff', dpi=400)

craton_order = [
    'Kaapvaal Craton', 'Pilbara Craton', 'Superior Craton', 'Central Wyoming Craton',
    'São Francisco Craton', 'West Africa Craton', 'North China Craton'
]

# Filter data and get unique craton
df_carton = df[df['Craton'].isin(craton_order)]

# Get all the different craton
unique_cratons = df_carton['Craton'].unique()
num_cratons = len(unique_cratons)

# Create a 7x1 subgraph grid with custom spacing using GridSpec
fig = plt.figure(figsize=(4.83, 6.28))
gs = GridSpec(7, 1, figure=fig)

# Create axis objects for each subplot
axes = [fig.add_subplot(gs[i]) for i in range(7)]

# Define horizontal and vertical coordinate ranges
xlim = [4.0, 1.8]
ylim = [0, 10]

# Define groupings
group_1 = craton_order[:2]
group_2 = craton_order[2:4]
group_3 = craton_order[4:]

# Plot KDE curves for each craton
uniform_yticks = [0, 2, 4, 6, 8, 10]

# Plot KDE curves for each craton
for i, craton in enumerate(craton_order):
    ax = axes[i]

    # Extract data from the current craton
    craton_data = df_carton[df_carton['Craton'] == craton]

    # Plotting KDE curves
    sns.kdeplot(data=craton_data[craton_data['Type'] == 'Algoma']['Age(Ga)'],
                ax=ax, label='Algoma', fill=True, color='red')
    sns.kdeplot(data=craton_data[craton_data['Type'] == 'Superior']['Age(Ga)'],
                ax=ax, label='Superior', fill=True, color='blue')

    # Setting the horizontal and vertical coordinate ranges
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add craton names to the top left of each subfigure with font size 7
    ax.text(0.03, 0.93, craton, transform=ax.transAxes, fontsize=7, color='black', ha='left', va='top')

    # Show horizontal coordinates only in the first subplot of each group
    if i in [0, 2, 4, 5]:
        ax.set_xticklabels([])

    # Apply consistent vertical spacing (yticks) for all subplots
    ax.set_yticks(uniform_yticks)

    # Add vertical dashed lines with labels at specific locations in groups
    if craton in group_1:
        ax.axvline(x=3.0, color='black', linestyle='--', linewidth=1.0)
        if i == 0:  # For the first subplot in group 1
            ax.text(3.2, 7, '3.0Ga', fontsize=7, verticalalignment='center', color='black')

    if craton in group_2:
        ax.axvline(x=2.75, color='black', linestyle='--', linewidth=1.0)
        if i == 2:  # For the first subplot in group 2
            ax.text(3.0, 7, '2.75Ga', fontsize=7, verticalalignment='center', color='black')

    if craton in group_3:
        ax.axvline(x=2.65, color='black', linestyle='--', linewidth=1.0)
        if i == 4:  # For the first subplot in group 3
            ax.text(2.9, 7, '2.65Ga', fontsize=7, verticalalignment='center', color='black')

# Add group headings (displayed only on the first subfigure of each group)
for i, ax in enumerate(axes[:2]):
    if i == 0:
        ax.set_title('Vaalbara Supercratons', fontsize=9, fontweight='bold', color='black', loc='center', pad=9)

for i, ax in enumerate(axes[2:4]):
    if i == 0:
        ax.set_title('Superia Supercratons', fontsize=9, fontweight='bold', color='black', loc='center', pad=8)

for i, ax in enumerate(axes[4:]):
    if i == 0:
        ax.set_title('Nunavutia Supercratons', fontsize=9, fontweight='bold', color='black', loc='center', pad=8)

# Place the legend for each subgroup in the upper left corner outside the first subfigure of each group with font size 7
axes[0].legend(loc='lower center', fontsize=7, bbox_to_anchor=(0.91, 1.00), ncol=1)

# Adjust vertical spacing between groups using hspace
gs.update(hspace=0.75)  # Increase hspace to create more space between groups

# Adjustment of position by axes (fine-tuning the exact position of each group of subgraphs)
axes[0].set_position([0.1, 0.84, 0.8, 0.1])
axes[1].set_position([0.1, 0.74, 0.8, 0.1])

axes[2].set_position([0.1, 0.55, 0.8, 0.1])
axes[3].set_position([0.1, 0.45, 0.8, 0.1])

axes[4].set_position([0.1, 0.26, 0.8, 0.1])
axes[5].set_position([0.1, 0.16, 0.8, 0.1])
axes[6].set_position([0.1, 0.06, 0.8, 0.1])

# Ensure tight layout with adjusted space
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 4.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure 4.tiff', format='tiff', dpi=400)

# Predicting results using the best model
X_train_subset = pd.DataFrame(X_train_subset)
X_test_subset = pd.DataFrame(X_test_subset)
X_external_testing = pd.DataFrame(X_external_testing)
X_predict = pd.DataFrame(X_predict)
all_X_1667 = pd.concat([X_train_subset, X_test_subset, X_external_testing,X_predict], ignore_index=True)
all_Y_1667_predict = {}

# Iterate over each best model for prediction
for model_name, _ in best_one:
    model = best_models[model_name]['Model']
    all_Y_1667_predict[model_name] = model.predict(all_X_1667)

# Add prediction results to final_data
for model_name in all_Y_1667_predict:
    final_data[f'{model_name} - Label'] = all_Y_1667_predict[model_name]

# Analysis of forecast results
labels = ['Porpoise Cove', 'Nanfen', 'Griquatown','Gunflint','Unknown']
for label in labels:
    filtered_df = final_data[final_data['Iron Formation'] == label]
    for model_name, _ in best_one:
        label_column = f'{model_name} - Label'
        count_0 = (filtered_df[label_column] == 0).sum()
        count_1 = (filtered_df[label_column] == 1).sum()
        # Print the count of predictions
        print(f"{model_name} - Label: {label}, Count of '0': {count_0}")
        print(f"{model_name} - Label: {label}, Count of '1': {count_1}")

# output result
final_data.to_excel(r"D:\iron_formation_recognition_main\result\Database-with-prediction.xlsx", index=False)

# Function to clean longitude
def clean_longitude(lon):
    lon = str(lon)
    if '° W' in lon:
        return -float(lon.replace('° W', '').strip())
    elif '° E' in lon:
        return float(lon.replace('° E', '').strip())
    else:
        return None

# Function to clean latitude
def clean_latitude(lat):
    lat = str(lat)
    if '° S' in lat:
        return -float(lat.replace('° S', '').strip())
    elif '° N' in lat:
        return float(lat.replace('° N', '').strip())
    else:
        return None

# Clean the data (assuming final_data is already loaded)
final_data['Longitude'] = final_data['Longitude'].apply(clean_longitude)
final_data['Latitude'] = final_data['Latitude'].apply(clean_latitude)

# Clean the data (assuming final_data is already loaded)
df['Longitude'] = df['Longitude'].apply(clean_longitude)
df['Latitude'] = df['Latitude'].apply(clean_latitude)

# Labels dictionary
labels = {0: 'Algoma-type', 1: 'Superior-type', 2: 'Application data'}

# Function to plot the map for a specific set (training/testing/prediction)
def plot_map(data, set_type, label_type=None, show_legend=True):
    # Create a new figure and axes for plotting
    fig = plt.figure(figsize=(5.83, 4.37))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Draw coastlines and additional geographic features
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)  # Add rivers
    rivers_110m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '110m')
    ax.add_feature(rivers_110m)

    # Add gridlines with labels for longitude and latitude, with increased font size
    gridlines = ax.gridlines(draw_labels=True, xlocs=[-179, -90, 0, 90, 179])
    gridlines.xlabel_style = {'fontsize': 7}
    gridlines.ylabel_style = {'fontsize': 7}

    # Add background image
    ax.stock_img()

    # Initialize legend labels
    handles = []
    added_colors = set()

    # Filter data based on 'Set' column and optionally the 'Label' column
    filtered_data = data[data['Set'] == set_type]

    if label_type is not None:
        filtered_data = filtered_data[filtered_data['Label'] == label_type]

    # Map numeric labels to descriptions
    label_map = {0: 'Algoma-type', 1: 'Superior-type', 2: 'Application data'}

    # Plot the data points
    for index, row in filtered_data.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        label = row['Label']

        if pd.notna(lat) and pd.notna(lon):
            # Convert lat, lon to NumPy arrays
            lon_array = np.array([lon])
            lat_array = np.array([lat])

            # Convert lat, lon to x, y using the projection
            transformed_points = ax.projection.transform_points(ccrs.PlateCarree(), lon_array, lat_array)
            x, y = transformed_points[:, 0], transformed_points[:, 1]

            # Define colors based on the 'Label' column
            if label == 0:
                color = 'red'
            elif label == 1:
                color = 'blue'
            elif label == 2:
                color = 'orange'
            else:
                continue

            # Plot the point on the map with a larger marker size
            ax.plot(x, y, 'o', color=color, markersize=7)

            # Add legend handles if color hasn't been added yet
            if color not in added_colors:
                # Map the numeric label to its description
                label_description = label_map.get(label, 'Unknown')
                handle = mlines.Line2D([0], [0], marker='o', color='w', label=label_description,
                                       markerfacecolor=color, markersize=7)
                handles.append(handle)
                added_colors.add(color)

    # Add legend if required
    if show_legend:
        # Re-order handles to ensure the correct order in the legend
        handles_sorted = sorted(handles, key=lambda h: h.get_label())
        plt.legend(handles=handles_sorted, loc='upper left', bbox_to_anchor=(0.01, 0.25),
                   fontsize=7, handleheight=1.5, handlelength=2.0, labelspacing=0.5)

    # Use tight layout to remove extra whitespace, set pad=0 to remove space entirely
    plt.tight_layout(pad=0)  # Removed extra padding

    # Remove extra space on the sides and adjust axis limits (if needed)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Merge 'Training set' and 'Testing set' into one category for plotting
final_data['Set'] = final_data['Set'].replace(['Training set', 'Testing set'], 'Training/Testing set')

# Plot Map 1: Total datasets combined into one plot
plot_map(df[df['Set'] == 'Total set'], 'Total set')
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S1.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S1.tiff', format='tiff', dpi=400)

# Plot Map 2: Training and Testing subsets combined into one plot
plot_map(final_data[final_data['Set'] == 'Training/Testing set'], 'Training/Testing set')
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S2.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S2.tiff', format='tiff', dpi=400)

# Plot Map 3: External testing datasets combined into one plot
plot_map(final_data[final_data['Set'] == 'External testing set'], 'External testing set')
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S3.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S3.tiff', format='tiff', dpi=400)

# Plot Map 4: Prediction datasets combined into one plot
plot_map(final_data[final_data['Set'] == 'Prediction set'], 'Prediction set')
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S4.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S4.tiff', format='tiff', dpi=400)

# Merge training and application data
X = np.vstack((all_X, X_predict))

# Principal Component Analysis (PCA) to reduce the data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get PCA results for all_X and X_predict
X_pca_all = X_pca[:len(all_X), :]
X_pca_predict = X_pca[len(all_X):, :]

# Create a new figure for the scatter plot
plt.figure(figsize=(4.83, 3.6))

# Plotting a scatter plot of PCA results with smaller point sizes
plt.scatter(X_pca_all[:, 0], X_pca_all[:, 1], c='red', marker='o', label='Training Data', alpha=0.7, s=20)
plt.scatter(X_pca_predict[:, 0], X_pca_predict[:, 1], c='blue', marker='o', label='Application Data', alpha=0.7, s=10)

# Set labels and title
plt.xlabel('PC1', fontsize=7)
plt.ylabel('PC2', fontsize=7)
plt.legend(fontsize=7)
plt.grid(False)

# Adjust layout for better spacing to ensure axis labels are not cut off
plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12)

# Save the plot
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S5.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S5.tiff', format='tiff', dpi=400)

# Create 2 rows, 4 columns subgraph layout (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(5.83, 4.38))

# Element combinations and tags
combinations = [
    ('(La/Sm)SN', 'Ni(ppm)', '(a) (La/Sm)SN vs Ni(ppm)', 'a'),
    ('(La/Yb)SN', 'Ni(ppm)', '(b) (La/Yb)SN vs Ni(ppm)', 'b'),
    ('Mn/Fe', 'Ni(ppm)', '(c) Mn/Fe vs Ni(ppm)', 'c'),
    ('Ni(ppm)', '(Eu/Eu*)SN', '(d) Ni(ppm) vs (Eu/Eu*)SN', 'd'),
]

# Plotting scatter plots and kernel densities for each combination
for i, (x_col, y_col, title, label) in enumerate(combinations):
    # Use divmod(i, 4) to determine where to draw, assigning to two rows and four columns
    row, scatter_col = divmod(i, 4)
    kde_col = scatter_col

    # Segmentation of data according to Y_train
    X_train_algoma = X_train_subset[Y_train_subset == 0]
    X_train_superior = X_train_subset[Y_train_subset == 1]

    # Scatterplotting (first row), reduced marker size (s=10 for smaller points)
    scatter_algoma = sns.scatterplot(x=x_col, y=y_col, data=X_train_algoma, ax=axes[0, scatter_col], color='red',
                                     label='Algoma', alpha=0.6, s=10)  # Smaller size
    scatter_superior = sns.scatterplot(x=x_col, y=y_col, data=X_train_superior, ax=axes[0, scatter_col], color='blue',
                                       label='Superior', alpha=0.6, s=10)  # Smaller size

    # Set font size for X and Y axis labels in scatter plot to 7, with SN as subscript
    axes[0, scatter_col].set_xlabel(x_col.replace("SN", r"$_{\text{SN}}$"), fontsize=7, fontname='Arial')
    axes[0, scatter_col].set_ylabel(y_col.replace("SN", r"$_{\text{SN}}$"), fontsize=7, fontname='Arial')

    # Set y-axis limits for the first row (scatter plots) from -6 to 6
    axes[0, scatter_col].set_ylim(-6, 6)

    # Set x-axis limits for scatter plots from -5 to 5
    axes[0, scatter_col].set_xlim(-5, 5)

    # Add a legend for the scatterplot
    axes[0, scatter_col].legend(loc='upper right', fontsize=7)

    # Plot KDE (second row) and set labels
    sns.kdeplot(x=x_col, y=y_col, data=X_train_algoma, ax=axes[1, kde_col], color='blue', linewidth=2, alpha=0.4,
                bw_adjust=2, fill=True)
    sns.kdeplot(x=x_col, y=y_col, data=X_train_superior, ax=axes[1, kde_col], color='orange', linewidth=2, alpha=0.4,
                bw_adjust=2, fill=True)

    # Manual setup legend with smaller font size
    handles = [
        Line2D([0], [0], color='blue', lw=2, label='Algoma'),
        Line2D([0], [0], color='orange', lw=2, label='Superior')
    ]

    # Add a legend for the KDE chart with automatic adjustment for better positioning
    axes[1, kde_col].legend(handles=handles, loc='upper right', fontsize=7, bbox_to_anchor=(1.1, 1), frameon=False)

    # Set font size for X and Y axis labels in KDE plot to 7, with SN as subscript
    axes[1, kde_col].set_xlabel(x_col.replace("SN", r"$_{\text{SN}}$"), fontsize=7, fontname='Arial')
    axes[1, kde_col].set_ylabel(y_col.replace("SN", r"$_{\text{SN}}$"), fontsize=7, fontname='Arial')

    # Set y-axis limits for the second row (KDE plots) from -6 to 6
    axes[1, kde_col].set_ylim(-6, 6)

    # Set x-axis limits for KDE plots from -5 to 5
    axes[1, kde_col].set_xlim(-5, 5)

# Adjust layout to avoid overlapping tags and clipping
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S6.pdf', format='pdf', dpi=400)
plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S6.tiff', format='tiff', dpi=400)

for model_name, metrics in best_one:
    model_info = best_models[model_name]
    model = model_info['Model']

    # Make predictions about X_predict
    y_external_testing_pred = model.predict(X_external_testing)

    # Output confusion matrix
    cm = confusion_matrix(Y_external_testing, y_external_testing_pred)
    plt.figure(figsize=(4.83, 3.86))

    # Create the heatmap for the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Algoma', 'Superior'], yticklabels=['Algoma', 'Superior'],
                cbar=False)

    # Set labels for the axes
    plt.xlabel('Predicted label', fontsize=7)
    plt.ylabel('True label', fontsize=7)

    # Adjust layout to avoid clipping of labels
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.15)

    # Save the figure
    plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S8.pdf', format='pdf', dpi=400)
    plt.savefig('D:/iron_formation_recognition_main/Figures/Figure S8.tiff', format='tiff', dpi=400)




