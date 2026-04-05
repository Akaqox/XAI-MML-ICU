import pandas as pd
import matplotlib.pyplot as plt

# 1. Define file paths (Change these if your file is in a different folder)
input_filename = 'dataset/shap_analysis.csv' 
output_filename = 'dataset/shap_data_with_averages.csv'

# 2. Load the CSV file
df = pd.read_csv(input_filename)

# 3. Calculate the average across the 5 lists for each row and round it
# df.iloc[:, 1:] selects all rows, and all columns from index 1 to the end (the percentage columns)
df['Average (%)'] = df.iloc[:, 1:].mean(axis=1).round(2)

# 4. Sort all features by average in descending order
df_sorted = df.sort_values(by='Average (%)', ascending=False)

# 5. Display the top 10 features in the console
print("Top 10 Features by Average SHAP Contribution:")
print(df_sorted[['Feature', 'Average (%)']].head(10))

# 6. Save the full results to a new CSV file
df_sorted.to_csv(output_filename, index=False)
print(f"\nFull results successfully saved to {output_filename}")

# 7. Extract the Top 21 features for plotting
top_21_df = df_sorted.head(21)

# 8. Sort values ascending for better horizontal bar visualization (highest at the top)
plot_df = top_21_df.sort_values(by="Average (%)", ascending=True)

# 9. Plotting
plt.figure(figsize=(10, 8))
bars = plt.barh(plot_df["Feature"], plot_df["Average (%)"], color='skyblue')

# Add data labels (percentages) next to each bar
for bar in bars:
    plt.text(
        bar.get_width() + 0.1,              # X position (slightly to the right of the bar)
        bar.get_y() + bar.get_height() / 2, # Y position (center of the bar)
        f'{bar.get_width():.2f}%',          # Text format (e.g., 5.34%)
        va='center'                         # Vertical alignment
    )

# Format the chart
plt.xlabel("Average SHAP Contribution (%)")
plt.title("Top 21 Clinical Features by Average SHAP Contribution")
plt.tight_layout() # Ensures labels are not cut off

# Display the plot
plt.savefig("results/shap_analysis.png", dpi=500)
plt.show()