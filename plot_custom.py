from train.evaluation import plot_combined_decision_curves, plot_combined_roc_curves
import pandas as pd

# path_a = "final_results/mobilenet/0874/decision_20260323_170824_75ep_8000img.csv"
# path_b = "final_results/clinical/clinical3/decision_20260322_151910_19ep_8000img.csv"
# path_c = "final_results/multi_test/multiv6/decision_20260402_151622_52ep_8000img.csv"

# csv_list = [path_a, path_b, path_c]
# model_labels = [ "Imaging-only Model", "Clinical-only Model", "Fusion Model"]

# # df = pd.read_csv("final_results/multi_test/multiv6/roc_prec_recall_20260402_151622_52ep_8000img.csv")
# # print(df)
 
# # 2. Call your combined plotter
# # (Assuming 'obj' is the instance of the class containing the method)
# plot_combined_decision_curves(csv_list, model_labels)
# path_a = "final_results/mobilenet/0874/roc_prec_recall_20260323_170824_75ep_8000img.csv"
# path_b = "final_results/clinical/clinical3/roc_prec_recall_20260322_151910_19ep_8000img.csv"
# path_c = "final_results/multi_test/multiv6/roc_prec_recall_20260402_151622_52ep_8000img.csv"

# csv_list = [path_a, path_b, path_c]
# # 2. Call your combined plotter
# # (Assuming 'obj' is the instance of the class containing the method)
# plot_combined_roc_curves(csv_list, model_labels)

path_a = "results/without_hm/external_roc.csv"
path_b = "results/with_hm/external_roc.csv"
path_c = "results/external_edge/external_roc.csv"

csv_list = [path_a, path_b, path_c]
model_labels = [ "Unadjusted", "Adjusted", "Neg vs. Sev. (Adj)"]

plot_combined_roc_curves(csv_list, model_labels)