
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESN_utilities import nrmse, psd_nrmse
from IPython.display import display, Image
import plotly.graph_objects as go



def show_df_as_image(df,header_color='paleturquoise',
                     row_colors=['beige','azure'],header_height=40,row_height=40,
                     img_folder="images",img_name="dataframe_image.png",
                     img_width=1500):

        # Create table figure
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.head().columns),
                        fill_color=header_color,
                        height=header_height,
                        font=dict(size=16, color='blue'),
                        align='left'),
        cells=dict(values=[df.head()[col] for col in df.head().columns],
                fill_color=row_colors,
                height=row_height,
                font=dict(size=14, color='black'),
                align='left'))
        ])

        fig.write_image(f"{img_folder}/{img_name}",width=img_width)
        display(Image(f"{img_folder}/{img_name}"))

def mae(preds, targets):
    return np.abs(preds - targets).mean()

def smape(preds, targets):
    return (100 * (2 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets)))).mean()

def plot_train_predictions_with_mean_sigma(preds_all_trials, train_data, val_data, train_len, val_len, seq_length, tavg_index, model, sigma_mult=1, titlestr="",box_x=0.6, box_y=0.23):

    # Training predictions aggregation
    train_preds_mean = np.mean(preds_all_trials, axis=0)
    train_preds_std = np.std(preds_all_trials, axis=0)
    train_targets = train_data[seq_length:train_len,0]

    # print(f'train_preds_mean shape: {train_preds_mean.shape}, train_targets shape: {train_targets.shape}')

    train_nrmse_range = nrmse(train_preds_mean, train_targets, method="range")
    # print(f"Training NRMSE (range): {train_nrmse_range:.4f}")

    train_psd_nrmse_range = psd_nrmse(train_preds_mean,train_targets, fs=1.0, method="range", per_series=False)
    # print(f"Training PSD NRMSE (range): {train_psd_nrmse_range:.4f}")

    train_nrmse_std = nrmse(train_preds_mean, train_targets, method="std")
    # print(f"Training NRMSE (std): {train_nrmse_std:.4f}")

    train_psd_nrmse_std = psd_nrmse(train_preds_mean,train_targets, fs=1.0, method="std", per_series=False)
    # print(f"Training PSD NRMSE (std): {train_psd_nrmse_std:.4f}")

    # Mean Absolute Error (MAE)
    train_mae = mae(train_preds_mean, train_targets)

    # Symmetric Mean Absolute Percentage Error (sMAPE)
    train_smape = smape(train_preds_mean, train_targets)   

    # Plot training predictions with mean and std
    fig,ax = plt.subplots(figsize=(10,5))

    ax.plot(tavg_index[seq_length:train_len], train_data[seq_length:train_len,0], label='Training Data', color='orange')    
    ax.plot(tavg_index[seq_length:train_len], train_preds_mean, label='Predictions', 
            color='green', linestyle='--',linewidth=1.5,alpha=0.8)

    # plot std as grey area around the mean
    ax.fill_between(tavg_index[seq_length:train_len],
                    train_preds_mean - sigma_mult*train_preds_std,
                    train_preds_mean + sigma_mult*train_preds_std,
                    color='gray', alpha=0.2, label='Predictions ±1 STD')     

    ax.set_title(titlestr + ' - Predictions vs Training Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.grid()
    ax.legend()

    # Add as a text box the validation metrics
    textstr = '\n'.join((
        f'Model: {model} ',
        f'NRMSE (range): {train_nrmse_range:.4f}',
        f'PSD NRMSE (range): {train_psd_nrmse_range:.4f}',
        f'NRMSE (std): {train_nrmse_std:.4f}',
        f'PSD NRMSE (std): {train_psd_nrmse_std:.4f}',
        f'MAE: {train_mae:.4f}',
        f'sMAPE: {train_smape:.4f}',
    ))
    plt.gca().text(box_x, box_y, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()

    return train_nrmse_range, train_psd_nrmse_range, train_nrmse_std, train_psd_nrmse_std, train_mae, train_smape


def plot_val_predictions_with_mean_sigma(preds_all_trials, val_data, train_len, val_len, tavg_index, model, sigma_mult=1, titlestr="", box_x=0.6, box_y=0.23):
    """
    Plot validation predictions with mean and std shading.
    
    Parameters:
    preds_all_trials (list of np.array): List of prediction arrays from different trials.
    val_data (np.array): Actual validation data.
    val_len (int): Length of the validation data.
    extended_combined_tavg_1960_to_2060_df (pd.DataFrame): DataFrame containing the time index.
    model: The trained model object with attribute 'rnn'.
    """
    val_preds_all_trials = preds_all_trials
    # Validation predictions aggregation
    val_preds_mean = np.mean(np.array(val_preds_all_trials), axis=0)
    val_preds_std = np.std(np.array(val_preds_all_trials), axis=0)

    # Inverse scale the validation predictions
    # val_lstm_preds_descaled = scaler.inverse_transform(val_preds)

    val_preds_np = np.array(val_preds_mean[:val_len])
    val_data_np = val_data[:val_len,0]

    # print(len(val_lstm_preds_np),len(val_data_np))

    val_nrmse_range = nrmse(val_preds_np, val_data_np, method="range")
    # print(f"Validation NRMSE (range): {val_nrmse_range:.4f}")

    val_psd_nrmse_range = psd_nrmse(val_preds_np,val_data_np, fs=1.0, method="range", per_series=False)
    # print(f"Validation PSD NRMSE (range): {val_psd_nrmse_range:.4f}")

    val_nrmse_std = nrmse(val_preds_np, val_data_np, method="std")
    # print(f"Validation NRMSE (std): {val_nrmse_std:.4f}")
    val_psd_nrmse_std = psd_nrmse(val_preds_np,val_data_np, fs=1.0, method="std", per_series=False)
    # print(f"Validation PSD NRMSE (std): {val_psd_nrmse_std:.4f}")

    # Mean Absolute Error (MAE)
    val_mae = mae(val_preds_np,val_data_np)
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    val_smape = smape(val_preds_np, val_data_np)    

    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(tavg_index[train_len:(train_len+val_len)], val_data[:,0], label='Validation Data', color='orange')    
    ax.plot(tavg_index[train_len:(train_len+val_len+40*12+1)], val_preds_mean, label='Predictions', 
            color='green', linestyle='--',linewidth=1.5,alpha=0.8)

    # plot std as grey area around the mean
    ax.fill_between(tavg_index[train_len:(train_len+val_len+40*12+1)],
                    val_preds_mean - val_preds_std,
                    val_preds_mean + val_preds_std,
                    color='gray', alpha=0.2, label='Predictions ±1 STD')

    # Add as a text box the validation metrics
    textstr = '\n'.join((
            f'Model: {model} ',
        f'NRMSE (range): {val_nrmse_range:.4f}',
        f'PSD NRMSE (range): {val_psd_nrmse_range:.4f}',
        f'NRMSE (std): {val_nrmse_std:.4f}',
        f'PSD NRMSE (std): {val_psd_nrmse_std:.4f}',
        f'MAE: {val_mae:.4f}',
        f'sMAPE: {val_smape:.4f}',
    ))
    plt.gca().text(box_x, box_y, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_title(titlestr + ' - Predictions vs Validation Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.grid()
    ax.legend()
    plt.show()

    return val_nrmse_range, val_psd_nrmse_range, val_nrmse_std, val_psd_nrmse_std, val_mae, val_smape


plot_train_predictions = plot_train_predictions_with_mean_sigma
plot_val_predictions = plot_val_predictions_with_mean_sigma


def box_plot_components_non_vectorized(data):
    ntrials, ntimes = data.shape

    median = np.zeros(ntimes)
    q1 = np.zeros(ntimes)
    q3 = np.zeros(ntimes)
    whiskers_low = np.zeros(ntimes)
    whiskers_high = np.zeros(ntimes)
    outliers_list = []

    for i in range(ntimes):
        col_data = data[:, i]
        q1[i] = np.percentile(col_data, 25)
        median[i] = np.percentile(col_data, 50)
        q3[i] = np.percentile(col_data, 75)
        iqr = q3[i] - q1[i]

        # Define whisker bounds
        whisker_low_bound = q1[i] - 1.5 * iqr
        whisker_high_bound = q3[i] + 1.5 * iqr

        # Find actual whisker endpoints: furthest points within bounds
        lower_data = col_data[col_data >= whisker_low_bound]
        upper_data = col_data[col_data <= whisker_high_bound]

        whiskers_low[i] = lower_data.min() if len(lower_data) > 0 else col_data.min()
        whiskers_high[i] = upper_data.max() if len(upper_data) > 0 else col_data.max()

        # Outliers: points outside whisker bounds
        outliers = col_data[(col_data < whisker_low_bound) | (col_data > whisker_high_bound)]

        outliers_list.append(list(outliers))

    return median, q1, q3, whiskers_low, whiskers_high, outliers_list, ntimes


def box_plot_components(data):
    ntrials, ntimes = data.shape
    
    # Fully vectorized percentiles (axis=0 computes along trials for each time)
    q1 = np.percentile(data, 25, axis=0)
    median = np.percentile(data, 50, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    
    # Vectorized whisker bounds
    whisker_low_bound = q1 - 1.5 * iqr
    whisker_high_bound = q3 + 1.5 * iqr
    
    # Vectorized whiskers using broadcasting and masking
    # Mask data outside bounds with nan, then take min/max
    data_for_low = np.where(data >= whisker_low_bound, data, np.nan)
    data_for_high = np.where(data <= whisker_high_bound, data, np.nan)
    
    whiskers_low = np.nanmin(data_for_low, axis=0)
    whiskers_high = np.nanmax(data_for_high, axis=0)
    
    # Handle cases where all values are outliers (use full range)
    whiskers_low = np.where(np.isnan(whiskers_low), np.min(data, axis=0), whiskers_low)
    whiskers_high = np.where(np.isnan(whiskers_high), np.max(data, axis=0), whiskers_high)
    
    # Outliers (still requires loop for per-column lists)
    outliers_list = [
        list(data[:, i][(data[:, i] < whisker_low_bound[i]) | (data[:, i] > whisker_high_bound[i])])
        for i in range(ntimes)
    ]
    
    return median, q1, q3, whiskers_low, whiskers_high, outliers_list, ntimes



def plot_predictions_with_median_iqr(preds, true_data, data_len, tavg_index, model, titlestr="", marker=None,
                                             q1=5, q3=95, y_lims=(None, None), box_x=0.6, box_y=0.23, show_plot=True):
    """
    Plot validation predictions with median and IQR shading.
    
    Parameters:
    predictions (list of np.array): List of prediction arrays from different trials.
    true_data (np.array): Actual validation data.
    data_len (int): Length of the validation data.
    extended_combined_tavg_1960_to_2060_df (pd.DataFrame): DataFrame containing the time index.
    model: The trained model object with attribute 'rnn'.
    """

    # Validation predictions aggregation    
    # preds_median = np.median(np.array(preds), axis=0)
    # preds_q1 = np.percentile(np.array(preds), q1, axis=0)
    # preds_q3 = np.percentile(np.array(preds), q3, axis=0)
    # preds_q0 = np.percentile(np.array(preds), 0, axis=0)
    # preds_q4 = np.percentile(np.array(preds), 100, axis=0)

    preds_median, preds_q1, preds_q3, preds_min, preds_max, outliers_list, ntimes = box_plot_components(np.array(preds))
    
    preds_iqr = preds_q3 - preds_q1
    # calculate preds_min as the max between preds_q1 - 1.5*IQR and min prediction value at each time step
    # preds_min = np.maximum(preds_q1 - 1.5 * preds_iqr, preds_q0)
    # preds_min = preds_q1 - 1.5 * preds_iqr
    # calculate preds_max as the min between preds_q3 + 1.5*IQR and max prediction value at each time step
    # preds_max = np.minimum(preds_q3 + 1.5 * preds_iqr, preds_q4)
    # preds_max = preds_q3 + 1.5 * preds_iqr
    preds_min_max = preds_max - preds_min
    preds_iqr_area = preds_iqr.sum()
    preds_min_max_area = preds_min_max.sum()

    # Inverse scale the validation predictions
    # preds = scaler.inverse_transform(preds)

    preds_np = np.array(preds_median[:data_len])
    true_data_np = true_data[:data_len,0]   

    # print(len(val_lstm_preds_np),len(val_data_np))

    nrmse_range = nrmse(preds_np, true_data_np, method="range")
    # print(f"Validation NRMSE (range): {val_nrmse_range:.4f}")

    psd_nrmse_range = psd_nrmse(preds_np,true_data_np, fs=1.0, method="range", per_series=False)
    # print(f"Validation PSD NRMSE (range): {val_psd_nrmse_range:.4f}")

    nrmse_std = nrmse(preds_np, true_data_np, method="std")
    # print(f"Validation NRMSE (std): {nrmse_std:.4f}")

    psd_nrmse_std = psd_nrmse(preds_np,true_data_np, fs=1.0, method="std", per_series=False)
    # print(f"Validation PSD NRMSE (std): {psd_nrmse_std:.4f}")

    # Mean Absolute Error (MAE)
    mae_ = mae(preds_np, true_data_np)
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    smape_ = smape(preds_np, true_data_np)    

    if show_plot:
        fig,ax = plt.subplots(figsize=(10,5))
        labels = ax.plot(tavg_index[:data_len], true_data[:data_len,0], label='Targets', color='orange')
        outliers_count = 0
        for single_pred in preds:
            if single_pred.sum() > preds_max.sum() or single_pred.sum() < preds_min.sum():  # Skip plotting if the mean prediction is unreasonably high            
                    # ax.plot(tavg_index, single_pred, label='', 
                    #         linestyle=':',linewidth=0.3,alpha=0.1)
                    outliers_count += 1

        median_predictions = ax.plot(tavg_index, preds_median, label='Predictions Median',
                    color='red', linestyle='--',linewidth=1.5,alpha=0.8)


        # plot interquartile range as grey area around the median
        iqr = ax.fill_between(tavg_index,
                        preds_q1,
                        preds_q3,
                        color='gray', alpha=0.2, label='25"%"-75"%" IQR')
        
        # plot min-max whiskers as dashed lines
        min_max = ax.plot(tavg_index, preds_min, color='cyan', linestyle='-', linewidth=0.8, alpha=0.8, label='Min-Max Whiskers')
        ax.plot(tavg_index, preds_max, color='cyan', linestyle='-', linewidth=0.8, alpha=0.8)   
        
        # Add as a text box the validation metrics
        textstr = '\n'.join((
                f'Model: {model} ',
            f'NRMSE (range): {nrmse_range:.4f}',
            f'PSD NRMSE (range): {psd_nrmse_range:.4f}',
            f'NRMSE (std): {nrmse_std:.4f}',
            f'PSD NRMSE (std): {psd_nrmse_std:.4f}',
            f'Outliers: {outliers_count} (out of {len(preds)} trials)',
            f'MAE: {mae_:.4f} - sMAPE: {smape_:.4f}',
            f'IQR area: {preds_iqr_area:.4f}',
            f'Min-Max area: {preds_min_max_area:.4f}'
        ))
        plt.gca().text(box_x, box_y, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_title(titlestr)
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature Anomaly (°C)')
        if y_lims != (None, None):
            ax.set_ylim(y_lims)
        ax.grid()

        # include in the legend only labels with handles
        handles = [labels[0], median_predictions[0], iqr, min_max[0]]
        labels = ['Targets', 'Predictions Median', 'Predictions IQR', 'Min-Max Whiskers']    
        ax.legend(handles, labels)

        if marker is not None:
            ax.axvline(x=marker, color='green', linestyle='--', linewidth=1.0, label='Marker')
            ax.legend()          

        plt.show()

    return nrmse_range, psd_nrmse_range, nrmse_std, psd_nrmse_std, mae_, smape_, preds_iqr_area, preds_min_max_area, fig, ax

