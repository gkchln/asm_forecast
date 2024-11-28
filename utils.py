import pandas as pd
from PIL import Image

hour_loss_days = [
    20170326,
    20180325,
    20190331,
    20200329,
    20210328,
    20220327,
    20230326,
    20240331
]
hour_gain_days = [
    20171029,
    20181028,
    20191027,
    20201025,
    20211031,
    20221030,
    20231029,
    20241027
]

def pretty_value_counts(df, column, **kwargs):
    output_df = pd.DataFrame.from_dict({
    'count': df[column].value_counts(**kwargs), 
    'pct': (df[column].value_counts(normalize=True, **kwargs)*100).map('{:.2f}%'.format),
    })
    return output_df

def get_timestamp_from_gme_system(input_df, date_col='Data', hour_col='Ora', add_timezone=False):
    """
    Converts date and hour columns from a dataframe into pandas Timestamps based on the GME (Gestore dei Mercati Energetici) datetime system.

    This function adjusts the hour column to represent the start of each hour interval, rather than the end. 
    It also handles specific adjustments for days where daylight saving time causes a loss or gain of an hour.

    Parameters:
    -----------
    input_df : pd.DataFrame
        The input dataframe containing at least the date and hour columns to be converted.
    date_col : str, optional (default='Data')
        The name of the column in input_df that contains the date information in YYYYMMDD format.
    hour_col : str, optional (default='Ora')
        The name of the column in input_df that contains the hour information.

    Returns:
    --------
    pd.Series
        A pandas Series of Timestamps corresponding to the adjusted date and hour values.

    Notes:
    ------
    - The output of the function defines the hour intervals by their start times. 
      For example, the interval from 02:00 to 03:00 is represented by 02:00.
    - Special handling is performed for days with daylight saving time changes:
        - On days with an hour loss (e.g., when clocks go forward), hours after 02:00 are adjusted forward by 1 hour.
        - On days with an hour gain (e.g., when clocks go back), hours after 02:00 are adjusted backward by 1 hour.
    """
    # Copy the input dataframe to avoid modifying the original
    df = input_df.copy()
    
    # Adjust the hour to refer to the start of the hour interval
    df[hour_col] = df[hour_col] - 1  # i-th hour is defined by (i-1):00
    
    # Handle days with hour loss (e.g., daylight saving time forward)
    hours_to_shift = df[date_col].isin(hour_loss_days) & (df[hour_col] > 1)
    df.loc[hours_to_shift, hour_col] = df.loc[hours_to_shift, hour_col] + 1
    
    # Handle days with hour gain (e.g., daylight saving time backward)
    hours_to_shift = df[date_col].isin(hour_gain_days) & (df[hour_col] > 2)
    df.loc[hours_to_shift, hour_col] = df.loc[hours_to_shift, hour_col] - 1
    
    timestamp = df[date_col].astype(str) + df[hour_col].apply(lambda x: "{:02d}".format(x))
    timestamp = pd.to_datetime(timestamp, format='%Y%m%d%H')

    if add_timezone:
        timestamp = timestamp.tz_convert('Europe/Rome', ambiguous='infer')
    
    return timestamp

def vertically_concatenate_images(image_paths, output_path):
    """
    Vertically concatenates a list of images and saves the result.

    Parameters:
        image_paths (list of str): List of file paths to the images to be concatenated.
        output_path (str): File path to save the concatenated image.

    Returns:
        None
    """
    # Open images
    images = [Image.open(img) for img in image_paths]

    # Calculate total height and max width
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # Create a blank image with the total height and max width
    combined_image = Image.new("RGBA", (max_width, total_height))

    # Paste each image below the previous one
    current_height = 0
    for img in images:
        combined_image.paste(img, (0, current_height))
        current_height += img.height

    # Save the result
    combined_image.save(output_path)

