import pandas as pd

def load_and_clean_data(file_path):
    """
    Load cricket match dataset and clean it from the specified path.
    """
    try:
        df = pd.read_csv(file_path)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Clean 'Runrate B' column by handling commas and '--'
        df['Runrate B'] = df['Runrate B'].astype(str).str.replace(',', '.', regex=False)
        df.loc[df['Runrate B'] == '--', 'Runrate B'] = '0'
        df['Runrate B'] = pd.to_numeric(df['Runrate B'])

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None