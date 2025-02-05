import os
from pathlib import Path
import requests


def print_folder_structure(folder, indent=0, max_file_count=100):
    """Recursively print the folder structure."""
    file_count = 0
    flag = False
    for entry in Path(folder).iterdir():
        if entry.is_file:
            file_count += 1

        if not entry.is_file or file_count <= max_file_count:
            print(" " * indent + "|-- " + entry.name)
        else:
            flag = True
            
        if entry.is_dir():
            print_folder_structure(entry, indent + 4, max_file_count)

    if flag:
        print(" " * indent + "|-- " + '...')


def download_folder_from_github(github_url, target_folder):
    """
    Downloads a folder and its contents from a GitHub repository using the GitHub API.

    Parameters:
        github_url (str): The GitHub URL of the folder (e.g., https://github.com/user/repo/tree/main/folder).
        target_folder (str): The local folder where the downloaded files should be saved.

    Returns:
        None

    Raises:
        ValueError: If the URL is invalid or cannot be processed.
    """
    if  Path(target_folder).exists:
        print(f'{target_folder} exists, please delete and try again.')
        return
    
    try:
        # Parse the GitHub URL to extract owner, repo, branch, and folder path
        if "/tree/" not in github_url:
            raise ValueError("The URL does not point to a folder in a GitHub repository.")

        base_url, branch_path = github_url.split("/tree/")
        owner_repo = base_url.split("github.com/")[1]
        branch, folder_path = branch_path.split("/", 1)

        # GitHub API URL to get folder contents
        api_url = f"https://api.github.com/repos/{owner_repo}/contents/{folder_path}?ref={branch}"

        # Fetch folder contents
        response = requests.get(api_url)
        response.raise_for_status()

        items = response.json()

        # Create target folder if it does not exist
        os.makedirs(target_folder, exist_ok=True)

        for item in items:
            item_name = item['name']
            item_path = os.path.join(target_folder, item_name)

            if item['type'] == 'file':
                # Download and save file
                file_response = requests.get(item['download_url'])
                file_response.raise_for_status()
                with open(item_path, 'wb') as f:
                    f.write(file_response.content)
            elif item['type'] == 'dir':
                # Recursively download folder
                download_folder_from_github(f"{github_url}/{item_name}", item_path)

        print(f"Folder successfully downloaded to: {target_folder}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def plot_group_by_dynamic(index, every, offset=None, period=None, start_by='window'):
    import polars as pl
    from matplotlib import pyplot as plt
    
    df = pl.DataFrame(dict(index=index))
    g = df.group_by_dynamic('index', every=every, offset=offset, period=period, start_by=start_by, include_boundaries=True)
    df2 = g.agg(pl.col('index').alias('index_in_group'))
    
    fig, ax = plt.subplots(figsize=(12, 0.6))
    ax.vlines(df['index'], 0, 1, lw=3, color='#0099FF')
    
    for v in df['index']:
        ax.text(v + 0.1, 0.1, str(v), color='#0099FF')
    
    for s, e in df2.select(pl.col('_lower_boundary') - 0.05, pl.col('_upper_boundary') - 0.2).rows():
        ax.axvspan(s, e, color='green', alpha=0.2)
    
    ax.vlines(df2['index'], 0, 1, color='red', ls='dashed')
    
    for v in df2['index']:
        ax.text(v + 0.1, 0.5, str(v), color='red')
    
    ax.axis('off')
    print(df2['index_in_group'].to_list())
    fig.patch.set_alpha(0.2)
    return fig        
