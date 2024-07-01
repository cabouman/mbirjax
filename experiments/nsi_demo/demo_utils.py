import os, sys
import numpy as np
import urllib.request
import tarfile
import warnings

def download_and_extract_tar(download_url, save_dir):
    """ Given a download url, download the tarball file from ``download_url`` , extract the tarball to ``save_dir``, and return the paths to the tarball file as well as the extracted file. 
        If the file already exists in ``save_dir``, user will be queried whether it is desired to download and overwrite the existing files.
        ``download_url`` is assumed to have the format <url/{tarball_name}>.
        The tarball file is assumed to contain a single top-level directory.
 
    Args:
        download_url: An url to download the data. This url needs to be public.
        save_dir (string): Path to parent directory where downloaded file will be saved and extracted to. 
    Return:
        A tuple containing:
            - path to the tarball file. This will be ``save_dir``+ downloaded_file_name.
            - A list containing the names of the top level files 
    """

    is_download = True
    # the download url is assumed to have the format "**/{tarball_name}"
    tarball_name = download_url.split('/')[-1]
    # full path to the tarball file 
    tarball_path = os.path.join(save_dir, tarball_name)
    
    # If the tarball already exists, then prompt user whether to download and overwrite the existing file.
    if os.path.exists(tarball_path):
        is_download = query_yes_no(f"{tarball_path} already exists. Do you still want to download and overwrite the file?")
    
    ################### Download and extract tarball file
    if is_download:
        # make the directory where the tarball will be saved, if necessary.
        os.makedirs(os.path.dirname(tarball_path), exist_ok=True)
        ###### download the tarball
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, tarball_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
            elif e.code == 403:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
            elif e.code == 404:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
            else:
                raise RuntimeError(
                    f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        except urllib.error.URLError as e:
            raise RuntimeError('URLError raised! Please check your internet connection.')
        
        # download is successful if no exceptions occur
        print(f"Download successful! Tarball file saved to {tarball_path}")
        
        ###### Extract to save_dir.
        print(f"Extracting tarball file to {save_dir} ...")
        try:
            tar_file = tarfile.open(tarball_path)
            extracted_file_name = os.path.join(save_dir, os.path.commonprefix(tar_file.getnames()))
            tar_file.extractall(save_dir)
            tar_file.close
            print(f"Extraction successful! File extracted to {extracted_file_name}")
        except:
            warnings.warn(f"Extraction failed. Please make sure {tarball_path} is a tarball file.")
            return tarball_path


    ################### Skip download and extraction steps
    else:
        print("Skipped data download and extraction step.")
        # Get top level file names without extracting the tarball
        tar_file =  tarfile.open(tarball_path, mode='r')
        extracted_file_name = os.path.join(save_dir, os.path.commonprefix(tar_file.getnames()))
    
    return tar_file, extracted_file_name


def query_yes_no(question, default="n"):
    """Ask a yes/no question via input() and return the answer.
        Code modified from reference: `https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input/3041990`

    Args:
        question (string): Question that is presented to the user.
    Returns:
        Boolean value: True for "yes" or "Enter", or False for "no".
    """

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = f" [y/n, default={default}] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    return
