# Author: shivam shakti 
# Date: 2017-12-06 23:29:39 
# Last Modified by:   shivam shakti 
# Last Modified time: 2017-12-06 23:29:39 

import os
import requests
import zipfile

class Download(object):
    '''
    Download dataset and store it
    '''

    def __init__(self, data_path, download_url):
        '''
        constructor
        
        Parameters
        ---------
        data_path: path to store data set
        download_url: url from which to download dataset
        '''

        self.DATA_PATH = data_path
        self.DOWNLOAD_URL = download_url

    def maybe_download(self):
        '''
        downloads the data from DOWNLOAD_URL if not already present in DATA_PATH
        '''
        
        FILE_PATH = os.path.join(self.DATA_PATH, 'ml-20m.zip')
        
        # check if data already exists
        if os.path.isfile(FILE_PATH):
            raise ValueError("file already exists")

        # start download
        print ("downloading file...")
        response = requests.get(self.DOWNLOAD_URL)
        
        # check download successful
        if response.status_code == 200:
            print ("download completed")
        else:
            raise IOError("download failed")

        # write file to data path
        FILE_PATH = os.path.join(self.DATA_PATH, 'ml-20m.zip')        
        with open(FILE_PATH, 'wb') as f:
            f.write(response.content)

        self.extract(FILE_PATH)
        print ("extracted and saved at ", self.DATA_PATH)


    def extract(self, file_path):
        '''
        extracts the downloaded file
        '''
        with zipfile.ZipFile(file_path, 'r') as zipped_file:
            zipped_file.extractall(self.DATA_PATH)



if __name__ == "__main__":
    DATA_PATH = "../../data"
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    download = Download(data_path=DATA_PATH, download_url=DOWNLOAD_URL)
    download.maybe_download()