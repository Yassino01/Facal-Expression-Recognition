from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

DATA_PATH = './data'

class Prepare_data():
  def __init__(self, datapath):

    self.data_path = datapath

  def split_train_val(self, train_filename = 'finaltrain', val_filename= 'val'):
    """
    Helper function to split the validation and test data from general train file as it contains (Train , Public test, Private test)
        params:-
            data_path = path to the folder that contains the train data file
    """
    csv_path = self.data_path +"/"+ 'train.csv'
    train = pd.read_csv(csv_path)
    
    train_data = pd.DataFrame(train.iloc[:18376,:])
    validation_data = pd.DataFrame(train.iloc[18376:,:])


    train_data.to_csv(self.data_path+"/"+train_filename+".csv")
    validation_data.to_csv(self.data_path+"/"+val_filename+".csv")
    print("Done splitting the train file into validation & final train file")

  def str_to_image(self, str_img = ' '):
    '''
    Convert string pixels from the csv file into image object
        params:- take an image string
        return :- return PIL image object
    '''
    imgarray_str = str_img.split(' ')
    imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
    return Image.fromarray(imgarray)

  def save_images(self, datatype='finaltrain'):
    '''
    save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
        params:-
        datatype= str e.g (finaltrain, val, finaltest)
    '''
    foldername= self.data_path+"/"+datatype
    csvfile_path= self.data_path+"/"+datatype+'.csv'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    data = pd.read_csv(csvfile_path)
    images = data['pixels'] #dataframe to series pandas
    numberofimages = images.shape[0]
    for index in tqdm(range(numberofimages)):
        img = self.str_to_image(images[index])
        img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
    print('Done saving {} data'.format((foldername)))

if __name__ == '__main__':

  generate_dataset = Prepare_data(DATA_PATH)
  generate_dataset.split_train_val()
  generate_dataset.save_images()
  generate_dataset.save_images('finaltest')
  generate_dataset.save_images('val')