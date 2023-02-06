# code used to rename the dataset enumerating all the images
import os
import uuid

def rename(Source_Path):
    files = os.listdir(Source_Path)
    for index, file in enumerate(files):
        os.rename(Source_Path + file,
                Source_Path + str(uuid.uuid4()).replace("-", "") + '.jpg')
        #print(Source_Path + str(uuid.uuid4()).replace("-","") + '.jpg')

    files = os.listdir(Source_Path)
    for index, file in enumerate(files):
        os.rename(Source_Path + file, Source_Path + str(index) + '.jpg')
    print("Finish")

rename('dataset/noah schnapp/')
rename('dataset/millie bobby brown/')
rename('dataset/finn wolfhard/')