import pandas as pd
from sqlalchemy import create_engine
import psycopg2

###### This is a script to obtain the data in our research on analytic ranks of modular curves 

###### We hope to test different machine learning/deep learning/reinforcement learning techniques to find the relationships between many other factors of modular curves and the analytic ranks of the modular curve ######

###### Each Selection Request may have to be very small, so we would like to write the following script to obtain all the data under the restriction ######


# I would like to divide it into several files so that we can stop and handle errors but perhaps we can restart from the middle without re-obtain the previous data and keep each file in a size that we can open without difficulties
def fileNameNumering(num):
    return "_DataForModularCurves2/lmfdbDataRanks" + str(num) + ".csv"


# I need to give the select sql based on id we are focusing on in the following for-loop
def sqlNumbering(num, fileSize):
    low = num * fileSize
    high = (num + 1) * fileSize - 1
    sql = "SELECT * FROM lmfdb.public.gps_gl2zhat_fine WHERE id BETWEEN " + str(low) + " AND " + str(high) + " ;" 
    return sql

def getData(fileSize, numFiles):
    # Build Connections to the Dataset


    for num in range(0, numFiles + 1):
        filePath = fileNameNumering(num)
        sql = sqlNumbering(num, fileSize)
        print("Write file "+str(num) + " : \n")
        try:
            connection = psycopg2.connect(database="lmfdb", user="lmfdb", password="lmfdb", host="devmirror.lmfdb.xyz",
                                          port=5432)
            cursor = connection.cursor()
            cursor.execute(sql)
            record = cursor.fetchall()
            # open and write rows into the file num-th file
            with open(filePath, 'w') as file:
                for row in record:
                    # this writting line can be improved if the style of the csv file is not good enough, or we can play with the download data locally on our own laptops my merging of add more structures, or input them to a postgresql database etc.
                    file.write(';'.join(map(str, row))+'\n') 
                    # close the num-th file
                file.close()
                cursor.close()
        except psycopg2.Error as e:
            print(f"An error occurred: {e}")
            break
    

    # stop the connection 


def main():
    fileSize = 10000 ### You can change it if you want a different size for each file
    numFiles = 1684 ### You can change it if you want more or less number of files
    getData(fileSize, numFiles) ### Download the data

if __name__ == "__main__":
    main()
        