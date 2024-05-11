from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
from PIL import Image


image_arrays = []
main_directory_path = r'C:\Users\dhanv\Desktop\pythonprac\handata\leapGestRecog\leapGestRecog'

for folder_name in os.listdir(main_directory_path):
    main_folder_path = os.path.join(main_directory_path, folder_name)

   
    if os.path.isdir(main_folder_path):
        for subfolder_name in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder_name)
        
           
            if os.path.isdir(subfolder_path):
                
                for filename in os.listdir(subfolder_path):

                    image_path = os.path.join(subfolder_path, filename)
                    
                    
                    if filename.endswith('.png'):
                        try:
                            
                            img = Image.open(image_path).convert('L')  

                            img_array = np.array(img)                            
                            flattened_array = img_array.flatten()    
                            image_arrays.append(flattened_array) 
                        except Exception as e: 
                            print(f"Error processing image: {filename}, Exception: {e}")


df = pd.DataFrame(image_arrays)




def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()



def plotCorrelationMatrix(df, graphWidth):
  
    df = df.dropna(axis=0)
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)  
    plt.show()


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])

    df = df.dropna(axis='columns')  

    if df.shape[1] < 2:
        print("No scatter plots shown: The number of remaining columns is less than 2.")
        return

    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                          va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


    


plotPerColumnDistribution(df, nGraphShown=10, nGraphPerRow=4)
plotCorrelationMatrix(df, graphWidth=10)
plotScatterMatrix(df, plotSize=20, textSize=10)