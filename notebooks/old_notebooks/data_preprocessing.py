import os
import shutil
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '/Users/misha/Desktop/My_Work/Datasets/DIAC-WOZ/data/'
labels_dir = '/Users/misha/Desktop/My_Work/Datasets/DIAC-WOZ/labels/'

df_daic_train = pd.read_csv(labels_dir+'train_split.csv')
df_daic_test = pd.read_csv(labels_dir+'test_split.csv')
df_daic_dev = pd.read_csv(labels_dir+'dev_split.csv')

df_daic = df_daic_train
df_daic = df_daic.append(df_daic_test)
df_daic = df_daic.append(df_daic_dev)
df_daic = df_daic.reset_index()
df_daic = df_daic.drop(columns=['index'])
print(df_daic)

df_daic['Gender'] = df_daic['Gender'].apply(lambda x: x.strip())

counts = df_daic['Gender'].value_counts()

print(counts)

grouped_df = df_daic.groupby(['Gender', 'PCL-C (PTSD)']).size().reset_index(name='count')
grouped_df1 = df_daic.groupby(['Gender', 'PHQ_Binary']).size().reset_index(name='count')

# Print the resulting DataFrame
print(grouped_df)
print(grouped_df1)

import matplotlib.pyplot as plt

# Define the data
depressed_females = 39
depressed_males = 48
non_depressed_females = 66
non_depressed_males = 122

# Create the plot
fig, ax = plt.subplots()

# Plot the bars for each category
ax.bar(['depressed', 'non-depressed'], [depressed_females, non_depressed_females], color='pink', label='Female', width=0.2)
ax.bar(['depressed', 'non-depressed'], [depressed_males, non_depressed_males], color='blue', label='Male',width=0.2)

# Add a legend
ax.legend(loc='upper right')

# Add labels to the x and y axes
ax.set_xlabel('Depression Status')
ax.set_ylabel('Number of People')

# Show the plot
plt.show()


# plt.bar(df_daic['PCL-C (PTSD)'], df_daic['count'], color=df_daic['Gender'])
# plt.xlabel('Depression status')
# plt.ylabel('Count')
# plt.show()

# plt.bar(df_daic['Gender'], df_daic['PCL-C (PTSD)'])
# plt.xlabel('Gender')
# plt.ylabel('PCL-C (PTSD)')
# plt.show()

# sns.violinplot(x='Gender', y='PCL-C (PTSD)', data=df_daic)
# plt.ylabel('Depression status')
# plt.show()



# list_of_all_files = []
# for subdir, dirs, files in os.walk(data_dir):
#     for file in files:
#         list_of_all_files.append(os.path.join(subdir, file))
# # print(list_of_all_files)

# # 300_OpenFace2.1.0_Pose_gaze_AUs

# dst = '/Users/misha/Desktop/Datasets/DIAC-WOZ/DIAC_just_openface'
# openface_csv_files = []
# for file in list_of_all_files:
#     if 'OpenFace2' in file:
#         openface_csv_files.append(file)  
#         shutil.copy(file, dst)     
# print(openface_csv_files)



# # CNN_VGG_mat_files = []
# # for file in list_of_all_files:
# #     if '_CNN_VGG.mat' in file:
# #         CNN_VGG_mat_files.append(file)       
# # print(len(CNN_VGG_mat_files))

# # mat_contents = sio.loadmat(CNN_VGG_mat_files[0], matlab_compatible=False, struct_as_record=False)
# # feature = mat_contents['feature'][0][0]

# # ref_model_layers = feature.layers
# # for layer in ref_model_layers:
# #     print(layer[0][0].name)