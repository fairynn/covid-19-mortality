#-*-coding:GBK -*- 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


th_dict = {"Age": 64.5000,
           "Sex": 0,
           "Blood_Oxygen": 53.5000,
           "Temperature": 37.9000,
           "D-dimer": 1.1050,
           "White_blood_cell_count": 5.8050,
           "Neutrophil_count": 0.7700,
           "C-Reactive_protein": 3.3800,
           "Lymphocyte_count": 4.2850,
           "Lactic_dehydrogenase": 227.0000,

           'Cardiovascular_disease': 0,
           'Hypertension': 0,
           'Diabetes': 0,
           'Cerebrovascular_Disease': 0,

           "Volume_of_total_pneumonia_infection": 437706.0391,
           "HU_of_total_pneumonia_infection": -501.1223,
           "Ratio_of_total_pneumonia_infection": 0.1594,
           "Volume_of_GGO": 219618.2280,
           "HU_of_GGO": -486.0722,
           "Ratio_of_GGO": 0.0805,
           "Volume_of_consolidation": 61083.4643,
           "HU_of_consolidation": -54.8057,
           "Ratio_of_consolidation": 0.0214,
           "Volume_of_pleural_effusion": 0.0000,
           "HU_of_pleural_effusion": 0.0000,
           "Ratio_of_pleural_effusion": 0.0000,

           }


def plot_confusion_matrix(cm, labels, title='Correlation Coefficient', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


def data_corr(data_df):

    for key in th_dict.keys():
        if key.find("HU") >0:
            data_df[key] = data_df[key].fillna(-10000000)
        if key.find("Volume_of_pleural_effusion") >0 or key.find("Ratio_of_pleural_effusion") >0:
            data_df[key] = data_df[key].fillna(0)
        else:
            data_df[key] = data_df[key].fillna(0)
        data_df[key] = data_df[key].map(lambda input:1 if input>th_dict[key] else 0 )


    add_DF = pd.DataFrame()
    add_DF["V-HU"]=data_df['HU_of_consolidation']+data_df['Volume_of_total_pneumonia_infection'] #0,1,2


    all_data = pd.concat([
                           data_df["Age"],
                           data_df["Sex"],
                           data_df["Blood_Oxygen"],
                           data_df["Temperature"],
                           data_df["D-dimer"],
                           data_df["White_blood_cell_count"],
                           data_df["Neutrophil_count"],
                           data_df["C-Reactive_protein"],
                           data_df["Lymphocyte_count"],
                           data_df["Lactic_dehydrogenase"],

                           data_df['Cardiovascular_disease'],
                           data_df['Hypertension'],
                           data_df['Diabetes'],
                           data_df['Cerebrovascular_Disease'],

                           data_df["Volume_of_total_pneumonia_infection"],
                           data_df["HU_of_total_pneumonia_infection"],
                           data_df["Ratio_of_total_pneumonia_infection"],
                           data_df["Volume_of_GGO"],
                           data_df["HU_of_GGO"],
                           data_df["Ratio_of_GGO"],
                           data_df["Volume_of_consolidation"],
                           data_df["HU_of_consolidation"],
                           data_df["Ratio_of_consolidation"],
                           data_df["Volume_of_pleural_effusion"],
                           data_df["HU_of_pleural_effusion"],
                           data_df["Ratio_of_pleural_effusion"],

                           add_DF["V-HU"],

                          ],axis=1)

    corr = all_data.corr()
    return corr

def main(data_df):
    corr = data_corr(data_df)

    labels = [
              "Age","Sex","Temperature","Blood oxygen","C-Reactive protein",
              "lymphocyte count","D-dimer","Lactic dehydrogenase",
              "White blood cell count","Neutrophil count","Cardiovascular disease",
              "Hypertension","Cerebrovascular disease","Diabetes",
              "Volume of total peumonia infection","Volume of GGO","Volume of consolidation","Volume of pleural effusion",
              "Ratio of total peumonia infection","Ratio of GGO","Ratio of consolidation","Ratio of pleural effusion",
              "HU of total peumonia infection","HU of GGO","HU of consolidation","HU of Pleural effusion",

              "V-HU",
            ]


    cm_normalized = np.array(corr)
    np.set_printoptions(precision=3)
    plt.figure(figsize=(16, 12), dpi=300)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=7, va='center', ha='center')

    # offset the tick
    tick_marks = np.array(range(len(labels))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, labels,title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('alldata_imputed_bin_corr_cm2_nn.svg', format='svg',dpi=300)


if __name__ == '__main__':
    data_path = "./data_after_impute.csv"
    data_df = pd.read_csv(data_path, encoding="gbk")

    main(data_df)