#-*-coding:GBK -*- 
import numpy as np
import pandas as pd
import math


import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


th_dict = {"Age": 64.5000,
           "Sex": 1.0000,
           "Blood_Oxygen": 53.5000,
           "Temperature": 37.9000,
           "D-dimer": 1.1050,
           "White_blood_cell_count": 5.8050,
           "Neutrophil_count": 0.7700,
           "C-Reactive_protein": 3.3800,
           "Lymphocyte_count": 4.2850,
           "Lactic_dehydrogenase": 227.0000,

           'Cardiovascular_disease': 1,
           'Hypertension': 1,
           'Diabetes': 1,
           'Cerebrovascular_Disease': 1,

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

def main(data_df):

    for key in th_dict.keys():
        if not key.find("hu") >0:
            data_df[key] = data_df[key].fillna(0)
        data_df[key] = data_df[key].map(lambda input:1 if input>=th_dict[key] else 0 )


    add_DF = pd.DataFrame()
    add_DF["V-HU"]=data_df['HU_of_consolidation']+data_df['Volume_of_total_pneumonia_infection'] #0,1,2

    all_data = pd.concat([
                        data_df["Duration"],
                        data_df["Death"] ,
                        add_DF["V-HU"],
                        ],axis=1)


    kmf = KaplanMeierFitter()
    T = all_data["Duration"]

    death = all_data['Death']
    key_word = "V-HU"

    risk_level_0 = all_data[key_word] == 0
    risk_level_1 = all_data[key_word] == 1
    risk_level_2 = all_data[key_word] == 2

    kmf.fit(T[risk_level_0], event_observed=death[risk_level_0],  label='low risk')
    ax = kmf.plot()
    kmf.fit(T[risk_level_1], event_observed=death[risk_level_1],  label='intermediate risk')
    ax = kmf.plot()

    kmf.fit(T[risk_level_2], event_observed=death[risk_level_2], label='high risk')
    kmf.plot(ax=ax)
    plt.legend(fontsize=7,loc='lower left')
    #kmf.plot()
    plt.ylabel('Survival Probability')
    plt.xlabel('Time since admission to death(days)')

    plt.text(37, 1, "Hazard ratio:",fontsize=8,style='italic')
    plt.text(37, 0.96, "low risk: reference",fontsize=8)
    plt.text(37, 0.92, "intermediate risk: 2·54; 95%CI, 1·44-4·49",fontsize=8)
    plt.text(37, 0.88, "high risk: 4·90; 95%CI, 2·78-8·64",fontsize=8)
    plt.text(12, 1, "p-value < 0·0001",fontsize=8,fontstyle='italic')

    #all data
    low_list = ['69','61','56','53','53','53','53','53']
    medium_list = ['100','69','58','55','53','53','53','53']
    high_list = ['69','37','27','26','21','21','20','20']

    plt.text(-30, 0.005,  "Numbers at low risk",fontsize=8)
    for i in range(len(low_list)):
        plt.text((i*10)-1, 0,low_list[i],fontsize=8)
    plt.text(-30, -0.035, "Numbers at intermediate risk",fontsize=8)
    for i in range(len(low_list)):
        plt.text((i*10)-1, -0.04,medium_list[i],fontsize=8)
    plt.text(-30, -0.075, "Numbers at high risk",fontsize=8)
    for i in range(len(low_list)):
        if len(high_list[i])==1:
            plt.text((i*10), -0.08,high_list[i],fontsize=8)
        else:
            plt.text((i*10)-1, -0.08,high_list[i],fontsize=8)

    plt.savefig("km_alldata_V-HU.pdf", bbox_inches='tight')

if __name__ == '__main__':
    data_path = "./data_after_impute.csv"
    data_df = pd.read_csv(data_path, encoding="gbk")

    main(data_df)