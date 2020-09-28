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
                        data_df["Age"],
                        add_DF["V-HU"],
                        ],axis=1)


    age_th = 1
    age_group_young=all_data[all_data["Age"]<age_th]
    age_group_old=all_data[all_data["Age"]>=age_th]

    kmf = KaplanMeierFitter()
    T = age_group_old["Duration"]

    death = age_group_old['Death']
    key_word = "V-HU"

    risk_level_0 = age_group_old[key_word] == 0
    risk_level_1 = age_group_old[key_word] == 1
    risk_level_2 = age_group_old[key_word] == 2

    plt.figure(figsize=(9,12))
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=2,rect=(0,0,1,1))

    kmf.fit(T[risk_level_0], event_observed=death[risk_level_0],  label='low risk')
    ax = kmf.plot()
    kmf.fit(T[risk_level_1], event_observed=death[risk_level_1],  label='intermediate risk')
    ax = kmf.plot()

    kmf.fit(T[risk_level_2], event_observed=death[risk_level_2], label='high risk')
    kmf.plot(ax=ax)
    #kmf.plot()
    plt.ylabel('Survival Probability')
    plt.xlabel('Time since admission to death(days)')
    plt.legend(fontsize=8,loc='lower left')

    plt.text(32, 1, "Hazard ratio:",fontsize=9,style='italic')
    plt.text(32, 0.96, "low risk: reference",fontsize=9)
    plt.text(32, 0.92, "intermediate risk: 2·59; 95%CI, 1·20-5·56",fontsize=9)
    plt.text(32, 0.88, "high risk: 3·56; 95%CI, 1·64-7·71",fontsize=9)
    plt.text(11, 1, "p-value = 0·0004",fontsize=9,style='italic')

    #old group
    low_list = ['22','18','15','14','14','14','14','14']
    medium_list = ['53','32','21','18','17','16','16','15']
    high_list = ['44','21','14','13','9','9','9','9']
    plt.text(-30, -0.1,  "Numbers at low risk",fontsize=10)
    for i in range(len(low_list)):
        plt.text((i*10)-1, -0.1,low_list[i],fontsize=9)
    plt.text(-30, -0.14, "Numbers at intermediate risk",fontsize=9)
    for i in range(len(low_list)):
        plt.text((i*10)-1, -0.14,medium_list[i],fontsize=9)
    plt.text(-30, -0.18, "Numbers at high risk",fontsize=9)
    for i in range(len(low_list)):
        if len(high_list[i])==1:
            plt.text((i*10), -0.18,high_list[i],fontsize=9)
        else:
            plt.text((i*10)-1, -0.18,high_list[i],fontsize=9)

    plt.text(20, -0.25, "(a) >=65 years group",fontsize=9)

    #plt.savefig("km_age_group_old_lesion_vol_and_solid_hu_0908.pdf", bbox_inches='tight')

    T = age_group_young["Duration"]
    death = age_group_young['Death']
    key_word = "V-HU"

    risk_level_0 = age_group_young[key_word] == 0
    risk_level_1 = age_group_young[key_word] == 1
    risk_level_2 = age_group_young[key_word] == 2

    plt.subplot(2, 1, 2)
    plt.tight_layout(pad=2,rect=(0,0,1,1))

    kmf.fit(T[risk_level_0], event_observed=death[risk_level_0],  label='low risk')
    ax = kmf.plot()
    kmf.fit(T[risk_level_1], event_observed=death[risk_level_1],  label='intermediate risk')
    ax = kmf.plot()

    kmf.fit(T[risk_level_2], event_observed=death[risk_level_2], label='high risk')
    kmf.plot(ax=ax)

    #kmf.plot()
    plt.ylabel('Survival Probability')
    plt.xlabel('Time since admission to death(days)')
    plt.legend(fontsize=8,loc='lower left')

    plt.text(32, 1, "Hazard ratio:",fontsize=9,style='italic')
    plt.text(32, 0.965, "low risk: reference",fontsize=9)
    plt.text(32, 0.93, "intermediate risk: 1·42; 95%CI, 0·56-3·61",fontsize=9)
    plt.text(32, 0.895, "high risk: 4·60; 95%CI, 1·92-10·99",fontsize=9)
    plt.text(11, 1, "p-value = 0·0026",fontsize=9,style='italic')

    #young group
    low_list = ['47','43','41','39','39','39','39','39']
    medium_list = ['47','37','37','37','37','37','37','37']
    high_list = ['25','16','13','13','12','12','11','11']
    plt.text(-30, 0.075,  "Numbers at low risk",fontsize=9)
    for i in range(len(low_list)):
        plt.text((i*10)-1, 0.075,low_list[i],fontsize=9)
    plt.text(-30, 0.035, "Numbers at intermediate risk",fontsize=9)
    for i in range(len(low_list)):
        plt.text((i*10)-1, 0.035,medium_list[i],fontsize=9)
    plt.text(-30, -0.005, "Numbers at high risk",fontsize=9)
    for i in range(len(low_list)):
        if len(high_list[i])==1:
            plt.text((i*10), -0.005,high_list[i],fontsize=9)
        else:
            plt.text((i*10)-1, -0.005,high_list[i],fontsize=9)

    plt.text(20, -0.055, "(b) <65 years group",fontsize=10)

    plt.savefig("km_old_and_young_V-HU.pdf", bbox_inches='tight')


if __name__ == '__main__':
    data_path = "./data_after_impute.csv"
    data_df = pd.read_csv(data_path, encoding="gbk")

    main(data_df)