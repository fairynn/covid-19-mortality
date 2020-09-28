#-*-coding:GBK -*- 
import numpy as np
import pandas as pd


from lifelines import CoxPHFitter


th_dict = { "Age":64.5000,
            "Sex":1.0000,
            "Blood_Oxygen":53.5000,
            "Temperature":37.9000,
            "D-dimer":1.1050,
            "White_blood_cell_count":5.8050,
            "Neutrophil_count":0.7700,
            "C-Reactive_protein":3.3800,
            "Lymphocyte_count":4.2850,
            "Lactic_dehydrogenase":227.0000,

            'Cardiovascular_disease':1,
            'Hypertension':1,
            'Diabetes':1,
            'Cerebrovascular_Disease':1,            
            
            "Volume_of_total_pneumonia_infection":437706.0391,
            "HU_of_total_pneumonia_infection":-501.1223,
            "Ratio_of_total_pneumonia_infection":0.1594,
            "Volume_of_GGO":219618.2280,
            "HU_of_GGO":-486.0722,
            "Ratio_of_GGO":0.0805,
            "Volume_of_consolidation":61083.4643,
            "HU_of_consolidation":-54.8057,
            "Ratio_of_consolidation":0.0214,
            "Volume_of_pleural_effusion":0.0000,
            "HU_of_pleural_effusion":0.0000,
            "Ratio_of_pleural_effusion":0.0000,

}

def main(data_df):

    for key in th_dict.keys():
        if not key.find("HU") >0:
            data_df[key] = data_df[key].fillna(0)
        data_df[key] = data_df[key].map(lambda input:1 if input>=th_dict[key] else 0 )


    add_DF = pd.DataFrame()
    add_DF["V-HU"]=data_df['HU_of_consolidation']+data_df['Volume_of_total_pneumonia_infection'] #0,1,2

    combinations_df = pd.concat([
                                data_df["Duration"],
                                data_df["Death"] ,
                                data_df["Age"],
                                data_df["Blood_Oxygen"],
                                data_df["C-Reactive_protein"] ,
                                #data_df["White_blood_cell_count"] ,
                                data_df["Lymphocyte_count"],
                                data_df["Cerebrovascular_Disease"] ,
                                data_df["Sex"],
                                #data_df["Neutrophil_count"],
                                #data_df["D-dimer"] ,
                                data_df["Lactic_dehydrogenase"],

                                add_DF["V-HU"],

                                ],axis=1)


    cph = CoxPHFitter()
    cph.fit(combinations_df,"Duration",event_col = "Death",step_size = 0.01)

    cph.print_summary()

if __name__ == '__main__':
    data_path = "./data_after_impute.csv"
    data_df = pd.read_csv(data_path, encoding="gbk")

    main(data_df)