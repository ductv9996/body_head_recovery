import csv
import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
PD = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ISO_20685_2010 = ['1.1', ' ', ' ', ' ', '1.2', '0.6', '1', '0.4', ' ', '1.1', '0.8', '1.4', '1.5']
TC133 = ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ']


def write_result(name, GT, PD):
    csv_dir = ROOT_DIR + '/data/outputs/' + name + '.csv'
    parts_name = ['neck', 'bust', 'upper_waist', 'under_waist',
                  'hip', 'thigh', 'calf', 'ankle', 'upper_arm', 'len_arm', 'len_shoulder', 'len_leg', 'height']
    GT = np.array(GT)
    PD = np.array(PD)
    Er = GT - PD
    with open(csv_dir, 'w+') as csvfile:
        fieldnames = ['Parts', 'Real(cm)', 'Predict(cm)', 'Error(cm)', 'Error%', 'ISO 20685:2010',
                      'ISO/TC133(garment, virtual human)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(13):
            writer.writerow({'Parts': parts_name[i], 'Real(cm)': GT[i] * 100,
                             'Predict(cm)': round(PD[i] * 100, 3), 'Error(cm)': round(abs(Er[i]), 3) * 100,
                             'Error%': round(abs(Er[i]) * 100 / GT[i], 3),
                             'ISO 20685:2010': ISO_20685_2010[i],
                             'ISO/TC133(garment, virtual human)': TC133[i]})
        writer.writerow({'Parts': ' ', 'Real(cm)': ' ',
                         'Predict(cm)': ' ', 'Error(cm)': ' ',
                         'Error%': ' ',
                         'ISO 20685:2010': 'https://www.iso.org/standard/63260.html ',
                         'ISO/TC133(garment, virtual human)': ' https://www.iso.org/committee/52374/x/catalogue/'})

def write_predict_measure(measurement , csv_dir):

    parts_name = list(measurement.keys())

    PD = np.array(list(measurement.values()))
    with open(csv_dir, 'w+') as csvfile:
        fieldnames = ['Parts', 'Predict(cm)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(parts_name)):
            writer.writerow({'Parts': parts_name[i],
                             'Predict(cm)': round(PD[i] * 100, 3)})
        writer.writerow({'Parts': ' ',
                         'Predict(cm)': ' '})

def write_result_25(name, GT, PD):
    PD = np.array(PD)
    GT = np.array(GT)
    csv_dir = ROOT_DIR + '/data/outputs/' + name + '.csv'
    parts_name = ['Cao', 'Vong_dau', 'Vong_co', 'Ngang_vai', 'Lung_ngang_nach', 'Nguc_ngang_nach', 'Dai_ao',
                  'Dai_tay_co_sau',
                  'Dai_tay_dai', 'Dai_tay_ngan', 'Vong_bap_tay', 'Vong_nach', 'Ha_chiet_nguc',
                  'Dang_nguc', 'Ha_eo_sau', 'Ha_eo_truoc', 'Vong_nguc_nach', 'Vong_nguc', 'Vong_eo', 'Vong_bung',
                  'Vong_mong', 'Vong_dui', 'Dai_chan', 'Ha_goi', 'Day_quan']
    Er_ = PD - GT
    with open(csv_dir, 'w') as csvfile:
        fieldnames = ['Parts', 'Real(cm)', 'Optim(cm)', 'Error_Optim(cm)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(25):
            writer.writerow(
                {'Parts': parts_name[i], 'Real(cm)': round(GT[i] * 100, 3), 'Optim(cm)': round(PD[i] * 100, 3),
                 'Error_Optim(cm)': round(abs(Er_[i]), 3) * 100})


if __name__ == '__main__':
    name = 'A_canh'
    write_result(name, GT, PD)
