import argparse
import os
import sys

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
import random

import nltk
import numpy as np
import tensorflow as tf
from BILSTM_CRF_BERT_model import BERTCRFModel
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import (aggregate_report_pkl, get_char_indices,
                   get_char_to_index_dict, get_pred_and_ground_string,
                   get_tag_to_index_mappings, get_word_to_index_mappings,
                   read_data_for_task, read_folds, save_report_to_file)

#kept for future use. only the food-classification task can be executed in this notebook
ds_tag_values = {
    'food-classification': ['B-FOOD', 'I-FOOD', 'O', 'PAD'],
    "hansard-closest": ['B-AG.01.n.04', 'I-AG.01.h.01.e', 'B-AF.12', 'I-AG.01.l.02', 'B-AG.01.d.07', 'B-AG.01.y.01', 'I-AG.01.p', 'B-AG.01.h.01.d', 'I-AG.01.y.01', 'B-AE.08.i', 'B-AG.01.h.02', 'B-AG.01.n.15', 'B-AG.01.g', 'I-AF.10.i', 'B-AG.01.h.01', 'B-AG.01.f', 'B-AG.01.ae.03', 'I-AG.01.e.02', 'I-AG.01.t.06', 'I-AG.01.t.05', 'B-AG.01.z', 'I-AG.01.n.09', 'B-AG.01', 'I-AG.01.l.03', 'B-AF.28', 'I-AG.01.t.08', 'I-AG.01.g', 'I-AG.01.h.01.c', 'I-AG.01.d.03', 'I-AG.01.h', 'B-AF.20.c', 'I-AG.01.t.07', 'B-AG.01.y.01.e', 'B-AF.20.b', 'B-AF.20.h', 'I-AG.01.j', 'B-AG.01.l.04', 'B-AG.01.n.02', 'I-AG.01.h.01', 'B-AG.01.h.01.b', 'I-AG.01', 'B-AG.01.h', 'B-AE.13.h.01', 'B-AG.01.h.02.e', 'B-AG.01.e.02', 'B-AG.01.h.01.a', 'B-AG.01.t.07', 'I-AG.01.l', 'B-AG.01.ad', 'O', 'I-AG.01.l.04', 'I-AG.01.o', 'I-AG.01.n.13', 'B-AG.01.t.05', 'B-AG.01.h.02.i', 'B-AG.01.n.16', 'I-AG.01.n.15', 'I-AG.01.n.03', 'I-AG.01.n.02', 'I-AG.01.m', 'B-AG.01.h.02.f', 'B-AG.01.n.13', 'B-AG.01.y.01.c', 'B-AG.01.d.04', 'I-AG.01.i', 'B-AG.01.h.01.f', 'I-AG.01.y.01.c', 'B-AG.01.ae.01', 'B-AF.20.g', 'B-AG.01.e', 'I-AG.01.h.02', 'I-AG.01.k', 'B-AE.10', 'B-AG.01.l.02', 'B-AG.01.h.02.h', 'B-AG.01.h.02.d', 'B-AG.01.n.14', 'I-AG.01.y.01.a', 'B-AG.01.d.05', 'I-AG.01.e', 'I-AG.01.h.02.e', 'B-AG.01.y.01.a', 'B-AG.01.i', 'I-AF.20.b', 'I-AG.01.n.11', 'B-AG.01.l.01', 'B-AG.01.h.01.e', 'B-AG.01.p', 'B-AG.01.y.01.b', 'I-AG.01.h.02.h', 'B-AG.01.j', 'B-AF.10.i', 'I-AF.02.a', 'B-AG.01.o', 'B-AG.01.d.02', 'I-AG.01.d.02', 'B-AG.01.m', 'B-AG.01.k', 'B-AG.01.d.03', 'B-AG.01.h.02.g', 'I-AG.01.n', 'I-AG.01.h.02.f', 'B-AG.01.h.02.b', 'I-AG.01.l.01', 'I-AG.01.z', 'I-AG.01.n.06', 'B-AG.01.d.06', 'I-AG.01.d.07', 'I-AG.01.ad', 'I-AG.01.h.02.i', 'I-AG.01.d.06', 'I-AG.01.n.14', 'B-AG.01.n', 'I-AG.01.e.01', 'B-AG.01.n.09', 'B-AE.08', 'I-AG.01.f', 'I-AG.01.h.02.b', 'B-AG.01.n.17.a', 'B-AG.01.ab', 'I-AG.01.h.02.c', 'I-AG.01.ab', 'I-AG.01.h.02.g', 'B-AG.01.h.02.c', 'B-AG.01.n.01', 'B-AG.01.t.08', 'B-AG.01.y.01.g', 'I-AF.20.g', 'I-AG.01.n.17.a', 'B-AG.01.ag', 'I-AG.01.n.05', 'B-AG.01.l.03', 'B-AG.01.n.05', 'I-AG.01.n.01', 'I-AG.01.h.01.b', 'B-AG.01.n.03', 'I-AG.01.ae.01', 'B-AG.01.y.01.f', 'I-AG.01.h.01.a', 'B-AG.01.u', 'B-AG.01.h.02.a', 'B-AG.01.h.01.c', 'B-AE.10.g', 'B-AF.20.e', 'B-AF.13', 'I-AG.01.h.02.a', 'B-AG.01.ac', 'B-AG.01.n.11', 'I-AG.01.y.01.g', 'B-AG.01.t.06', 'B-AG.01.e.01', 'I-AG.01.n.12', 'B-AG.01.d', 'B-AG.01.n.12', 'B-AF.02.a', 'I-AG.01.ac', 'I-AG.01.h.01.f', 'B-AG.01.n.06', 'I-AG.01.u', 'B-AG.01.l', 'I-AG.01.ag', 'I-AG.01.d.05', 'I-AG.01.ae.03', 'PAD'],
    "hansard-parent": ['B-AG.01.a', 'B-AG.01.i', 'B-AF.12', 'B-AG.01.r', 'B-AE.08', 'I-AG.01.l', 'I-AG.01.f', 'B-AG.01.w', 'B-AF.10', 'B-AG.01.p', 'I-AG.01.r', 'I-AG.01.p', 'B-AG.01.t', 'O', 'B-AG.01.ad', 'I-AG.01.o', 'B-AG.01.j', 'B-AG.01.ae', 'B-AG.01.aa', 'B-AG.01.ab', 'B-AE.12', 'I-AG.01.ab', 'B-AG.01.g', 'I-AG.01.t', 'B-AG.01.f', 'I-AE.13', 'B-AG.01.h', 'I-AG.01.aa', 'B-AG.01.b', 'B-AG.01.z', 'B-AE.13', 'I-AG.01.m', 'B-AG.01.o', 'B-AG.01', 'B-AG.01.y', 'I-AF.02', 'B-AF.28', 'B-AG.01.m', 'I-AG.01.g', 'I-AG.01.i', 'B-AG.01.k', 'B-AF.02', 'I-AG.01.n', 'I-AG.01.y', 'I-AE.12', 'B-AG.01.e', 'I-AG.01.k', 'B-AE.10', 'I-AG.01.h', 'I-AG.01.ae', 'I-AF.10', 'I-AG.01.z', 'I-AG.01.a', 'B-AG.01.d', 'I-AE.10', 'I-AG.01.j', 'I-AG.01.ad', 'B-AF.20', 'I-AG.01', 'B-AG.01.l', 'I-AG.01.e', 'I-AG.01.d', 'B-AG.01.n', 'PAD'],
    "foodon": ['B-NCBITaxon_49992', 'B-FOODON_03310272', 'I-NCBITaxon_4006', 'I-FOODON_03315188', 'I-hancestro_0383', 'I-CHEBI_24866', 'I-FOODON_03430137', 'I-FOODON_03305680', 'B-FOODON_03311869', 'B-FOODON_03307668', 'B-ENVO_00002006', 'B-FOODON_03302772', 'B-NCBITaxon_4530', 'I-FOODON_03302034', 'I-FOODON_03310273', 'B-FOODON_03530217', 'B-FOODON_03302904', 'B-FOODON_03310351', 'B-FOODON_03411328', 'B-FOODON_03304042', 'B-FOODON_03301072', 'I-ENVO_00002006', 'B-ENVO_01001125', 'I-FOODON_03315835', 'B-FOODON_03315597', 'B-FOODON_03311146', 'I-FOODON_03302908', 'I-FOODON_03315597', 'I-FOODON_03302458', 'B-FOODON_03303380', 'O', 'B-FOODON_03315647', 'B-FOODON_03301008', 'I-NCBITaxon_3888', 'B-FOODON_03307663', 'I-NCBITaxon_4530', 'B-NCBITaxon_16718', 'B-FOODON_03317455', 'I-NCBITaxon_4071', 'I-FOODON_03304010', 'B-NCBITaxon_4113', 'B-FOODON_03301217', 'B-FOODON_03301051', 'I-FOODON_03530020', 'I-NCBITaxon_381124', 'I-FOODON_03301614', 'I-FOODON_00002087', 'B-FOODON_03303659', 'B-NCBITaxon_22663', 'B-FOODON_03301614', 'I-NCBITaxon_34199', 'B-FOODON_03316070', 'I-FOODON_03301244', 'I-FOODON_03302835', 'I-FOODON_03315272', 'B-FOODON_03530021', 'B-FOODON_03307808', 'I-FOODON_03307280', 'I-FOODON_03303578', 'B-FOODON_03301505', 'I-FOODON_03301072', 'B-FOODON_03305236', 'B-CHEBI_24866', 'B-FOODON_03306616', 'I-FOODON_03301008', 'B-FOODON_03430137', 'I-NCBITaxon_59895', 'B-NCBITaxon_39350', 'B-FOODON_03305417', 'B-FOODON_03303508', 'I-FOODON_03420157', 'I-ENVO_01001125', 'I-FOODON_03317654', 'B-FOODON_03309491', 'B-FOODON_03315872', 'I-CHEBI_60004', 'I-FOODON_03303886', 'I-FOODON_03315872', 'B-FOODON_03310185', 'B-CHEBI_83163', 'I-UBERON_0001913', 'B-FOODON_03301441', 'B-NCBITaxon_13450', 'B-FOODON_03316347', 'I-FOODON_00001926', 'I-NCBITaxon_89151', 'I-FOODON_03301705', 'I-FOODON_03411044', 'B-FOODON_03305639', 'B-FOODON_03411269', 'I-FOODON_03301577', 'I-NCBITaxon_3755', 'I-NCBITaxon_22663', 'B-FOODON_03301842', 'I-FOODON_03302897', 'B-FOODON_03303578', 'I-FOODON_03301842', 'I-FOODON_03411669', 'B-NCBITaxon_4071', 'B-NCBITaxon_6563', 'B-FOODON_03305263', 'I-FOODON_03411335', 'B-NCBITaxon_4682', 'B-FOODON_03309457', 'I-FOODON_03302060', 'I-NCBITaxon_37656', 'I-FOODON_03301128', 'I-FOODON_03310689', 'I-FOODON_03305159', 'B-NCBITaxon_6566', 'B-UBERON_0001913', 'I-PO_0009001', 'I-NCBITaxon_4565', 'B-NCBITaxon_80379', 'I-FOODON_03310272', 'B-NCBITaxon_63459', 'I-NCBITaxon_9031', 'B-FOODON_00001287', 'B-FOODON_03301671', 'I-FOODON_03315647', 'B-NCBITaxon_3562', 'B-FOODON_03306766', 'B-FOODON_03305518', 'I-FOODON_03309457', 'I-NCBITaxon_29780', 'I-FOODON_03315498', 'I-FOODON_03307062', 'B-FOODON_03306867', 'B-NCBITaxon_3760', 'B-FOODON_03302458', 'B-FOODON_03301256', 'I-PATO_0000386', 'B-FOODON_03301630', 'B-FOODON_03301632', 'B-FOODON_03301244', 'B-UBERON_0002113', 'B-FOODON_03302713', 'I-NCBITaxon_4682', 'B-FOODON_03305428', 'B-FOODON_03307312', 'I-FOODON_03302111', 'B-CHEBI_33290', 'I-FOODON_03309832', 'I-FOODON_03316347', 'B-FOODON_03301585', 'I-FOODON_03301660', 'B-FOODON_03301397', 'B-FOODON_03306347', 'B-FOODON_03309554', 'B-FOODON_03316284', 'B-CHEBI_60004', 'I-FOODON_03307240', 'B-NCBITaxon_9031', 'B-NCBITaxon_4081', 'B-FOODON_03302946', 'I-FOODON_03301671', 'B-NCBITaxon_59895', 'I-FOODON_03317068', 'I-NCBITaxon_3562', 'B-NCBITaxon_89151', 'B-NCBITaxon_94328', 'I-FOODON_03307668', 'B-NCBITaxon_4006', 'B-FOODON_03301605', 'B-PATO_0001985', 'I-FOODON_03310351', 'I-FOODON_03420108', 'B-FOODON_03301701', 'I-NCBITaxon_4615', 'B-FOODON_03305617', 'B-FOODON_03302908', 'B-FOODON_03301175', 'I-FOODON_03305003', 'B-FOODON_03303225', 'B-FOODON_03411335', 'B-FOODON_03317034', 'B-FOODON_03411237', 'B-NCBITaxon_3493', 'B-FOODON_03305086', 'I-FOODON_03305428', 'B-FOODON_03302034', 'B-PO_0009001', 'I-FOODON_03301585', 'B-FOODON_03301564', 'B-FOODON_03301189', 'I-FOODON_03310387', 'B-FOODON_03302515', 'B-NCBITaxon_4329', 'B-FOODON_03302897', 'I-FOODON_03301256', 'B-FOODON_03301240', 'I-FOODON_03315258', 'B-FOODON_03307240', 'B-FOODON_03301577', 'I-FOODON_03301889', 'B-FOODON_03310795', 'B-FOODON_03301329', 'B-FOODON_03301802', 'I-FOODON_03411328', 'I-FOODON_03305417', 'B-FOODON_03301844', 'I-NCBITaxon_51238', 'I-FOODON_03305086', 'B-FOODON_03411044', 'B-FOODON_03306160', 'B-FOODON_03317654', 'B-FOODON_03307280', 'B-FOODON_03530020', 'I-FOODON_03305617', 'I-FOODON_03306160', 'B-FOODON_03315258', 'B-FOODON_03310086', 'I-FOODON_03530021', 'B-FOODON_03307062', 'I-FOODON_03301051', 'B-FOODON_03301468', 'B-FOODON_00002087', 'B-FOODON_03414363', 'B-FOODON_03315146', 'B-FOODON_03411669', 'I-FOODON_03307312', 'B-FOODON_03317068', 'I-FOODON_03311146', 'B-UBERON_0002107', 'I-FOODON_03301116', 'I-FOODON_03301329', 'B-FOODON_03302060', 'I-FOODON_03303508', 'I-FOODON_03301505', 'B-FOODON_03301455', 'B-FOODON_03301128', 'B-UBERON_0007378', 'B-FOODON_03317294', 'B-NCBITaxon_3755', 'B-NCBITaxon_381124', 'I-FOODON_03301217', 'B-UBERON_0036016', 'B-FOODON_03301619', 'B-FOODON_03305159', 'B-FOODON_03304564', 'B-FOODON_03315188', 'I-GAZ_00000558', 'I-FOODON_03301455', 'I-FOODON_03303225', 'B-FOODON_03310760', 'B-ancestro_0354', 'B-FOODON_03301105', 'I-FOODON_03306867', 'B-NCBITaxon_34199', 'B-FOODON_03305680', 'I-NCBITaxon_4113', 'B-FOODON_03302111', 'I-FOODON_00001287', 'B-NCBITaxon_23211', 'B-NCBITaxon_29780', 'B-FOODON_03301660', 'B-PATO_0000386', 'B-FOODON_03301889', 'I-FOODON_03301710', 'B-FOODON_03301710', 'I-FOODON_03302904', 'I-UBERON_0002113', 'B-FOODON_03304704', 'B-hancestro_0383', 'B-NCBITaxon_3747', 'B-NCBITaxon_37656', 'B-NCBITaxon_3888', 'B-FOODON_03302835', 'I-FOODON_03317294', 'B-FOODON_03315272', 'B-FOODON_03310290', 'I-FOODON_03311869', 'I-FOODON_03301441', 'B-NCBITaxon_39352', 'I-NCBITaxon_4081', 'B-FOODON_03301126', 'B-FOODON_03420108', 'B-FOODON_03302062', 'I-FOODON_03306766', 'I-FOODON_03304564', 'B-NCBITaxon_4615', 'I-FOODON_03430168', 'B-FOODON_03309462', 'I-FOODON_03316042', 'I-ancestro_0354', 'I-FOODON_03301304', 'I-FOODON_03309462', 'B-FOODON_03315025', 'I-FOODON_03302515', 'I-FOODON_03310795', 'B-FOODON_03315259', 'I-NCBITaxon_4039', 'B-FOODON_03420157', 'B-FOODON_03301440', 'B-GAZ_00000558', 'I-NCBITaxon_80379', 'B-FOODON_03430168', 'I-FOODON_03412974', 'B-NCBITaxon_117781', 'B-FOODON_03301116', 'B-NCBITaxon_32201', 'I-UBERON_0002107', 'B-FOODON_03315835', 'B-FOODON_03304010', 'B-NCBITaxon_3827', 'I-PATO_0001985', 'B-FOODON_03301304', 'B-NCBITaxon_51238', 'B-FOODON_03301672', 'B-FOODON_03315498', 'I-FOODON_03304042', 'B-NCBITaxon_4565', 'I-FOODON_03303659', 'B-FOODON_03310689', 'I-FOODON_03301605', 'I-FOODON_03301175', 'I-FOODON_03530217', 'B-FOODON_03301705', 'B-FOODON_03316042', 'B-NCBITaxon_4039', 'B-FOODON_03303886', 'B-FOODON_00001926', 'I-FOODON_03302772', 'I-FOODON_03310086', 'I-NCBITaxon_3747', 'I-NCBITaxon_3760', 'B-FOODON_03310387', 'B-FOODON_03305003', 'B-FOODON_03309832', 'I-FOODON_03301844', 'I-UBERON_0007378', 'I-FOODON_03301619', 'I-NCBITaxon_3827', 'I-FOODON_03305263', 'I-FOODON_03309554', 'B-FOODON_03310273', 'I-NCBITaxon_4329', 'B-FOODON_03302775', 'I-FOODON_03303380', 'B-FOODON_03412974', 'I-FOODON_03302062', 'I-FOODON_03301701', 'PAD'],
    "snomedct": ['I-442681000124105', 'B-227592002', 'B-227757007', 'I-735030001', 'I-226551004', 'B-226855000', 'B-227418000', 'I-129559002', 'B-226562004', 'I-226726003', 'B-227218003', 'B-227239005', 'I-227219006', 'I-227215000', 'I-226928005', 'B-227411006', 'B-256443002', 'B-226888007', 'B-226863004', 'I-28647000', 'B-256307007', 'I-735248001', 'B-227282006', 'I-412065005', 'B-227549007', 'B-735336002', 'B-22836000', 'B-226041007', 'B-419420009', 'B-226559002', 'B-256354006', 'I-412066006', 'B-735249009', 'I-102264005', 'B-412061001', 'B-53410008', 'I-226838004', 'I-260184002', 'B-24515005', 'B-226587006', 'I-226855000', 'B-226802006', 'B-102262009', 'I-226802006', 'B-229908005', 'B-227436000', 'I-227519005', 'B-255621006', 'B-735009005', 'B-226493000', 'I-256350002', 'B-227362005', 'B-735049002', 'I-70813002', 'B-735106000', 'B-227400003', 'B-226639005', 'B-226849005', 'B-735050002', 'B-226753004', 'B-35748005', 'B-735048005', 'B-442891000124107', 'B-227407000', 'B-735248001', 'B-227430006', 'I-227282006', 'I-102262009', 'B-16313001', 'B-227553009', 'I-226849005', 'B-227612008', 'I-227436000', 'B-256329006', 'I-102261002', 'I-901000161107', 'I-226493000', 'I-226519000', 'I-226041007', 'I-735045008', 'I-444021000124105', 'I-443981000124106', 'B-226496008', 'I-227553009', 'B-9424004', 'I-735049002', 'I-226559002', 'I-67990004', 'B-226735005', 'B-736031006', 'B-256313003', 'O', 'I-226942002', 'I-226888007', 'B-227395004', 'B-256442007', 'B-227566009', 'B-420823005', 'I-256307007', 'B-41834005', 'I-735047000', 'B-734881000', 'I-412061001', 'B-226890008', 'B-29263009', 'B-226519000', 'I-67324005', 'I-63766005', 'B-226723006', 'I-227395004', 'I-230055000', 'B-256350002', 'B-412071004', 'I-41834005', 'I-9424004', 'I-226756007', 'B-227365007', 'B-735047000', 'B-735040003', 'I-227260004', 'I-256329006', 'B-226057007', 'B-227350006', 'B-227219006', 'B-735010000', 'B-226928005', 'B-412066006', 'B-227545001', 'B-70813002', 'B-412062008', 'B-278840001', 'B-227501001', 'B-226756007', 'B-735215001', 'B-227421003', 'B-226543002', 'B-226725004', 'I-443701000124100', 'I-89707004', 'I-734881000', 'I-226496008', 'B-735245003', 'B-227423000', 'I-735214002', 'B-443701000124100', 'B-736159005', 'B-226064009', 'I-53410008', 'B-444021000124105', 'B-227390009', 'B-735123009', 'B-226901000', 'B-226038003', 'B-260184002', 'B-227449005', 'B-89707004', 'B-227382009', 'I-227020009', 'B-226749001', 'B-443691000124100', 'I-226853007', 'B-226483007', 'B-227598003', 'I-227607007', 'I-278840001', 'I-227545001', 'B-226528004', 'B-226838004', 'B-226814003', 'B-226942002', 'I-72511004', 'B-256326004', 'B-412070003', 'I-412071004', 'I-227449005', 'I-227430006', 'B-226853007', 'B-328685004', 'I-229944000', 'B-227607007', 'I-229862008', 'B-28647000', 'I-442361000124108', 'I-226837009', 'I-35748005', 'B-226551004', 'I-230053007', 'B-72511004', 'B-227425007', 'I-226054000', 'I-762952008', 'B-608772009', 'B-226492005', 'I-226038003', 'B-735108004', 'I-735009005', 'B-443981000124106', 'B-735030001', 'I-444001000124100', 'B-226018004', 'B-226704004', 'I-227444000', 'B-412065005', 'B-226031009', 'B-735053000', 'B-442581000124106', 'B-412357001', 'B-227408005', 'B-227388008', 'I-226639005', 'B-226733003', 'I-412357001', 'I-256442007', 'B-229862008', 'B-227515004', 'I-256326004', 'I-419420009', 'B-227410007', 'B-227413009', 'B-226719003', 'I-713648000', 'B-226916002', 'B-735045008', 'I-29263009', 'B-227519005', 'I-443691000124100', 'B-226934003', 'B-227150003', 'B-102264005', 'B-230055000', 'B-226498009', 'I-226498009', 'I-22836000', 'B-227463004', 'I-226934003', 'B-256317002', 'I-226916002', 'B-256319004', 'I-226787009', 'B-608773004', 'B-442341000124109', 'B-63766005', 'I-16313001', 'I-256313003', 'B-901000161107', 'I-227592002', 'I-226647005', 'B-226604005', 'B-226726003', 'B-226769006', 'I-227362005', 'B-227215000', 'I-226831005', 'B-229948002', 'I-226516007', 'I-735215001', 'B-230053007', 'B-226019007', 'B-13577000', 'I-735053000', 'I-735340006', 'B-442751000124107', 'I-256354006', 'I-256443002', 'B-226837009', 'B-227689008', 'B-226054000', 'I-227757007', 'B-226516007', 'B-442681000124105', 'I-328685004', 'B-102261002', 'I-226753004', 'I-735211005', 'B-227260004', 'I-412070003', 'B-226647005', 'B-227020009', 'B-735042006', 'B-762952008', 'I-735042006', 'B-226787009', 'I-255621006', 'B-713648000', 'B-229887001', 'I-735245003', 'B-444001000124100', 'I-227501001', 'B-129559002', 'I-227400003', 'B-67990004', 'B-735211005', 'I-226814003', 'I-226749001', 'B-227722009', 'I-226955001', 'B-51905005', 'B-735340006', 'B-442861000124104', 'B-226831005', 'B-229944000', 'B-227444000', 'I-227612008', 'I-226543002', 'B-444161000124100', 'B-226955001', 'B-735213008', 'I-227388008', 'B-442811000124102', 'I-736159005', 'B-67324005', 'B-226740002', 'B-735214002', 'B-442361000124108', 'I-442341000124109', 'I-226492005', 'B-23182003', 'B-227606003', 'PAD']
}

def example_to_features(input_ids,attention_masks,label_ids):
  return {"input_ids": input_ids, "attention_mask": attention_masks}, label_ids


# tasks = ['food-classification']

stored_models ={
  'bert':'bert-base-cased', 
  #'bioBert-standard': '/content/drive/My Drive/Colab Notebooks/data/biobert',
  #'bioBert-large': '/content/drive/My Drive/Colab Notebooks/data/biobert_large'
}

def NER_driver(fold = None):

    max_sentence_length = 50
    EPOCHS = 1000
    BATCH_SIZE = 128
    vectorizer_model_name = 'bert'
    missing_values_handled = False
    pre_proc = "none"
    task_name = "food-classification"
    nn_model_name = "BILSTM_BERT"
    task = 'food-classification'
    model_prefix = 'bert'
    load_model = stored_models[model_prefix]

    # seeds
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    model_instance = BERTCRFModel()

    if fold == None:
        full_data, train_data, test_data = read_data_for_task(task_name)
        print(full_data.shape)
        print(train_data.shape)
        print(test_data.shape)
    else:
        full_data, train_data, test_data = read_folds(task_name, fold)
        print(full_data.shape)
        print(train_data.shape)
        print(test_data.shape)
  
    word2idx, idx2word, n_words, words = get_word_to_index_mappings(full_data)
    tag2idx, idx2tag, n_tags, tags = get_tag_to_index_mappings(full_data)
    char2idx, idx2char, n_chars, chars = get_char_to_index_dict(words)

    tr_tags, _ = model_instance.process_Y(train_data, tag2idx, max_sentence_length, n_tags)
    te_tags, _ = model_instance.process_Y(test_data, tag2idx, max_sentence_length, n_tags)

    tr_inputs, tr_masks = model_instance.process_X(train_data, word2idx, max_sentence_length, tag2idx)
    te_inputs, te_masks = model_instance.process_X(test_data, word2idx, max_sentence_length, tag2idx)


    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tr_inputs, tr_tags, random_state=seed, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(tr_masks, tr_masks, random_state=seed, test_size=0.1)

    train_ds = tf.data.Dataset.from_tensor_slices((tr_inputs, tr_masks, tr_tags)).map(example_to_features).batch(BATCH_SIZE)
    # print(dir(train_ds))
    # print(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices((te_inputs, te_masks, te_tags)).map(example_to_features)#.shuffle(buffer_size=1000).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_masks, val_tags)).map(example_to_features).batch(BATCH_SIZE)

    # sys.exit()
    cbks = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 5e-3)

    model = model_instance.get_compiled_model(load_model, n_tags)
    # history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1, batch_size=BATCH_SIZE, callbacks = [cbks])
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1, callbacks = [cbks])

    preds = model.predict(te_inputs)
    preds_str, ground_str = get_pred_and_ground_string(Y_test=te_tags, predictions=preds, idx2tag=idx2tag)

    assert len(preds_str) == len(ground_str)
    
    report = classification_report(ground_str, preds_str, output_dict=True)
    if fold != None:
        if vectorizer_model_name != "lexical":
            report_file_name_txt = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_fold={fold}_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_fold={fold}_res.pkl"
        else:
            report_file_name_txt = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_fold={fold}_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_fold={fold}_res.pkl"
        save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_txt, n_epochs = len(history.history['loss']), which_fold = fold, nn_model_name = nn_model_name)
        aggregate_report_pkl(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_pkl, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)

    else:
        if vectorizer_model_name != "lexical":
            report_file_name_txt = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_earlystop_res.pkl"
        else:
            report_file_name_txt = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_res.txt"
            report_file_name_pkl = f"{task_name}_{pre_proc}_e{EPOCHS}_earlystop_res.pkl"
        save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_txt, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)
        aggregate_report_pkl(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name_pkl, n_epochs = len(history.history['loss']), nn_model_name = nn_model_name)

    print(report_file_name_txt)


if __name__ == "__main__":
    tf.config.list_physical_devices('TPU')

    print("\n-----TRAIN START-----\n")
    parser = argparse.ArgumentParser(prog = "CRF_train")
    parser.add_argument("--fold", type = int, dest = "which_fold", default = None)
    args = parser.parse_args()    

    NER_driver(args.which_fold)
