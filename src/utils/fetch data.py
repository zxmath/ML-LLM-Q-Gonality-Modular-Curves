import  load_data_2
import os
'''
base = os.path.dirname(__file__)             # src/
data_path = os.path.normpath(os.path.join(base, "..", "data", "_DataForularCurves3"))
print("cwd =", os.getcwd())
print("data_path exists?", os.path.exists(data_path))
print("files in data_path:", os.listdir(data_path))
'''
#feature_2 = ['id','level','nu2','nu3', 'psl2index', 'qbar_gonality', 'has_obstruction', 'pointless',
 #            'cusps','rational_cusps', 'num_bad_primes','genus', 
 #            'log_conductor','orbits','generators','canonical_generators',
  #           'isogeny_orbits','dims','mults']
feature_2 = ['id','level','nu2','nu3', 'psl2index', 'qbar_gonality', 'has_obstruction','pointless',
             'cusps','rational_cusps', 'num_bad_primes','genus','canonical_conjugator','conductor','log_conductor',
'coarse_class_num','coarse_level']
categorical_2 = ['contains_negative_one','level_is_squarefree']
target_2 = ['genus_minus_rank','q_gonality_bounds']
data_path=r'/home/zxmath/Machine_Learning_Arithmetic-_object/data/_DataForModularCurves3'
combined_data=load_data_2.combine_data_by_feature_target(feature= feature_2+categorical_2, target= target_2,data_path= data_path,range_start= 0,range_end= 1684)
#combined_data.to_p(r'/home/zxmath/Machine_Learning_Arithmetic-_object/data/combined_data_4.csv', index=False)
# save as high effciency data file h5
combined_data.to_hdf(r'/home/zxmath/Machine_Learning_Arithmetic-_object/data/combined_data_7.h5', key='df', index=False)