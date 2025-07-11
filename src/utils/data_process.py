import pandas as pd 
def data_process(data_path):

    df = pd.read_hdf(data_path)
    df['rank'] = df['genus'] - df['genus_minus_rank']
    df['log_conductor'] = pd.to_numeric(df['log_conductor'], errors='coerce')
    print('df.shape', df.shape)

    import ast 
    def ensure_list(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return list(x)
    df['canonical_conjugator'] = df['canonical_conjugator'].map(ensure_list)
    for i in range(4):
        df['canonical_conjugator_{}'.format(i)] = df['canonical_conjugator'].apply(lambda x: x[i] if len(x) > i else None)
    df['conductor'] = df['conductor'].map(ensure_list)

    df['q_gonality_bounds'] = df['q_gonality_bounds'].map(ensure_list)
    # First seperate data based on q_gonality bounds, this will test how many predication on real bound 

    q_gonality_same = df[df['q_gonality_bounds'].map(lambda b: b[0] == b[1])]

    q_gonality_diff = df[df['q_gonality_bounds'].map(lambda b: b[0] != b[1])]
    # display two datas 
    #q_gonality_same.head()
    
    q_gonality_same['q_gonality']= q_gonality_same['q_gonality_bounds'].map(lambda b: b[0])

    return q_gonality_same, q_gonality_diff




    
