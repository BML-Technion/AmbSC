import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from folktables import BasicProblem, ACSDataSource
import torch


def get_folks(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ACSEmploymentCustom = BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'RELP',
            'ESP',
            'CIT',
            'MIL',
        ],
        target='ESR',
        target_transform=lambda x: x == 1,
        group='RAC1P',
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmploymentCustom.df_to_pandas(acs_data)
    X = pd.DataFrame(features)
    y = pd.Series(label.iloc[:, 0], dtype=int)
    y.loc[y == 0] = -1
    
    idx_pos = y[y == 1].index
    idx_neg = y[y == -1].index

    sampled_idx_pos = np.random.choice(idx_pos, 4500, replace=False)
    sampled_idx_neg = np.random.choice(idx_neg, 4500, replace=False)

    balanced_indices = np.concatenate([sampled_idx_pos, sampled_idx_neg])
    np.random.shuffle(balanced_indices)

    X_balanced = X.loc[balanced_indices].reset_index(drop=True)
    y_balanced = y.loc[balanced_indices].reset_index(drop=True)

    X_scaled = MinMaxScaler().fit_transform(X_balanced)
    
    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y_balanced.values).float()
    
    return X_tensor, y_tensor