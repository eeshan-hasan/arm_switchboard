from .switchboard import Switch

Switches_Strength_1 = [
    Switch(name='attention_update_type', n_variants=5, variant=('none','loss','p_regularization_2','p_regularization_1','sum_to_constant'), human_readable=('None','Loss-based','P-Regularization p=2','P-Regularization p=1',"Sum to Constant"), default=0),
    Switch(name='w_update_type', n_variants=2, variant=('hebbian','prediction_error'), human_readable=('Hebbian','Prediction Error'), default=0),
    Switch(name='decision_rule', n_variants=1, variant=('softmax',), human_readable=('Softmax'), default=0)
]

Switches_Strength_2 = [
    Switch(name='attention_update_type', n_variants=7, variant=('none','loss','p_regularization_2','p_regularization_1','p_regularization_0.75','p_regularization_0.5','sum_to_constant'), human_readable=('None','Loss-based','P-Regularization p=2','P-Regularization p=1','P-Regularization p=0.75','P-Regularization p=0.5',"Sum to Constant"), default=0),
    Switch(name='w_update_type', n_variants=2, variant=('hebbian','prediction_error'), human_readable=('Hebbian','Prediction Error'), default=0),
    Switch(name='decay', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0)
]


Switches_Strength_3 = [
    Switch(name='attention_update_type', n_variants=6, variant=('none','loss','p_regularization_2','p_regularization_1','p_regularization_0.75','p_regularization_0.5','sum_to_constant'), human_readable=('None','Loss-based','P-Regularization p=2','P-Regularization p=1','P-Regularization p=0.75','P-Regularization p=0.5',"Sum to Constant"), default=0),
    Switch(name='attention_parameterization', n_variants=2, variant=('sigmoid','none'), human_readable=('Sigmoid','None'), default=0),
    Switch(name='initial_alpha', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='w_update_type', n_variants=2, variant=('prediction_error','rescorla_wagner'), human_readable=('Prediction Error','Rescorla-Wagner'), default=0),
    Switch(name='delta', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='decay', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='guessing', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=1),
    Switch(name='loss_derivative', n_variants=2, variant=('ce','sse'), human_readable=('Cross-Entropy','Sum of Squared Errors'), default=0),
    Switch(name='alpha_clip', n_variants=2, variant=((-4,3),(-1000,1000)), human_readable=('Clip Alpha','No Clipping'), default=1),
    Switch(name='decision_rule', n_variants=2, variant=('luce','softmax'), human_readable=('Luce','Softmax'), default=0)
]