from .switchboard import Switch
Switches = [
    Switch(name='attention_update_type', n_variants=6, variant=('none','loss','p_regularization_2','p_regularization_1','p_regularization_0.75','p_regularization_0.5'), human_readable=('None','Loss-based','P-Regularization p=2','P-Regularization p=1','P-Regularization p=0.75','P-Regularization p=0.5'), default=0),
    Switch(name='attention_parameterization', n_variants=2, variant=('sigmoid','none'), human_readable=('Sigmoid','None'), default=0),
    Switch(name='initial_alpha', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='w_update', n_variants=1, variant=('perfect_instances',), human_readable=('Perfect Instances',), default=0),
    Switch(name='delta', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='decay', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=0),
    Switch(name='guessing', n_variants=2, variant=('fit_to_data','default'), human_readable=('Fit to Data','Default'), default=1),
    Switch(name='partial_encoding', n_variants=2, variant=(True,False), human_readable=('Partial Encoding','No Partial Encoding'), default=0),
    Switch(name='loss_derivative', n_variants=2, variant=('ce','sse'), human_readable=('Cross-Entropy','Sum of Squared Errors'), default=0),
    Switch(name='attention_update_dims', n_variants=2, variant=('all','single'), human_readable=('All Dimensions','Single Dimension'), default=0),
    Switch(name='initialization', n_variants=2, variant=('point','grid'), human_readable=('Point','Grid'), default=1),
    Switch(name='alpha_clip', n_variants=2, variant=((-4,3),(-1000,1000)), human_readable=('Clip Alpha','No Clipping'), default=1)
]